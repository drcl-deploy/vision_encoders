#!/usr/bin/env python3
"""Combined viewer: raw frame + model-input frame + Theia embedding heatmap.

Subscribes to `/theia_embedding` and `/theia_input_frame` (the post-resize,
pre-normalize frame the ONNX model literally sees), and opens a TCP client
to the same stream the inference node reads for the raw frame. Three panels:

  - top:    raw_frame_* — what the camera/sim sends over TCP
  - middle: /theia_input_frame — model_input_* the node feeds to ONNX,
            upscaled with INTER_NEAREST so the pixelation is visible
  - bottom: per-patch embedding heatmap

Sizes are read from `config/theia_tiny.yaml` so viewer and node stay in
lockstep. Override the config path with `--config /path/to/theia_tiny.yaml`.

Offboard: pass `--camera-ip <orin_wlan0_ip>` to override the yaml's camera_ip
(the streamer must be running with `--bind 0.0.0.0` on the Orin).
"""

import argparse
import os
import socket
import struct
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import torch
import yaml
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

HEATMAP_H = 120
EMBED_DIM = 192
FPS_EMA_ALPHA = 0.1
OVERLAY_WINDOW = 'Theia Viewer'


def _label(img, text_or_lines):
    out = img.copy()
    lines = [text_or_lines] if isinstance(text_or_lines, str) else list(text_or_lines)
    for i, text in enumerate(lines):
        y = 22 + i * 22
        cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA)
    return out


def _ema(prev, sample, alpha=FPS_EMA_ALPHA):
    return sample if prev == 0.0 else (1 - alpha) * prev + alpha * sample


def _default_config_path():
    here = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(here, '..', 'config', 'theia_tiny.yaml'))


def _load_config(path):
    with open(path) as f:
        params = yaml.safe_load(f)['theia_tiny_node']['ros__parameters']
    return {
        'camera_ip': params['camera_ip'],
        'camera_port': int(params['camera_port']),
        'raw_frame_width': int(params['raw_frame_width']),
        'raw_frame_height': int(params['raw_frame_height']),
        'model_input_width': int(params['model_input_width']),
        'model_input_height': int(params['model_input_height']),
        'input_frame_topic': params.get('input_frame_topic', '/theia_input_frame'),
    }


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _load_fpv_dataset(dataset_dir: Path):
    """Load fpv_dataset.pt + metadata.yaml, return (first_frames_bgr, object_names).

    first_frames_bgr: list of BGR images (one per env), scaled later to raw feed size.
    """
    pt = torch.load(dataset_dir / 'fpv_dataset.pt', map_location='cpu', weights_only=False)
    with open(dataset_dir / 'metadata.yaml') as f:
        meta = yaml.safe_load(f)
    frames = pt['frames']  # (N, T, H, W, 3) uint8 RGB
    if torch.is_tensor(frames):
        frames = frames.numpy()
    first_frames = []
    for i in range(frames.shape[0]):
        rgb = frames[i, 0]  # first timestep
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        first_frames.append(bgr)
    names = meta.get('ordered_object_names', [f'env_{i}' for i in range(len(first_frames))])
    return first_frames, names


class TheiaViewer(Node):
    def __init__(self, cfg, overlay_frames=None, overlay_names=None):
        super().__init__('theia_viewer')
        self.cfg = cfg
        self.raw_frame = None
        self.model_input_frame = None
        self.heatmap_frame = None
        self._stop = threading.Event()

        self._cam_fps = 0.0
        self._cam_last_t = 0.0
        self._infer_hz = 0.0
        self._infer_last_t = 0.0
        self._tcp_ms = 0.0
        self._infer_ms = 0.0

        # Overlay state
        self._overlay_frames = overlay_frames or []
        self._overlay_names = overlay_names or []
        self._overlay_scaled = {}   # cache: (env_id, w, h) -> scaled BGR
        self._overlay_edges = {}    # cache: (env_id, w, h) -> edge mask BGR
        self._env_id = 0
        self._alpha = 40   # 0-100, trackbar units (divided by 100 for blend)
        self._mode = 0     # 0 = alpha blend, 1 = contour overlay

        self.create_subscription(
            Float32MultiArray, '/theia_embedding', self._on_embedding, 10)
        self.create_subscription(
            Float32MultiArray, '/theia_stats', self._on_stats, 10)
        self.create_subscription(
            Image, cfg['input_frame_topic'], self._on_input_frame, 10)
        self.create_timer(1.0 / 30.0, self._refresh)

        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._cam_thread.start()

    def _camera_loop(self):
        ip, port = self.cfg['camera_ip'], self.cfg['camera_port']
        raw_w, raw_h = self.cfg['raw_frame_width'], self.cfg['raw_frame_height']
        while not self._stop.is_set():
            try:
                with socket.create_connection((ip, port), timeout=5.0) as s:
                    self.get_logger().info(f'Camera TCP connected to {ip}:{port}')
                    while not self._stop.is_set():
                        hdr = _recv_exact(s, 4)
                        if hdr is None:
                            break
                        (length,) = struct.unpack('<I', hdr)
                        payload = _recv_exact(s, length)
                        if payload is None:
                            break
                        arr = np.frombuffer(payload, dtype=np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
                        if img is None:
                            continue
                        if img.shape[1] != raw_w or img.shape[0] != raw_h:
                            img = cv2.resize(img, (raw_w, raw_h))
                        self.raw_frame = img
                        now_t = time.monotonic()
                        if self._cam_last_t:
                            self._cam_fps = _ema(
                                self._cam_fps,
                                1.0 / max(now_t - self._cam_last_t, 1e-6))
                        self._cam_last_t = now_t
            except OSError as e:
                self.get_logger().warn(f'Camera TCP: {e}; retrying in 2s')
                self._stop.wait(2.0)

    def _on_input_frame(self, msg):
        # Node publishes rgb8; cv2.imshow expects BGR.
        if msg.encoding != 'rgb8':
            return
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)
        self.model_input_frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def _on_embedding(self, msg):
        vec = np.asarray(msg.data, dtype=np.float32)
        rows = vec.size // EMBED_DIM
        if rows == 0:
            return
        normed = cv2.normalize(vec.reshape(rows, EMBED_DIM), None, 0, 255,
                               cv2.NORM_MINMAX)
        self.heatmap_frame = cv2.applyColorMap(normed.astype(np.uint8),
                                               cv2.COLORMAP_HOT)
        now_t = time.monotonic()
        if self._infer_last_t:
            self._infer_hz = _ema(
                self._infer_hz, 1.0 / max(now_t - self._infer_last_t, 1e-6))
        self._infer_last_t = now_t

    def _on_stats(self, msg):
        if len(msg.data) >= 2:
            self._tcp_ms = float(msg.data[0])
            self._infer_ms = float(msg.data[1])

    def _get_overlay_scaled(self, env_id, w, h):
        """Return the overlay frame for env_id, scaled to (w, h). Cached."""
        key = (env_id, w, h)
        if key not in self._overlay_scaled:
            src = self._overlay_frames[env_id]
            self._overlay_scaled[key] = cv2.resize(src, (w, h),
                                                    interpolation=cv2.INTER_LINEAR)
        return self._overlay_scaled[key]

    def _get_overlay_edges(self, env_id, w, h):
        """Return edge contours of the overlay frame, as a colored BGR mask. Cached."""
        key = (env_id, w, h)
        if key not in self._overlay_edges:
            src = self._overlay_frames[env_id]
            # Upscale with INTER_CUBIC for smooth gradients before edge detection
            scaled = cv2.resize(src, (w, h), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
            # Blur to suppress upscale noise, then Canny
            gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
            edges = cv2.Canny(gray, 20, 80)
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
            # Cyan contour on black background
            mask = np.zeros((h, w, 3), dtype=np.uint8)
            mask[edges > 0] = (255, 255, 0)  # BGR cyan
            self._overlay_edges[key] = mask
        return self._overlay_edges[key]

    def _refresh(self):
        panels = []
        display_w = self.cfg['raw_frame_width']
        display_h = self.cfg['raw_frame_height']
        if self.raw_frame is not None:
            raw = self.raw_frame.copy()

            # Overlay sim frame if dataset loaded
            if self._overlay_frames and self._alpha > 0:
                eid = self._env_id
                a = self._alpha / 100.0
                mode_name = 'contour' if self._mode else 'blend'
                if self._mode == 1:
                    # Contour mode: draw edges, alpha controls contour brightness
                    edges = self._get_overlay_edges(eid, display_w, display_h)
                    cv2.addWeighted(edges, a, raw, 1.0, 0, dst=raw)
                else:
                    # Alpha blend mode
                    ovl = self._get_overlay_scaled(eid, display_w, display_h)
                    cv2.addWeighted(ovl, a, raw, 1.0 - a, 0, dst=raw)
                obj_name = self._overlay_names[eid] if eid < len(self._overlay_names) else f'env_{eid}'
                cv2.putText(raw, f'[{eid}] {obj_name}  {mode_name} {self._alpha}%',
                            (8, display_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 255), 2, cv2.LINE_AA)

            pass_ms = self._tcp_ms + self._infer_ms
            overlay_text = [
                f'raw  |  cam {self._cam_fps:5.1f} FPS  |  infer {self._infer_hz:5.1f} Hz',
                f'tcp {self._tcp_ms:5.1f} ms  +  infer {self._infer_ms:5.1f} ms  =  pass {pass_ms:5.1f} ms',
            ]
            panels.append(_label(raw, overlay_text))
        if self.model_input_frame is not None:
            up = cv2.resize(self.model_input_frame, (display_w, display_h),
                            interpolation=cv2.INTER_NEAREST)
            m_w, m_h = self.cfg['model_input_width'], self.cfg['model_input_height']
            panels.append(_label(up, f'/theia_input_frame ({m_w}x{m_h}, nearest-up)'))
        if self.heatmap_frame is not None:
            panels.append(cv2.resize(self.heatmap_frame, (display_w, HEATMAP_H),
                                     interpolation=cv2.INTER_NEAREST))
        if not panels:
            return
        cv2.imshow(OVERLAY_WINDOW, np.vstack(panels))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            raise SystemExit
        elif key == ord('m') and self._overlay_frames:
            self._mode = 1 - self._mode
            self._overlay_edges.clear()  # rebuild if params ever change

    def shutdown(self):
        self._stop.set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=_default_config_path())
    parser.add_argument('--camera-ip', default=None,
                        help='override yaml camera_ip (e.g. Orin wlan0 IP for offboard)')
    parser.add_argument('--dataset', type=Path, default=None,
                        help='fpv_dataset dir (fpv_dataset.pt + metadata.yaml) for overlay alignment')
    args, _ = parser.parse_known_args()

    cfg = _load_config(args.config)
    if args.camera_ip:
        cfg['camera_ip'] = args.camera_ip

    overlay_frames, overlay_names = None, None
    if args.dataset:
        print(f'[viewer] loading overlay dataset: {args.dataset}')
        overlay_frames, overlay_names = _load_fpv_dataset(args.dataset)
        print(f'[viewer]   {len(overlay_frames)} envs: {overlay_names}')

    rclpy.init()
    node = TheiaViewer(cfg, overlay_frames, overlay_names)

    # Create trackbars after the first imshow call creates the window
    if overlay_frames:
        # Need a dummy frame to create the window so trackbars attach
        dummy = np.zeros((cfg['raw_frame_height'], cfg['raw_frame_width'], 3), dtype=np.uint8)
        cv2.imshow(OVERLAY_WINDOW, dummy)
        cv2.createTrackbar('env_id', OVERLAY_WINDOW, 0, len(overlay_frames) - 1,
                           lambda v: setattr(node, '_env_id', v))
        cv2.createTrackbar('alpha%', OVERLAY_WINDOW, node._alpha, 100,
                           lambda v: setattr(node, '_alpha', v))
        print('[viewer] keys: m=toggle blend/contour, q=quit')
        cv2.waitKey(1)

    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        node.shutdown()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
