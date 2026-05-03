#!/usr/bin/env python3
"""Combined viewer: raw frame + model-input frame + Theia embedding heatmap.

Subscribes to `/theia_embedding` and opens a second TCP client to the same
JPEG stream the inference node reads. Shows two image panels:

  - top:    raw_frame_* — what the camera (or sim) actually sends
  - middle: model_input_* — same frame after the stage-2 resize, i.e. the
            pixels the ONNX model sees (before ImageNet normalization),
            upscaled with INTER_NEAREST so the pixelation is visible
  - bottom: per-patch embedding heatmap

Reads all sizes from `config/theia_tiny.yaml` so viewer and node stay in
lockstep. Override the config path with `--config /path/to/theia_tiny.yaml`.
"""

import argparse
import os
import socket
import struct
import threading
import time

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

HEATMAP_H = 120
EMBED_DIM = 192
FPS_EMA_ALPHA = 0.1


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
    }


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


class TheiaViewer(Node):
    def __init__(self, cfg):
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

        self.create_subscription(
            Float32MultiArray, '/theia_embedding', self._on_embedding, 10)
        self.create_subscription(
            Float32MultiArray, '/theia_stats', self._on_stats, 10)
        self.create_timer(1.0 / 30.0, self._refresh)

        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._cam_thread.start()

    def _camera_loop(self):
        ip, port = self.cfg['camera_ip'], self.cfg['camera_port']
        raw_w, raw_h = self.cfg['raw_frame_width'], self.cfg['raw_frame_height']
        m_w, m_h = self.cfg['model_input_width'], self.cfg['model_input_height']
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
                        self.model_input_frame = cv2.resize(img, (m_w, m_h))
                        now_t = time.monotonic()
                        if self._cam_last_t:
                            self._cam_fps = _ema(
                                self._cam_fps,
                                1.0 / max(now_t - self._cam_last_t, 1e-6))
                        self._cam_last_t = now_t
            except OSError as e:
                self.get_logger().warn(f'Camera TCP: {e}; retrying in 2s')
                self._stop.wait(2.0)

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

    def _refresh(self):
        panels = []
        display_w = self.cfg['raw_frame_width']
        display_h = self.cfg['raw_frame_height']
        if self.raw_frame is not None:
            pass_ms = self._tcp_ms + self._infer_ms
            overlay = [
                f'raw  |  cam {self._cam_fps:5.1f} FPS  |  infer {self._infer_hz:5.1f} Hz',
                f'tcp {self._tcp_ms:5.1f} ms  +  infer {self._infer_ms:5.1f} ms  =  pass {pass_ms:5.1f} ms',
            ]
            panels.append(_label(self.raw_frame, overlay))
        if self.model_input_frame is not None:
            up = cv2.resize(self.model_input_frame, (display_w, display_h),
                            interpolation=cv2.INTER_NEAREST)
            m_w, m_h = self.cfg['model_input_width'], self.cfg['model_input_height']
            panels.append(_label(up, f'model input ({m_w}x{m_h}, nearest-up)'))
        if self.heatmap_frame is not None:
            panels.append(cv2.resize(self.heatmap_frame, (display_w, HEATMAP_H),
                                     interpolation=cv2.INTER_NEAREST))
        if not panels:
            return
        cv2.imshow('Theia Viewer', np.vstack(panels))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise SystemExit

    def shutdown(self):
        self._stop.set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=_default_config_path())
    args, _ = parser.parse_known_args()

    cfg = _load_config(args.config)
    rclpy.init()
    node = TheiaViewer(cfg)
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
