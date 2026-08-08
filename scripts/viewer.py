#!/usr/bin/env python3
"""Generic encoder viewer: raw frame + model-input frame + token heatmap.

Model-agnostic — grid, dim, and the backbone tag all come from the
`ImageTokens` msg, never from config. Three panels:

  - top:    raw_frame_* — what the camera/sim sends over TCP (own TCP client)
  - middle: /enc/frame — the model-input frame the graph literally consumes,
            upscaled with INTER_NEAREST so the pixelation is visible
  - bottom: per-patch token heatmap (P rows x D cols)

Plumbing (camera ip/port, raw size, topics) is read from config/encoder.yaml
so viewer and node stay in lockstep; `--camera-ip` overrides for offboard use
(streamer must run with `--bind 0.0.0.0` on the Orin).
"""

import argparse
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from vision_encoders.msg import ImageTokens
import wire
import yaml

HEATMAP_H = 120
FPS_EMA_ALPHA = 0.1


def _label(img, text_or_lines):
    out = img.copy()
    lines = [text_or_lines] if isinstance(text_or_lines, str) else list(text_or_lines)
    for i, text in enumerate(lines):
        cv2.putText(out, text, (8, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA)
    return out


def _ema(prev, sample, alpha=FPS_EMA_ALPHA):
    return sample if prev == 0.0 else (1 - alpha) * prev + alpha * sample


def _default_config_path():
    here = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(here, '..', 'config', 'encoder.yaml'))


def _load_config(path):
    with open(path) as f:
        p = yaml.safe_load(f)['encoder_node']['ros__parameters']
    return {
        'camera_ip': p['camera_ip'],
        'camera_port': int(p['camera_port']),
        'raw_frame_width': int(p['raw_frame_width']),
        'raw_frame_height': int(p['raw_frame_height']),
        'tokens_topic': p.get('tokens_topic', '/enc/tokens'),
        'frame_topic': p.get('frame_topic', '/enc/frame'),
    }


class EncoderViewer(Node):
    """Display the camera, model input, and encoder tokens."""

    def __init__(self, cfg):
        super().__init__('encoder_viewer')
        self.cfg = cfg
        self.raw_frame = None
        self.input_frame = None
        self.heatmap = None
        self.tag = '?'
        self.grid = (0, 0)
        self.dim = 0
        self._stop = threading.Event()

        self._cam_fps = 0.0
        self._cam_last_t = 0.0
        self._tok_hz = 0.0
        self._tok_last_t = 0.0
        self._decode_ms = 0.0
        self._infer_ms = 0.0

        # The encoder may be reached through cdr_tcp_bridge, whose offboard
        # publishers intentionally use best-effort/latest-only QoS. A reliable
        # subscription is incompatible with a best-effort publisher, so make
        # the viewer's telemetry QoS explicit while retaining one fresh sample.
        telemetry_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            ImageTokens, cfg['tokens_topic'], self._on_tokens, telemetry_qos)
        self.create_subscription(Image, cfg['frame_topic'], self._on_frame, telemetry_qos)
        self.create_timer(1.0 / 30.0, self._refresh)

        threading.Thread(target=self._camera_loop, daemon=True).start()

    def _camera_loop(self):
        ip, port = self.cfg['camera_ip'], self.cfg['camera_port']
        raw_w, raw_h = self.cfg['raw_frame_width'], self.cfg['raw_frame_height']
        while not self._stop.is_set():
            try:
                with wire.connect(ip, port) as s:
                    self.get_logger().info(f'Camera TCP connected to {ip}:{port}')
                    while not self._stop.is_set():
                        img = wire.recv_frame(s)
                        if img is None:
                            break
                        if img.shape[1] != raw_w or img.shape[0] != raw_h:
                            img = cv2.resize(img, (raw_w, raw_h))
                        self.raw_frame = img
                        now_t = time.monotonic()
                        if self._cam_last_t:
                            self._cam_fps = _ema(
                                self._cam_fps, 1.0 / max(now_t - self._cam_last_t, 1e-6))
                        self._cam_last_t = now_t
            except OSError as e:
                self.get_logger().warn(f'Camera TCP: {e}; retrying in 2s')
                self._stop.wait(2.0)

    def _on_frame(self, msg):
        if msg.encoding != 'bgr8':
            return
        self.input_frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)

    def _on_tokens(self, msg):
        self.tag = msg.header.frame_id
        self.grid = (msg.grid_h, msg.grid_w)
        self.dim = msg.dim
        self._decode_ms = msg.decode_ms
        self._infer_ms = msg.infer_ms
        rows = msg.grid_h * msg.grid_w
        if rows == 0:
            return
        vec = np.asarray(msg.patches, dtype=np.float32).reshape(rows, msg.dim)
        normed = cv2.normalize(vec, None, 0, 255, cv2.NORM_MINMAX)
        self.heatmap = cv2.applyColorMap(normed.astype(np.uint8), cv2.COLORMAP_HOT)
        now_t = time.monotonic()
        if self._tok_last_t:
            self._tok_hz = _ema(self._tok_hz, 1.0 / max(now_t - self._tok_last_t, 1e-6))
        self._tok_last_t = now_t

    def _refresh(self):
        panels = []
        w = self.cfg['raw_frame_width']
        h = self.cfg['raw_frame_height']
        if self.raw_frame is not None:
            overlay = [
                f'{self.tag}  |  cam {self._cam_fps:5.1f} FPS  |  tokens {self._tok_hz:5.1f} Hz',
                f'decode {self._decode_ms:5.1f} ms + infer {self._infer_ms:5.1f} ms'
                f' = {self._decode_ms + self._infer_ms:5.1f} ms',
            ]
            panels.append(_label(self.raw_frame, overlay))
        if self.input_frame is not None:
            up = cv2.resize(self.input_frame, (w, h), interpolation=cv2.INTER_NEAREST)
            ih, iw = self.input_frame.shape[:2]
            panels.append(_label(up, f'/enc/frame ({iw}x{ih}, nearest-up)'))
        if self.heatmap is not None:
            gh, gw = self.grid
            hm = cv2.resize(self.heatmap, (w, HEATMAP_H), interpolation=cv2.INTER_NEAREST)
            panels.append(_label(hm, f'{gh * gw} patches x {self.dim} dim'))
        if not panels:
            return
        cv2.imshow('Encoder Viewer', np.vstack(panels))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise SystemExit

    def shutdown(self):
        self._stop.set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=_default_config_path())
    parser.add_argument('--camera-ip', default=None,
                        help='override yaml camera_ip (e.g. Orin wlan0 IP for offboard)')
    args, _ = parser.parse_known_args()

    cfg = _load_config(args.config)
    if args.camera_ip:
        cfg['camera_ip'] = args.camera_ip
    rclpy.init()
    node = EncoderViewer(cfg)
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
