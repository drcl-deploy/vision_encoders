#!/usr/bin/env python3
"""Combined viewer: head camera (exact model input) + Theia embedding heatmap.

Subscribes to `/theia_embedding` for the latents and opens a second TCP client
to the same JPEG stream the inference node reads, so what you see is the
exact BGR frame after `cv2.resize(cam_width, cam_height)` — i.e. the pixels
that enter the model (before per-channel ImageNet normalization).

Reads camera_ip / camera_port / cam_width / cam_height from the shared
`config/theia_tiny.yaml` so viewer and node stay in lockstep. Override the
config path with `--config /path/to/theia_tiny.yaml`.
"""

import argparse
import os
import socket
import struct
import threading

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

HEATMAP_H = 120
EMBED_DIM = 192


def _default_config_path():
    here = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(here, '..', 'config', 'theia_tiny.yaml'))


def _load_config(path):
    with open(path) as f:
        params = yaml.safe_load(f)['theia_tiny_node']['ros__parameters']
    return {
        'camera_ip': params['camera_ip'],
        'camera_port': int(params['camera_port']),
        'cam_width': int(params['cam_width']),
        'cam_height': int(params['cam_height']),
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
        self.cam_frame = None
        self.heatmap_frame = None
        self._stop = threading.Event()

        self.create_subscription(
            Float32MultiArray, '/theia_embedding', self._on_embedding, 10)
        self.create_timer(1.0 / 30.0, self._refresh)

        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._cam_thread.start()

    def _camera_loop(self):
        ip, port = self.cfg['camera_ip'], self.cfg['camera_port']
        w, h = self.cfg['cam_width'], self.cfg['cam_height']
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
                        if img.shape[1] != w or img.shape[0] != h:
                            img = cv2.resize(img, (w, h))
                        self.cam_frame = img
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

    def _refresh(self):
        panels = []
        display_w = self.cfg['cam_width']
        if self.cam_frame is not None:
            panels.append(self.cam_frame)
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
