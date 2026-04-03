#!/usr/bin/env python3
"""Lightweight combined viewer: TCP camera stream + Theia embedding heatmap.

Mirrors theia_viewer.py but reads camera frames from camera_streamer.py
over TCP instead of the DDS /frontvideostream topic.

Usage:
    python3 theia_tcp_viewer.py [orin_ip]
    # or via launch:
    ros2 launch vision_encoders theia_tcp_viewer.launch.py
"""

import sys
import socket
import struct
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import cv2

DISPLAY_W = 960
HEATMAP_H = 120
EMBED_DIM = 192


class TheiaTcpViewer(Node):
    def __init__(self):
        super().__init__('theia_tcp_viewer')

        self.declare_parameter('orin_ip', '192.168.123.164')
        self.declare_parameter('orin_port', 5555)

        self.orin_ip = self.get_parameter('orin_ip').get_parameter_value().string_value
        self.orin_port = self.get_parameter('orin_port').get_parameter_value().integer_value

        self.cam_frame = None
        self.heatmap_frame = None
        self._frame_lock = threading.Lock()
        self._running = True
        self._tcp_thread = None

        self.create_subscription(
            Float32MultiArray, '/theia_embedding', self._on_embedding, 10)
        self.create_timer(1.0 / 30.0, self._refresh)

    def _tcp_loop(self):
        """Connect to Orin and receive JPEG frames."""
        while self._running:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.orin_ip, self.orin_port))
                self.get_logger().info(f'Connected to {self.orin_ip}:{self.orin_port}')
            except OSError as e:
                self.get_logger().warn(f'Cannot connect to {self.orin_ip}:{self.orin_port}: {e}')
                import time
                time.sleep(2)
                continue

            buf = b''
            try:
                while self._running:
                    # Read 4-byte length header
                    while len(buf) < 4:
                        chunk = sock.recv(4096)
                        if not chunk:
                            raise ConnectionError
                        buf += chunk
                    length = struct.unpack('<I', buf[:4])[0]
                    buf = buf[4:]

                    # Read JPEG payload
                    while len(buf) < length:
                        chunk = sock.recv(65536)
                        if not chunk:
                            raise ConnectionError
                        buf += chunk
                    jpg_data = buf[:length]
                    buf = buf[length:]

                    img = cv2.imdecode(
                        np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        with self._frame_lock:
                            self.cam_frame = img
            except (OSError, ConnectionError):
                self.get_logger().warn('Disconnected, reconnecting...')
                sock.close()

    def _on_embedding(self, msg):
        vec = np.array(msg.data, dtype=np.float32)
        rows = vec.size // EMBED_DIM
        if rows == 0:
            return
        normed = cv2.normalize(vec.reshape(rows, EMBED_DIM), None, 0, 255,
                               cv2.NORM_MINMAX)
        self.heatmap_frame = cv2.applyColorMap(normed.astype(np.uint8),
                                               cv2.COLORMAP_VIRIDIS)

    def _refresh(self):
        panels = []
        with self._frame_lock:
            cam = self.cam_frame
        if cam is not None:
            h, w = cam.shape[:2]
            panels.append(cv2.resize(cam, (DISPLAY_W, int(h * DISPLAY_W / w))))
        if self.heatmap_frame is not None:
            panels.append(cv2.resize(self.heatmap_frame,
                                     (DISPLAY_W, HEATMAP_H),
                                     interpolation=cv2.INTER_NEAREST))
        if not panels:
            return
        cv2.imshow('Theia TCP Viewer', np.vstack(panels))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise SystemExit

    def destroy_node(self):
        self._running = False
        super().destroy_node()


def main():
    rclpy.init()

    # Allow passing orin_ip as a positional arg for convenience
    args = {}
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        args['orin_ip'] = sys.argv[1]

    node = TheiaTcpViewer()
    if args:
        node.orin_ip = args['orin_ip']

    # Start TCP thread after IP is finalized
    node._tcp_thread = threading.Thread(target=node._tcp_loop, daemon=True)
    node._tcp_thread.start()

    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
