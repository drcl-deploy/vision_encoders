#!/usr/bin/env python3
"""Lightweight combined viewer: head camera + Theia embedding heatmap."""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from unitree_go.msg import Go2FrontVideoData
import numpy as np
import cv2
import av

DISPLAY_W = 960
HEATMAP_H = 120
EMBED_DIM = 192


class TheiaViewer(Node):
    def __init__(self):
        super().__init__('theia_viewer')

        self.declare_parameter('mode', 'sim')
        self.mode = self.get_parameter('mode').get_parameter_value().string_value

        if self.mode == 'real':
            self.codec = av.CodecContext.create('h264', 'r')

        self.cam_frame = None
        self.heatmap_frame = None

        self.create_subscription(
            Go2FrontVideoData, '/head_camera', self._on_video, 10)
        self.create_subscription(
            Float32MultiArray, '/theia_embedding', self._on_embedding, 10)
        self.create_timer(1.0 / 30.0, self._refresh)

    def _on_video(self, msg):
        try:
            if self.mode == 'real':
                self._decode_h264(msg)
            else:
                self._decode_raw(msg)
        except Exception as e:
            self.get_logger().warn(f'Decode error: {e}')

    def _decode_h264(self, msg):
        for packet in self.codec.parse(bytes(msg.video720p)):
            for frame in self.codec.decode(packet):
                self.cam_frame = frame.to_ndarray(format='bgr24')

    def _decode_raw(self, msg):
        data = np.frombuffer(bytes(msg.video720p), dtype=np.uint8)
        if data.size == 0:
            return
        for w, h in [(57, 32), (640, 480), (1280, 720)]:
            if w * h * 3 == data.size:
                img = np.flipud(data.reshape(h, w, 3))
                self.cam_frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return

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
        if self.cam_frame is not None:
            h, w = self.cam_frame.shape[:2]
            panels.append(cv2.resize(self.cam_frame,
                                     (DISPLAY_W, int(h * DISPLAY_W / w))))
        if self.heatmap_frame is not None:
            panels.append(cv2.resize(self.heatmap_frame,
                                     (DISPLAY_W, HEATMAP_H),
                                     interpolation=cv2.INTER_NEAREST))
        if not panels:
            return
        cv2.imshow('Theia Viewer', np.vstack(panels))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise SystemExit


def main():
    rclpy.init()
    node = TheiaViewer()
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
