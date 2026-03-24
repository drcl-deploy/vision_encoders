import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import av
from unitree_go.msg import Go2FrontVideoData


class VideoToImage(Node):
    def __init__(self):
        super().__init__('video_to_image')
        self.bridge = CvBridge()

        self.declare_parameter('mode', 'sim')
        self.mode = self.get_parameter('mode').get_parameter_value().string_value

        if self.mode == 'real':
            self.codec = av.CodecContext.create('h264', 'r')
            self.get_logger().info('Using H.264 decode (real robot)')
        else:
            self.get_logger().info('Using raw RGB decode (simulator)')

        self.sub = self.create_subscription(
            Go2FrontVideoData,
            '/head_camera',
            self.callback, 10)
        self.pub = self.create_publisher(Image, '/head_camera_image', 10)

    def callback(self, msg):
        try:
            if self.mode == 'real':
                self._decode_h264(msg)
            else:
                self._decode_raw(msg)
        except Exception as e:
            self.get_logger().warn(f'Decode error: {e}')

    def _decode_h264(self, msg):
        packets = self.codec.parse(bytes(msg.video720p))
        for packet in packets:
            frames = self.codec.decode(packet)
            for frame in frames:
                img = frame.to_ndarray(format='bgr24')
                self.pub.publish(self.bridge.cv2_to_imgmsg(img, encoding='bgr8'))

    def _decode_raw(self, msg):
        data = np.frombuffer(bytes(msg.video720p), dtype=np.uint8)
        if data.size == 0:
            return
        # Infer dimensions: simulator uses CAM_WIDTH x CAM_HEIGHT x 3 (RGB, bottom-up)
        num_pixels = data.size // 3
        if num_pixels == 0:
            return
        # Try known simulator resolutions
        for w, h in [(57, 32), (640, 480), (1280, 720)]:
            if w * h * 3 == data.size:
                img = data.reshape((h, w, 3))
                img = np.flipud(img)  # MuJoCo renders bottom-up
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                self.pub.publish(self.bridge.cv2_to_imgmsg(img, encoding='bgr8'))
                return
        self.get_logger().warn(
            f'Unexpected raw frame size: {data.size} bytes ({num_pixels} pixels)')


def main():
    rclpy.init()
    rclpy.spin(VideoToImage())

if __name__ == '__main__':
    main()
