import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge
import cv2

class VecToHeatmap(Node):
    def __init__(self):
        super().__init__('vec_to_heatmap')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Float32MultiArray,
            '/theia_embedding',
            self.callback, 10)
        self.pub = self.create_publisher(Image, '/heatmap_image', 10)

    def callback(self, msg):
        vec = np.array(msg.data)  # shape: (1152,)
        heatmap = vec.reshape(6, 192)  # 6 x 192

        # normalize to 0-255 and apply colormap
        normed = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        colored = cv2.applyColorMap(normed.astype(np.uint8), cv2.COLORMAP_VIRIDIS)

        self.pub.publish(self.bridge.cv2_to_imgmsg(colored, encoding='bgr8'))

def main():
    rclpy.init()
    rclpy.spin(VecToHeatmap())

if __name__ == '__main__':
    main()