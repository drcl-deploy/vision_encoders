import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from messages.msg import ImageRGB

import numpy as np
import cv2
from datetime import datetime


class TestImagePublisher(Node):
    def __init__(self, image_path, topic_name='/image_input'):
        super().__init__('test_image_publisher')
        
        # Load the RGB image from IsaacSim
        self.rgb_image = np.load(image_path)
        self.get_logger().info(f"Loaded image: {self.rgb_image.shape}, range=[{self.rgb_image.min()}, {self.rgb_image.max()}]")
        
        # QoS matching your subscriber
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        
        # Publisher
        self.publisher = self.create_publisher(ImageRGB, topic_name, qos)
        
        # Publish once after 1 second, then every 0.5 seconds
        self.timer = self.create_timer(0.5, self.publish_image)
        self.pub_count = 0
        
        self.get_logger().info(f"Publishing on topic: {topic_name}")
    
    def publish_image(self):
        """Publish the test image"""
        # Create ImageRGB message
        msg = ImageRGB()
        
        # Create inner sensor_msgs/Image
        inner_msg = Image()
        inner_msg.header.stamp = self.get_clock().now().to_msg()
        inner_msg.header.frame_id = 'camera'
        
        # Image dimensions
        h, w = self.rgb_image.shape[:2]
        inner_msg.height = h
        inner_msg.width = w
        inner_msg.encoding = 'rgb8'
        inner_msg.is_bigendian = 0
        inner_msg.step = w * 3
        
        # # Convert to uint8 if needed
        # if self.rgb_image.dtype != np.uint8:
        #     # Normalize from [1, 197] to [0, 255]
        #     rgb_normalized = (self.rgb_image - self.rgb_image.min()) / (self.rgb_image.max() - self.rgb_image.min())
        #     rgb_uint8 = (rgb_normalized * 255).astype(np.uint8)
        # else:
        #     rgb_uint8 = self.rgb_image
        
        # Flatten and convert to bytes
        inner_msg.data = self.rgb_image.tobytes()
        # Assign to outer message
        msg.img_rgb = inner_msg
        
        # Publish
        self.publisher.publish(msg)
        self.pub_count += 1
        
        if self.pub_count <= 3:  # Only log first 3 times
            self.get_logger().info(f"Published image #{self.pub_count}: {h}x{w}, encoding={inner_msg.encoding}")

def main(args=None):
    rclpy.init(args=args)
    
    # Path to your saved RGB image from IsaacSim
    image_path = '/home/learning/shravan/fcrl/debug_vision_encoder/rgb_image_raw_20251107_131214.npy'  # Update with actual timestamp
    topic_name = '/image_rgb'  # Update with your actual topic name
    
    print("\n" + "="*70)
    print("🚀 TEST IMAGE PUBLISHER")
    print("="*70)
    print(f"Image: {image_path}")
    print(f"Topic: {topic_name}")
    print("="*70)
    
    try:
        node = TestImagePublisher(image_path, topic_name)
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n✅ Publisher stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()