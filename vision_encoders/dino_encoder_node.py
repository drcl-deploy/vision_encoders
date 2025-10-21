#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
import yaml
import os
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image as PILImage
from ament_index_python.packages import get_package_share_directory

class DinoEncoderNode(Node):
    def __init__(self, config_name):
        super().__init__('dino_encoder_node')
        
        # Load config
        pkg_share = get_package_share_directory('vision_encoders')
        config_path = os.path.join(pkg_share, 'config', f'{config_name}.yaml')
        
        if not os.path.exists(config_path):
            self.get_logger().error(f"Config not found: {config_path}")
            raise FileNotFoundError(config_path)
        
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        self.get_logger().info(f"Loaded config: {config_name}")
        
        # QoS
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        
        # Subscribers and publishers
        self.image_sub = self.create_subscription(
            Image, 
            self.cfg['topics']['image_input'], 
            self.image_callback, 
            qos
        )
        
        self.feature_pub = self.create_publisher(
            Float32MultiArray, 
            self.cfg['topics']['features_output'], 
            qos
        )
        
        # Load model
        from vision_encoders.model_loader import ModelLoader
        
        loader = ModelLoader(self.cfg)
        self.model, self.device = loader.load_dinov3()
        
        # Preprocessing
        self.transform = self._build_transform()
        
        self.get_logger().info(f"DINO encoder ready: {self.cfg['model']['name']}")
    
    def _build_transform(self):
        prep = self.cfg['preprocessing']
        return T.Compose([
            T.Resize(prep['input_size'], interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(prep['input_size']),
            T.ToTensor(),
            T.Normalize(mean=prep['mean'], std=prep['std'])
        ])
    
    def image_callback(self, msg: Image):
        try:
            # Decode
            h, w = msg.height, msg.width
            if msg.encoding != 'rgb8':
                self.get_logger().warn(f"Unsupported encoding: {msg.encoding}")
                return
            
            rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))
            
            # Preprocess
            pil_img = PILImage.fromarray(rgb)
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(tensor)
                
                output_type = self.cfg['features']['output_type']
                if output_type == 'cls_token':
                    features = output[:, 0]
                elif output_type == 'avg_pool':
                    features = output[:, 1:].mean(dim=1)
                else:
                    features = output
                
                if self.cfg['features'].get('normalize', False):
                    features = torch.nn.functional.normalize(features, dim=-1)
            
            # Publish
            feat_array = features.cpu().numpy().flatten()
            feat_msg = Float32MultiArray()
            feat_msg.data = feat_array.tolist()
            self.feature_pub.publish(feat_msg)
            
        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")


def main(args=None):
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        type=str, 
        default='dinov3_vits16',
        help='Model config name (without .yaml extension)'
    )
    cli_args, ros_args = parser.parse_known_args()
    
    rclpy.init(args=ros_args)
    node = DinoEncoderNode(config_name=cli_args.config)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
