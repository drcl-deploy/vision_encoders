#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from messages.msg import ImageFeatures, ImageRGB
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
import yaml
import os
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image as PILImage
from ament_index_python.packages import get_package_share_directory
from vision_encoders.model_utils import load_dinov3_model, preprocess_image_config, extract_patch_features

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
            ImageRGB, 
            self.cfg['topics']['image_input'], 
            self.image_callback, 
            qos
        )
        
        self.feature_pub = self.create_publisher(
            ImageFeatures, 
            self.cfg['topics']['features_output'], 
            qos
        )
        
        self.model = load_dinov3_model(
            model_name=self.cfg['model']['name'],
            repo_path=self.cfg['model']['repo_path'],
            model_path=self.cfg['model']['model_path'],
            device=self.cfg['model']['device']
        )

        self.device = self.cfg['model']['device']

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
    
    def image_callback(self, msg):
        try:
            inner_img = msg.img_rgb 
            
            h, w = inner_img.height, inner_img.width
            if inner_img.encoding != 'rgb8':
                self.get_logger().warn(f"Unsupported encoding: {inner_img.encoding}")
                return
            
            # Decode from inner_img
            rgb = np.frombuffer(inner_img.data, dtype=np.uint8).reshape((h, w, 3))
            # Preprocess
            pil_img = PILImage.fromarray(rgb)
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                n_layers = self.cfg['features'].get('n_layers', 12)
                
                feats = self.model.get_intermediate_layers(
                    tensor, 
                    n=range(n_layers),  # Which layers to extract from
                    reshape=True,       # Reshape to [embed_dim, H, W]
                    norm=True          # Apply normalization
                )
                
                # Choose which layer(s) to use
                output_type = self.cfg['features']['output_type']
                
                if output_type == 'last_layer':
                    # Use only the last layer
                    features = feats[-1]  # Shape: [batch, embed_dim, H, W]
                elif output_type == 'concatenate':
                    # Concatenate multiple layers
                    features = torch.cat(feats, dim=1)  # Shape: [batch, embed_dim*n_layers, H, W]
                elif output_type == 'average':
                    # Average across layers
                    features = torch.stack(feats).mean(dim=0)  # Shape: [batch, embed_dim, H, W]
                else:
                    # Return all layers as list
                    features = feats  # List of tensors
            
            # Publish
            feat_array = features.cpu().numpy().flatten()
            feat_msg = ImageFeatures()
            feat_msg.header.stamp = self.get_clock().now().to_msg()
            feat_msg.features = feat_array.tolist()
            feat_msg.model_name = self.cfg['model']['name']
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
