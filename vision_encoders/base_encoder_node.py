#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from messages.msg import ImageRGB, ImageFeatures
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
import yaml
import os
import numpy as np
from PIL import Image
from ament_index_python.packages import get_package_share_directory
from abc import ABC, abstractmethod
import torchvision.transforms as T
import torch

class BaseEncoderNode(Node, ABC):
    """Base class for vision encoder nodes"""
    
    def __init__(self, node_name, config_name):
        super().__init__(node_name)
        
        # Load config
        pkg_share = get_package_share_directory('vision_encoders')
        config_path = os.path.join(pkg_share, 'config', f'{config_name}.yaml')
        
        if not os.path.exists(config_path):
            self.get_logger().error(f"Config not found: {config_path}")
            raise FileNotFoundError(config_path)
        
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        self.get_logger().info(f"Loaded config: {config_name}")
        
        # Load model (implemented by child classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.get_logger().info(f"Model loaded successfully")
        
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
        
        self.features_pub = self.create_publisher(
            ImageFeatures,
            self.cfg['topics']['features_output'],
            qos
        )
        
        self.get_logger().info(f"{node_name} ready")


    def _preprocess_image(self, pil_img):
        input_size = self.cfg['model'].get('input_size', 32)
        mean = self.cfg['model'].get('mean', [0.485, 0.456, 0.406])
        std = self.cfg['model'].get('std', [0.229, 0.224, 0.225])
        
        # Build transform like _build_transform
        transform = T.Compose([
            T.Resize(input_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])                                                                                                                                                  
        # Apply transform                                                                                                                                                                                                                                                                                                                                                           
        img = pil_img.convert('RGB')
        img_tensor = transform(img)
        # Add batch dimension and move to device
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        return img_tensor   


    @abstractmethod
    def _load_model(self):
        """Load the model - implemented by child classes"""
        pass

    @abstractmethod
    def _extract_features(self, img_tensor):
        """Extract features - implemented by child classes"""
        pass
    
    def image_callback(self, msg: ImageRGB):
        """Common callback logic for all encoders"""
        try:
            # Decode image
            inner_img = msg.img_rgb
            h, w = inner_img.height, inner_img.width
            
            if inner_img.encoding != 'rgb8':
                self.get_logger().warn(f"Unsupported encoding: {inner_img.encoding}")
                return
            
            rgb = np.frombuffer(inner_img.data, dtype=np.uint8).reshape((h, w, 3))
            print("ROS2 img shape:", rgb.shape)
            print("ROS2 img dtype:", rgb.dtype)
            print("Top-left pixel RGB:", rgb[0,0])
            print("Pixel (15,20) RGB:", rgb[15,20])
            for i, c in enumerate(['R','G','B']):
                print(f"Channel {c}: min {rgb[:,:,i].min()}, max {rgb[:,:,i].max()}, mean {rgb[:,:,i].mean():.2f}")
            Image.fromarray(rgb).save("ros2_topic_debug.png")
            pil_img = Image.fromarray(rgb)

            # Preprocess (model-specific)
            img_tensor = self._preprocess_image(pil_img)

            # Extract features (model-specific)
            features = self._extract_features(img_tensor)
            # Publish features
            self._publish_features(features)
            
        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
            import traceback
            traceback.print_exc()
    
    def _publish_features(self, features):
        """Publish features as custom message"""
        feat_array = features.cpu().numpy().flatten()
        feat_msg = ImageFeatures()
        feat_msg.header.stamp = self.get_clock().now().to_msg()
        feat_msg.features = feat_array.tolist()
        feat_msg.model_name = self.cfg['model']['model_name']
        self.features_pub.publish(feat_msg)