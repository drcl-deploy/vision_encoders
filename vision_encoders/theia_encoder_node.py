#!/usr/bin/env python3
import rclpy
import torch
import time
from transformers import AutoModel
from vision_encoders.base_encoder_node import BaseEncoderNode
import torchvision.transforms as T

class TheiaEncoderNode(BaseEncoderNode):
    """Theia encoder node with model-specific logic"""
    
    def __init__(self, config_name='theia'):
        super().__init__(node_name='theia_encoder_node', config_name=config_name)
    
    def _load_model(self):
        """Load Theia model from Hugging Face"""
        model_name = self.cfg['model']['model_name']
        
        self.get_logger().info(f"Loading Theia model: {model_name}")
        
        # Load Theia model
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model = model.to(self.device)
        model.eval()
        
        self.get_logger().info(f"Theia model loaded successfully")
        
        return model
    
    def _preprocess_image(self, pil_img):
        """Theia preprocessing: resize, center crop, to tensor (NO normalization)"""
        input_size = self.cfg['model'].get('input_size', 224)
        
        # Theia does its own normalization internally
        # Only resize, crop, and convert to tensor
        transform = T.Compose([
            T.Resize(input_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(input_size),
            T.ToTensor(),  # Converts to [0, 1] range, no normalization
        ])
        
        img = pil_img.convert('RGB')
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        return img_tensor
    
    
    def _extract_features(self, img_tensor):
        """Extract Theia patch features using forward_feature method"""
        self.model.eval()
        
        with torch.inference_mode():
            img_tensor = img_tensor.to(self.device)
            spatial_feats = self.model.forward_feature(img_tensor, do_rescale=False)
            spatial_feats = spatial_feats.squeeze().detach().cpu()
            num_patches, embed_dim = spatial_feats.shape
            h = w = int(num_patches ** 0.5)  # sqrt to get grid size (e.g., 14x14)
            spatial_feats = spatial_feats.transpose(0, 1).reshape(embed_dim, h, w)
        return spatial_feats


def main(args=None):
    import argparse
    
    # Parse arguments BEFORE rclpy.init
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='theia',
                        help='Config file name (without .yaml extension)')
    parsed_args, remaining = parser.parse_known_args()
    
    # Initialize ROS with remaining args
    rclpy.init(args=remaining)
    
    try:
        node = TheiaEncoderNode(config_name=parsed_args.config)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Node failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

