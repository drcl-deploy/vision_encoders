#!/usr/bin/env python3
import rclpy
import torch
from vision_encoders.base_encoder_node import BaseEncoderNode


class DinoEncoderNode(BaseEncoderNode):
    """DINOv3 encoder node with all model-specific logic"""
    
    def __init__(self, config_name):
        super().__init__(node_name='dino_encoder_node', config_name=config_name)
    
    def _load_model(self):
        """Load DINOv3 model from local repository"""
        model_name = self.cfg['model']['model_name']
        model_path = self.cfg['model']['model_path']
        repo_path = self.cfg['model']['repo_path']
        
        # Load model architecture
        model = torch.hub.load(repo_path, model_name, source='local', pretrained=False)
        
        # Load pretrained weights
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        # Move to device and eval mode
        model.to(self.device)
        model.eval()
        return model
    
    def _extract_features(self, img_tensor):
        """Extract DINOv3 patch features from intermediate layers"""
        n_layers = self.cfg['model'].get('n_layers', 12)
        
        self.model.eval()
        
        with torch.inference_mode():
            feats = self.model.get_intermediate_layers(
                img_tensor,
                n=range(n_layers),
                reshape=True,
                norm=True
            )
            # Return last layer features: [embed_dim, H, W]
            last_layer = feats[-1].squeeze().detach().cpu()
        return last_layer


def main(args=None):
    import argparse
    
    # Parse arguments BEFORE rclpy.init
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='dinov3',
                       help='Config file name (without .yaml extension)')
    parsed_args, remaining = parser.parse_known_args()
    
    # Initialize ROS with remaining args
    rclpy.init(args=remaining)
    
    try:
        node = DinoEncoderNode(config_name=parsed_args.config)
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

