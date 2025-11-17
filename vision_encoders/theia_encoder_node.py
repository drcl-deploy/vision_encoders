#!/usr/bin/env python3
import rclpy
import torch
import time
from transformers import AutoModel
from vision_encoders.base_encoder_node import BaseEncoderNode
import torchvision.transforms as T
import numpy as np
from messages.msg import ImageRGB
from PIL import Image



class TheiaEncoderNode(BaseEncoderNode):
    """Theia encoder node with model-specific logic"""
    
    def __init__(self, config_name='theia'):
        super().__init__(node_name='theia_encoder_node', config_name=config_name)
        # self.image_callback(None)  # For testing immediately upon startup
    
    def _load_model(self):
        """Load Theia model from Hugging Face"""
        model_name = self.cfg['model']['model_name']
        
        # Load Theia model
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model = model.to(self.device)
        model.eval()

        # print(f"\n{'='*70}")
        # print("MODEL LOADING INFO")
        # print(f"{'='*70}")
        # print(f"Model name: {model_name}")
        # print(f"Model class: {type(model)}")
        # print(f"Model device: {self.device}")

        # # Check a specific weight to verify
        # first_weight = next(model.parameters())
        # print(f"First weight stats:")
        # print(f"  Shape: {first_weight.shape}")
        # print(f"  Mean: {first_weight.mean():.8f}")
        # print(f"  Std: {first_weight.std():.8f}")
        # print(f"  Min/Max: [{first_weight.min():.8f}, {first_weight.max():.8f}]")
        # print(f"{'='*70}\n")

        return model
    
    def _preprocess_image(self, pil_img):
        """Theia preprocessing - NO RESIZING, direct from 32x32"""
        
        # DON'T resize - keep original 32x32
        transform = T.Compose([
            T.ToTensor(),  # Converts to [0, 1] range
        ])
        
        img = pil_img.convert('RGB')
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # (1, 3, 32, 32)
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        img_tensor = img_tensor.to(self.device)
        img_tensor = (img_tensor - mean) / std
        # print(f"\n[ROS2] Preprocessed tensor stats:")
        # print(f"  Shape: {img_tensor.shape}")
        # print(f"  Mean: {img_tensor.mean():.10f}")
        # print(f"  Std: {img_tensor.std():.10f}")
        # print(f"  Min/Max: [{img_tensor.min():.10f}, {img_tensor.max():.10f}]")
        # print(f"  First pixel [R,G,B]: {img_tensor[0, :, 0, 0]}")
        return img_tensor  # (1, 3, 32, 32)
    
    def _extract_features(self, img_tensor):
        """Extract features - MUST use interpolate_pos_encoding=True"""
        self.model.eval()
        
        with torch.inference_mode():
            img_tensor = img_tensor.to(self.device)
            # CRITICAL: Must include interpolate_pos_encoding=True!
            features = self.model.backbone.model(
                pixel_values=img_tensor, 
                interpolate_pos_encoding=True  # Allows 32x32 input
            )
            
            # Extract spatial features (excluding CLS token)
            spatial_feats = features.last_hidden_state[:, 1:]  # (1, 4, 192)
            
            # Flatten to 1D exactly like IsaacSim
            spatial_feats = spatial_feats.squeeze(0).flatten().cpu()  # (768,)
        return spatial_feats

    # def image_callback(self, msg: ImageRGB):
    #     """Common callback logic for all encoders"""
    #     try:
    #         img_np = np.load('/home/shravan447/fcrl/collected_vision_test/compare_rgb_image.npy')
    #         pil_img = Image.fromarray(img_np)
    #         # Preprocess (model-specific)
    #         img_tensor = self._preprocess_image(pil_img)

    #         # Extract features (model-specific)
    #         features = self._extract_features(img_tensor)
    #         np.save("/home/shravan447/fcrl/collected_vision_test/features_ros2_mjc.npy", features.numpy())
    #         self.compare_features()
    #         # self._publish_features(features)
            
    #     except Exception as e:
    #         self.get_logger().error(f"Inference error: {e}")
    #         import traceback
    #         traceback.print_exc()

    # def compare_features(self):
    #     isaac_obs = np.load('/home/shravan447/fcrl/collected_vision_test/features_ros2.npy')
    #     isaac_features = isaac_obs[-768:]  # Last 768 elements (adjust if different)
    #     isaac_features = isaac_features.flatten()

    #     print(f"IsaacSim features shape: {isaac_features.shape}")

    #     # Load ROS2 features
    #     ros_features = np.load('/home/shravan447/fcrl/collected_vision_test/features_ros2_mjc.npy')
    #     print(f"ROS2 features shape: {ros_features.shape}")

    #     # Ensure shapes match
    #     if isaac_features.shape != ros_features.shape:
    #         print(f"\n❌ Shape mismatch!")
    #         print(f"   IsaacSim: {isaac_features.shape}")
    #         print(f"   ROS2:     {ros_features.shape}")
    #         exit(1)

    #     print("\n" + "="*70)
    #     print("🔍 FEATURE COMPARISON")
    #     print("="*70)

    #     # Statistics
    #     print(f"\nIsaacSim Features:")
    #     print(f"  L2 Norm: {np.linalg.norm(isaac_features):.6f}")
    #     print(f"  Mean: {isaac_features.mean():.6f}")
    #     print(f"  Std: {isaac_features.std():.6f}")
    #     print(f"  Range: [{isaac_features.min():.6f}, {isaac_features.max():.6f}]")

    #     print(f"\nROS2 Features:")
    #     print(f"  L2 Norm: {np.linalg.norm(ros_features):.6f}")
    #     print(f"  Mean: {ros_features.mean():.6f}")
    #     print(f"  Std: {ros_features.std():.6f}")
    #     print(f"  Range: [{ros_features.min():.6f}, {ros_features.max():.6f}]")

    #     # Cosine similarity
    #     dot_product = np.dot(isaac_features, ros_features)
    #     norm_isaac = np.linalg.norm(isaac_features)
    #     norm_ros = np.linalg.norm(ros_features)
    #     cosine_sim = dot_product / (norm_isaac * norm_ros)

    #     # MSE
    #     mse = np.mean((isaac_features - ros_features) ** 2)

    #     # Element-wise metrics
    #     abs_diff = np.abs(isaac_features - ros_features)
    #     max_abs_diff = np.max(abs_diff)
    #     mean_abs_diff = np.mean(abs_diff)

    #     # Correlation
    #     correlation = np.corrcoef(isaac_features, ros_features)[0, 1]

    #     print("\n" + "="*70)
    #     print("📊 SIMILARITY METRICS")
    #     print("="*70)

    #     print(f"\n🎯 Cosine Similarity: {cosine_sim:.8f}")
    #     if cosine_sim > 0.999:
    #         print("   ✅ EXCELLENT (> 0.999) - Nearly identical!")
    #     elif cosine_sim > 0.99:
    #         print("   ✅ PASS (> 0.99) - Very good match")
    #     elif cosine_sim > 0.95:
    #         print("   ⚠️  ACCEPTABLE (> 0.95) - Minor differences")
    #     else:
    #         print("   ❌ FAIL (< 0.95) - Significant mismatch")

    #     print(f"\n📏 Mean Squared Error: {mse:.8f}")
    #     print(f"📏 Max Absolute Diff: {max_abs_diff:.6f}")
    #     print(f"📏 Mean Absolute Diff: {mean_abs_diff:.6f}")
    #     print(f"📐 Correlation: {correlation:.8f}")

    #     # Worst mismatches
    #     print(f"\n🔬 Top 5 Worst Mismatches:")
    #     worst_idx = np.argsort(abs_diff)[-5:][::-1]
    #     for i, idx in enumerate(worst_idx, 1):
    #         print(f"   {i}. Index {idx}: Isaac={isaac_features[idx]:.6f}, "
    #             f"ROS={ros_features[idx]:.6f}, Diff={abs_diff[idx]:.6f}")

    #     # Final verdict
    #     print("\n" + "="*70)
    #     print("🏁 FINAL VERDICT")
    #     print("="*70)

    #     if cosine_sim > 0.99:
    #         print("✅ VISION ENCODER MATCHES!")
    #         print("   → ROS2 Theia encoder produces same features as IsaacSim")
    #         print("   → Problem is NOT in vision encoding")
    #         print("   → Move to Test 2: Frozen Observation Test")
    #         print("   → Debug dynamics/observation construction instead")
    #     else:
    #         print("❌ VISION ENCODER MISMATCH DETECTED")
    #         print("   → Debug checklist:")
    #         print("     1. Verify same model weights")
    #         print("     2. Check preprocessing steps")
    #         print("     3. Ensure interpolate_pos_encoding=True")
    #         print("     4. Verify feature extraction (no reshape/transpose)")

    #     print("="*70)


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

