#!/usr/bin/env python3
"""
Export the Theia-Tiny model to ONNX format.

Reads config from config/theia_tiny.yaml for resolution, model name, and output path.
CLI args override config values.

Matches Isaac Lab's image_features._prepare_theia_transformer_model:
    features = model.backbone.model(pixel_values=img, interpolate_pos_encoding=True)
    return features.last_hidden_state[:, 1:]

Usage:
    python3 export_theia_onnx.py                          # uses config/theia_tiny.yaml
    python3 export_theia_onnx.py --height 32 --width 57   # override resolution

Requires:
    pip install torch "transformers==4.41.2" "numpy<2" "Pillow>=9.1" onnx onnxscript onnxruntime einops omegaconf pyyaml
"""

import argparse
import os

import torch
import torch.nn as nn
import yaml
from transformers import AutoModel

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG = os.path.join(PKG_ROOT, "config", "theia_tiny.yaml")


def load_config(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    # Extract ros__parameters from the ROS2 param YAML format
    return raw["theia_tiny_node"]["ros__parameters"]


class TheiaONNXWrapper(nn.Module):
    """Wraps Theia backbone for ONNX export with interpolate_pos_encoding=True.

    Input:  (B, 3, H, W) float32, ImageNet-normalized (H/W must be multiples of 16)
    Output: (B, num_patches, 192) float32 patch tokens (CLS excluded)
            where num_patches = (H/16) * (W/16)
    """

    def __init__(self, theia_model):
        super().__init__()
        self.backbone_model = theia_model.backbone.model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.backbone_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=True,
        )
        return features.last_hidden_state[:, 1:]


def main():
    parser = argparse.ArgumentParser(description="Export Theia-Tiny to ONNX")
    parser.add_argument("--config", "-c", default=DEFAULT_CONFIG, help="Path to YAML config")
    parser.add_argument("--output", "-o", default=None, help="Override output ONNX path")
    parser.add_argument("--model-name", default=None, help="Override HuggingFace model name")
    parser.add_argument("--height", type=int, default=None, help="Override image height")
    parser.add_argument("--width", type=int, default=None, help="Override image width")
    parser.add_argument("--opset", type=int, default=None, help="Override ONNX opset version")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Config: {args.config}")

    # Resolve values: CLI overrides > config (flat namespace now)
    model_name = args.model_name or cfg["hf_name"]
    height = args.height or cfg["model_input_height"]
    width = args.width or cfg["model_input_width"]
    opset = args.opset or cfg["opset"]
    output_path = args.output or os.path.join(PKG_ROOT, cfg["model_path"])

    num_patches = (height // 16) * (width // 16)
    if num_patches == 0:
        raise ValueError(f"Resolution {height}x{width} too small — need at least 16px in each dimension")

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Loading model: {model_name}")
    theia = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    theia.eval()

    wrapper = TheiaONNXWrapper(theia)
    wrapper.eval()

    print(f"Export resolution: {height}x{width} → {num_patches} patches")

    dummy = torch.randn(1, 3, height, width, dtype=torch.float32)

    with torch.no_grad():
        out = wrapper(dummy)
    print(f"Output shape: {out.shape}")

    print(f"Exporting to ONNX (opset {opset})...")
    torch.onnx.export(
        wrapper,
        (dummy,),
        output_path,
        opset_version=opset,
        input_names=["pixel_values"],
        output_names=["features"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "features": {0: "batch_size"},
        },
    )

    print(f"Saved: {output_path}")

    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(output_path)
        ort_out = sess.run(
            None,
            {"pixel_values": np.random.randn(1, 3, height, width).astype(np.float32)},
        )
        print(f"ONNX Runtime validation OK — output shape: {ort_out[0].shape}")
    except ImportError:
        print("onnxruntime not installed, skipping validation")


if __name__ == "__main__":
    main()
