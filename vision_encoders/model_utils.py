"""
DINOv3 Model Loader - Local Repository Version
Loads DINOv3 models from local repository instead of Hugging Face
"""

from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import time


# Default constants
PATCH_SIZE = 16
MODEL_INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_image_config(img, input_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD, device='cuda'):
    """
    Preprocess image by strictly resizing to input_size x input_size,
    normalizing with mean/std, and converting to torch tensor.

    Args:
        img (PIL.Image): Input image.
        input_size (int): Size to resize the image to (e.g., 224).
        mean (tuple): Mean for normalization.
        std (tuple): Std for normalization.
        device (str): Device to place tensor on ('cuda' or 'cpu').

    Returns:
        torch.Tensor: Preprocessed image tensor of shape [1, 3, input_size, input_size].
    """
    img = img.convert('RGB')
    img = img.resize((input_size, input_size), Image.BILINEAR)
    img_tensor = TF.to_tensor(img)
    img_tensor = TF.normalize(img_tensor, mean=mean, std=std)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    return img_tensor


def load_dinov3_model(model_name, model_path, repo_path='/home/learning/shravan/dinov3', device='cuda'):
    """
    Loads a DINOv3 model from the specified local repo and checkpoint.
    
    Args:
        model_name (str): Name of the model (e.g., 'dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16')
        model_path (str or Path): Path to the .pth checkpoint file
        repo_path (str): Path to the local DINOv3 repository
        device (str): Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded and ready-to-use DINOv3 model in eval mode
    """
    print(f"Loading {model_name} from local repo: {repo_path}")
    
    # Load model architecture from local repository
    model = torch.hub.load(repo_path, model_name, source='local', pretrained=False)
    
    # Load pretrained weights
    print(f"Loading weights from: {model_path}")
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    # Move to device and set eval mode
    model.to(device)
    model.eval()
    
    # Print model info
    patch_size = getattr(model, 'patch_size', 32)
    embed_dim = getattr(model, 'embed_dim', None)
    print(f"✓ Patch size: {patch_size}")
    print(f"✓ Embedding dimension: {embed_dim}")
    
    if device == 'cuda':
        print(f"✓ Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    
    return model


def extract_patch_features(model, input_tensor, n_layers):
    """
    Extract patch (local) features from a preprocessed image tensor using the given model.
    
    Args:
        model: Loaded DINOv3 model
        input_tensor: Preprocessed image tensor [1, 3, H, W] on device
        n_layers: Number of transformer layers to extract features from
    
    Returns:
        dict: Dictionary mapping layer index to patch features from each transformer layer
        float: Inference time in seconds
    """
    model.eval()
    patch_features_dict = {}
    
    start_time = time.time()
    with torch.inference_mode():
        feats = model.get_intermediate_layers(
            input_tensor, 
            n=range(n_layers), 
            reshape=True, 
            norm=True
        )
        for i, layer_feats in enumerate(feats):
            layer_feats = layer_feats.squeeze().detach().cpu()
            patch_features_dict[i] = layer_feats
    
    inference_time = time.time() - start_time
    return patch_features_dict, inference_time


# Optional: Alternative preprocessing that maintains aspect ratio
def preprocess_image_aspect_ratio(img, patch_size=PATCH_SIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD, device='cuda'):
    """
    Alternative preprocessing that maintains aspect ratio and ensures dimensions 
    are multiples of patch size.
    
    Args:
        img (PIL.Image): Input image.
        patch_size (int): Patch size to ensure dimensions are multiples of.
        mean (tuple): Mean for normalization.
        std (tuple): Std for normalization.
        device (str): Device to place tensor on.
    
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    img = img.convert('RGB')
    w, h = img.size
    
    # Adjust dimensions to be multiples of patch_size
    if w % patch_size != 0 or h % patch_size != 0:
        new_w = (w // patch_size) * patch_size
        new_h = (h // patch_size) * patch_size
        img = img.resize((new_w, new_h), Image.BILINEAR)
    
    img_tensor = TF.to_tensor(img)
    img_tensor = TF.normalize(img_tensor, mean=mean, std=std)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    return img_tensor
