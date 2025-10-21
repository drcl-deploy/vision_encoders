import os
import torch
import requests
from tqdm import tqdm

class ModelLoader:
    """Shared model loader for all encoder nodes."""
    
    def __init__(self, config, cache_dir=None):
        self.cfg = config
        
        if cache_dir is None:
            self.models_dir = os.path.expanduser('~/.cache/vision_encoders/models')
        else:
            self.models_dir = cache_dir
        
        os.makedirs(self.models_dir, exist_ok=True)
        print(f"[ModelLoader] Cache directory: {self.models_dir}")
    
    def download_weights(self):
        """Download model weights if not cached."""
        url = self.cfg['model']['weights']['url']
        filename = self.cfg['model']['weights']['filename']
        weights_path = os.path.join(self.models_dir, filename)
        
        if os.path.exists(weights_path):
            print(f"[ModelLoader] Using cached weights: {weights_path}")
            return weights_path
        
        print(f"[ModelLoader] Downloading {filename}...")
        print(f"[ModelLoader] URL: {url}")
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to download weights: {e}")
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(weights_path, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        print(f"[ModelLoader] Downloaded to {weights_path}")
        return weights_path
    
    def build_dinov3_model(self, weights_path):
        """Build DINOv3 model using our minimal ViT architecture."""
        from vision_encoders.vit_architecture import vit_small, vit_base, vit_large, vit_giant
        
        model_cfg = self.cfg['model']
        
        # Select architecture
        arch = model_cfg['architecture']
        img_size = self.cfg['preprocessing']['input_size']
        patch_size = model_cfg['patch_size']
        
        print(f"[ModelLoader] Building {arch} (patch_size={patch_size}, img_size={img_size})")
        
        if arch == 'vit_small':
            model = vit_small(patch_size=patch_size, img_size=img_size, num_classes=0)
        elif arch == 'vit_base':
            model = vit_base(patch_size=patch_size, img_size=img_size, num_classes=0)
        elif arch == 'vit_large':
            model = vit_large(patch_size=patch_size, img_size=img_size, num_classes=0)
        elif arch == 'vit_giant':
            model = vit_giant(patch_size=patch_size, img_size=img_size, num_classes=0)
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        # Load weights
        print(f"[ModelLoader] Loading weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in state_dict:
            state_dict = state_dict['model']
        if 'teacher' in state_dict:
            state_dict = state_dict['teacher']
        
        # Load with strict=False to ignore missing/extra keys
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"[ModelLoader] Load result: {msg}")
        
        return model
    
    def load_dinov3(self):
        """Load DINOv3 model (download weights if needed)."""
        weights_path = self.download_weights()
        model = self.build_dinov3_model(weights_path)
        
        device = torch.device(self.cfg['runtime']['device'])
        dtype_str = self.cfg['runtime'].get('dtype', 'float32')
        dtype = torch.float32 if dtype_str == 'float32' else torch.float16
        
        model.to(device=device, dtype=dtype)
        model.eval()
        
        print(f"[ModelLoader] Model ready: {self.cfg['model']['name']} on {device} ({dtype})")
        return model, device
