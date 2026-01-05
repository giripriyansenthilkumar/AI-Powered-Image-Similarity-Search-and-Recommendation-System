import io
import os
import json
from typing import Optional, Tuple

try:
    import numpy as np  # type: ignore
except Exception:
    np = None

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None

try:
    import torch  # type: ignore
    import torchvision.transforms as T  # type: ignore
except Exception:
    torch = None
    T = None

try:
    from sklearn.decomposition import PCA  # type: ignore
except Exception:
    PCA = None


def preprocess_image(file_bytes: bytes, size: int = 224):
    """Return a torch tensor ready for model input (batch dim included).
    Uses ImageNet normalization by default.
    """
    if torch is None or T is None or Image is None:
        raise RuntimeError("torch or torchvision or PIL not available")

    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    transforms = T.Compose([
        T.Resize(int(size * 1.1)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transforms(img).unsqueeze(0)
    return tensor


def load_model(model_path: str, device: str = "cpu"):
    if torch is None:
        raise RuntimeError("torch not installed")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    map_loc = torch.device(device)
    loaded_obj = torch.load(model_path, map_location=map_loc)
    
    # If it's a dict, it could be:
    # 1. A checkpoint dict with 'model_state_dict' key
    # 2. A raw state_dict
    if isinstance(loaded_obj, dict):
        # Extract state_dict from checkpoint if needed
        if 'model_state_dict' in loaded_obj:
            state_dict = loaded_obj['model_state_dict']
            print("[INFO] Loaded checkpoint format with 'model_state_dict'")
        else:
            state_dict = loaded_obj
            print("[INFO] Loaded raw state_dict format")
        
        # Build TripletModel architecture and load weights
        try:
            import torchvision.models as models
            import torch.nn.functional as F
            
            class TripletModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = models.resnet50(weights=None)
                    self.encoder.fc = torch.nn.Identity()
                    self.embed = torch.nn.Linear(2048, 128)  # Project to 128D
                
                def forward(self, x):
                    features = self.encoder(x)
                    emb = self.embed(features)
                    return F.normalize(emb, p=2, dim=1)  # L2 normalize
            
            model = TripletModel()
            model.load_state_dict(state_dict, strict=False)
            print(f"[INFO] Loaded TripletModel with {sum(p.numel() for p in model.parameters())} parameters")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load state_dict as TripletModel: {str(e)}. "
                "Please verify checkpoint format."
            )
    else:
        # Assume it's a full model object
        model = loaded_obj
        print("[INFO] Loaded full model object")
    
    model.to(torch.device(device))
    model.eval()
    return model


def load_gallery_map(path: Optional[str]):
    """Load a gallery metadata JSON mapping index ids to image URLs.
    Expected format: list or dict. If list, index positions are used as IDs.
    """
    if not path:
        return None
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Normalize to dict of id->url
            if isinstance(data, list):
                return {i: item for i, item in enumerate(data)}
            if isinstance(data, dict):
                return data
    except Exception:
        return None
    return None


def reduce_embedding_dimension(vec: np.ndarray, target_dim: int) -> np.ndarray:
    """Reduce embedding dimension using PCA if necessary.
    If vec.shape[1] == target_dim, return as-is.
    Otherwise, fit PCA on the vector and reduce.
    """
    if vec is None or np is None:
        return vec
    
    current_dim = vec.shape[1] if vec.ndim > 1 else 1
    
    if current_dim == target_dim:
        return vec
    
    if current_dim < target_dim:
        # If embedding is smaller than target, pad with zeros
        print(f"[WARNING] Embedding dimension {current_dim} < target {target_dim}. Padding with zeros.")
        padding = np.zeros((vec.shape[0], target_dim - current_dim), dtype=vec.dtype)
        return np.hstack([vec, padding])
    
    # Reduce dimension using PCA
    if PCA is None:
        print(f"[WARNING] sklearn not available; cannot reduce {current_dim}D → {target_dim}D. Using first {target_dim} features.")
        return vec[:, :target_dim]
    
    try:
        print(f"[INFO] Reducing embedding from {current_dim}D → {target_dim}D using PCA")
        pca = PCA(n_components=target_dim)
        reduced = pca.fit_transform(vec)
        return reduced.astype('float32')
    except Exception as e:
        print(f"[WARNING] PCA failed: {e}. Using first {target_dim} features.")
        return vec[:, :target_dim].astype('float32')