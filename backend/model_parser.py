"""
ModelSentinel — Model Parser
Unified loader for PyTorch (.pt/.pth) and SafeTensors (.safetensors) model files.
Extracts weight tensors as NumPy arrays organized by layer name.
"""

import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

_FORMAT_PT = "pytorch"
_FORMAT_SAFETENSORS = "safetensors"
_SUPPORTED_EXTENSIONS = {".pt", ".pth", ".bin", ".safetensors"}


@dataclass
class LayerInfo:
    """Metadata about a single layer/parameter tensor."""
    name: str
    shape: tuple
    dtype: str
    param_count: int
    weights: np.ndarray  # The actual weight data as float32 numpy array


@dataclass
class ModelArtifact:
    """Parsed model with all metadata and weight tensors."""
    file_path: str
    file_name: str
    file_size_bytes: int
    sha256: str
    model_format: str
    total_params: int
    total_layers: int
    layer_names: List[str]
    layers: Dict[str, LayerInfo] = field(default_factory=dict)
    raw_state_dict_keys: List[str] = field(default_factory=list)
    has_non_tensor_values: bool = False
    extra_metadata: Dict = field(default_factory=dict)


def _compute_sha256(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _load_pytorch(file_path: str) -> dict:
    """Load a PyTorch state dict, handling both raw state_dicts and wrapped checkpoints."""
    import torch

    # Use weights_only=False for compatibility but we'll scan with fickling separately
    try:
        data = torch.load(file_path, map_location="cpu", weights_only=True)
    except Exception:
        # Fallback for older formats — fickling will scan this for safety
        data = torch.load(file_path, map_location="cpu", weights_only=False)

    # Handle wrapped checkpoints like {"model_state_dict": ..., "optimizer_state_dict": ...}
    if isinstance(data, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in data and isinstance(data[key], dict):
                return data[key]
    return data


def _load_safetensors(file_path: str) -> dict:
    """Load a SafeTensors file and return tensors as a dict."""
    from safetensors.numpy import load_file
    return load_file(file_path)


def parse_model(file_path: str) -> ModelArtifact:
    """
    Parse a model file and extract all weight tensors.

    Args:
        file_path: Path to the model file (.pt, .pth, .safetensors)

    Returns:
        ModelArtifact with all metadata and weight data

    Raises:
        ValueError: If the file format is not supported
        FileNotFoundError: If the file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{ext}'. Supported: {_SUPPORTED_EXTENSIONS}"
        )

    file_size = os.path.getsize(file_path)
    sha256 = _compute_sha256(file_path)
    file_name = os.path.basename(file_path)

    # Determine format and load
    if ext == ".safetensors":
        model_format = _FORMAT_SAFETENSORS
        raw_data = _load_safetensors(file_path)
    else:
        model_format = _FORMAT_PT
        raw_data = _load_pytorch(file_path)

    # Extract tensors
    layers: Dict[str, LayerInfo] = {}
    raw_keys = list(raw_data.keys()) if isinstance(raw_data, dict) else []
    has_non_tensor = False
    total_params = 0

    if isinstance(raw_data, dict):
        for key, value in raw_data.items():
            arr = _to_numpy(value)
            if arr is not None:
                layer = LayerInfo(
                    name=key,
                    shape=tuple(arr.shape),
                    dtype=str(arr.dtype),
                    param_count=int(arr.size),
                    weights=arr.astype(np.float32),
                )
                layers[key] = layer
                total_params += layer.param_count
            else:
                has_non_tensor = True

    layer_names = list(layers.keys())

    return ModelArtifact(
        file_path=file_path,
        file_name=file_name,
        file_size_bytes=file_size,
        sha256=sha256,
        model_format=model_format,
        total_params=total_params,
        total_layers=len(layers),
        layer_names=layer_names,
        layers=layers,
        raw_state_dict_keys=raw_keys,
        has_non_tensor_values=has_non_tensor,
    )


def _to_numpy(value) -> Optional[np.ndarray]:
    """Convert a tensor-like value to a numpy array, or return None."""
    # PyTorch tensor
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass

    # NumPy array (from safetensors)
    if isinstance(value, np.ndarray):
        return value

    # Try generic conversion
    try:
        arr = np.array(value)
        if arr.dtype.kind in ("f", "i", "u"):  # float, int, unsigned int
            return arr
    except (TypeError, ValueError):
        pass

    return None
