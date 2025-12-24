"""
Utilities for downloading pretrained models.
"""

import os
from typing import Optional


def download_pretrained(
    model_name: str,
    save_dir: Optional[str] = None,
    force: bool = False,
) -> str:
    """
    Download a pretrained model.
    
    Parameters
    ----------
    model_name : str
        Name of the pretrained model (from list_pretrained_models()).
    save_dir : str, optional
        Directory to save the model. If None, uses default cache directory.
    force : bool, optional (default: False)
        Force re-download even if model exists locally.
    
    Returns
    -------
    str
        Path to downloaded model.
    
    Examples
    --------
    >>> import neurosc as nsc
    >>> 
    >>> # List available models
    >>> models = nsc.list_pretrained_models()
    >>> print(models)
    >>> 
    >>> # Download a specific model
    >>> model_path = nsc.download_pretrained("scgpt-base-neuroscience")
    >>> 
    >>> # Load the downloaded model
    >>> model = nsc.load_model(model_path)
    """
    from ..models.model_registry import PRETRAINED_MODELS
    from .huggingface import download_from_hub
    
    if model_name not in PRETRAINED_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(PRETRAINED_MODELS.keys())}"
        )
    
    model_info = PRETRAINED_MODELS[model_name]
    hf_id = model_info.get('hf_id')
    
    if hf_id is None:
        raise ValueError(f"No download URL available for {model_name}")
    
    # Set default save directory
    if save_dir is None:
        save_dir = os.path.expanduser(f"~/.cache/neurosc/models/{model_name}")
    
    # Check if already downloaded
    if os.path.exists(save_dir) and not force:
        print(f"Model already exists at {save_dir}")
        print("Use force=True to re-download")
        return save_dir
    
    # Download from HuggingFace
    print(f"Downloading {model_name}...")
    print(f"Description: {model_info.get('description', 'N/A')}")
    print(f"Parameters: {model_info.get('params', 'N/A')}")
    
    path = download_from_hub(hf_id, save_path=save_dir)
    
    if path is not None:
        print(f"✓ Successfully downloaded {model_name}")
        return path
    else:
        raise RuntimeError(f"Failed to download {model_name}")

