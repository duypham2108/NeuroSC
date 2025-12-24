"""
HuggingFace Hub integration for model hosting and sharing.
"""

import os
from typing import Optional
import warnings


def setup_huggingface(token: Optional[str] = None):
    """
    Set up HuggingFace Hub authentication.
    
    Parameters
    ----------
    token : str, optional
        HuggingFace API token. If not provided, will look for HF_TOKEN
        environment variable or use cached credentials.
    
    Examples
    --------
    >>> import neurosc as nsc
    >>> 
    >>> # Set up with token
    >>> nsc.setup_huggingface(token="hf_...")
    >>> 
    >>> # Or use environment variable
    >>> # export HF_TOKEN=hf_...
    >>> nsc.setup_huggingface()
    """
    try:
        from huggingface_hub import login
        
        if token is None:
            token = os.environ.get("HF_TOKEN")
        
        if token is not None:
            login(token=token)
            print("✓ Successfully authenticated with HuggingFace Hub")
        else:
            # Try to use cached credentials
            login()
            print("✓ Using cached HuggingFace credentials")
    
    except ImportError:
        print("huggingface_hub is required. Install with: pip install huggingface_hub")
    except Exception as e:
        print(f"Failed to authenticate: {e}")
        print("You can manually set your token with: huggingface-cli login")


def upload_to_hub(
    model,
    repo_id: str,
    commit_message: str = "Upload model",
    private: bool = False,
    token: Optional[str] = None,
):
    """
    Upload a model to HuggingFace Hub.
    
    This function allows you to freely host your finetuned models on HuggingFace,
    making them accessible to the community and enabling easy sharing.
    
    Parameters
    ----------
    model : nn.Module
        Model to upload.
    repo_id : str
        Repository ID (e.g., 'username/model-name').
    commit_message : str, optional (default: 'Upload model')
        Commit message for the upload.
    private : bool, optional (default: False)
        Whether to make the repository private.
    token : str, optional
        HuggingFace API token. If not provided, uses cached credentials.
    
    Examples
    --------
    >>> import neurosc as nsc
    >>> 
    >>> # Finetune a model
    >>> model = nsc.load_model("scgpt-base-neuroscience")
    >>> # ... training code ...
    >>> 
    >>> # Upload to HuggingFace Hub (free hosting!)
    >>> nsc.upload_to_hub(
    ...     model,
    ...     repo_id="myusername/scgpt-finetuned-neurons",
    ...     commit_message="Finetuned on cortical neurons"
    ... )
    >>> 
    >>> # Now anyone can use your model:
    >>> # loaded_model = nsc.load_model("myusername/scgpt-finetuned-neurons")
    """
    try:
        from huggingface_hub import HfApi, create_repo
        
        # Set up authentication
        if token is not None:
            setup_huggingface(token)
        
        # Create repository
        api = HfApi()
        try:
            create_repo(repo_id, private=private, exist_ok=True)
            print(f"✓ Repository created/verified: {repo_id}")
        except Exception as e:
            warnings.warn(f"Could not create repository: {e}")
        
        # Save model locally first
        temp_dir = f"./temp_{repo_id.split('/')[-1]}"
        os.makedirs(temp_dir, exist_ok=True)
        
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(temp_dir)
        else:
            import torch
            torch.save(model.state_dict(), os.path.join(temp_dir, "pytorch_model.bin"))
        
        # Upload to hub
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=repo_id,
            commit_message=commit_message,
        )
        
        print(f"✓ Model uploaded successfully to: https://huggingface.co/{repo_id}")
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
    
    except ImportError:
        print("huggingface_hub is required. Install with: pip install huggingface_hub")
    except Exception as e:
        print(f"Upload failed: {e}")


def download_from_hub(
    repo_id: str,
    save_path: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    """
    Download a model from HuggingFace Hub.
    
    Parameters
    ----------
    repo_id : str
        Repository ID (e.g., 'username/model-name').
    save_path : str, optional
        Local path to save the model. If None, uses cache.
    token : str, optional
        HuggingFace API token for private models.
    
    Returns
    -------
    str
        Path to downloaded model.
    
    Examples
    --------
    >>> import neurosc as nsc
    >>> 
    >>> # Download a model from the Hub
    >>> model_path = nsc.download_from_hub("neurosc/scgpt-base-neuroscience")
    >>> model = nsc.load_model(model_path)
    """
    try:
        from huggingface_hub import snapshot_download
        
        if token is not None:
            setup_huggingface(token)
        
        print(f"Downloading model from {repo_id}...")
        
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=save_path,
            use_auth_token=token,
        )
        
        print(f"✓ Model downloaded to: {path}")
        return path
    
    except ImportError:
        print("huggingface_hub is required. Install with: pip install huggingface_hub")
        return None
    except Exception as e:
        print(f"Download failed: {e}")
        return None

