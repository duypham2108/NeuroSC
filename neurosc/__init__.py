"""
NeuroSC - Neuroscience Single-Cell Foundation Models
=====================================================

An open-source package for finetuning and deploying foundation single-cell models
for neuroscience scRNA-seq data.

Features:
- Finetune foundation models (scGPT, Geneformer, etc.) on neuroscience data
- Scanpy-compatible API for seamless integration with existing workflows
- Pre-trained model access and management
- HuggingFace Hub integration for model hosting and sharing
- Comprehensive tools for neuroscience scRNA-seq analysis
"""

__version__ = "0.1.0"
__author__ = "NeuroSC Contributors"

from . import data
from . import models
from . import training
from . import inference
from . import utils
from . import tools
from . import tl  # scanpy-compatible API

# Convenience imports for common functions
from .models import load_model, list_pretrained_models
from .data import prepare_anndata
from .inference import predict, embed
from .tools import setup_huggingface, download_pretrained

__all__ = [
    "data",
    "models",
    "training",
    "inference",
    "utils",
    "tools",
    "tl",
    "load_model",
    "list_pretrained_models",
    "prepare_anndata",
    "predict",
    "embed",
    "setup_huggingface",
    "download_pretrained",
]

