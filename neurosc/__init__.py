"""
NeuroSC - Neuroscience Single-Cell Foundation Models
=====================================================

An open-source package for cell type annotation using finetuned foundation models
for neuroscience scRNA-seq data.

Main Workflow:
1. Load your h5ad file
2. Load a finetuned classification model
3. Annotate cell types automatically

Features:
- Simple cell type annotation with finetuned foundation models
- Finetune foundation models (scGPT, Geneformer, etc.) on your data
- Scanpy-compatible API for seamless integration
- HuggingFace Hub integration for model hosting and sharing
- Comprehensive tools for neuroscience scRNA-seq analysis
"""

__version__ = "0.1.0"
__author__ = "NeuroSC Contributors"

# Main API - Cell Type Annotation
from .annotate import annotate_celltype, quick_annotate, load_and_annotate

# Core modules
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
    # Main API
    "annotate_celltype",
    "quick_annotate", 
    "load_and_annotate",
    # Modules
    "data",
    "models",
    "training",
    "inference",
    "utils",
    "tools",
    "tl",
    # Common functions
    "load_model",
    "list_pretrained_models",
    "prepare_anndata",
    "predict",
    "embed",
    "setup_huggingface",
    "download_pretrained",
]

