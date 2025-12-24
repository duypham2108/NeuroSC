"""
Simplified cell type annotation workflow.

This module provides the main high-level API for NeuroSC's primary use case:
loading single-cell data and annotating cell types using finetuned foundation models.
"""

import numpy as np
from typing import Optional, Union, Dict, List
from pathlib import Path
from anndata import AnnData
import scanpy as sc


def annotate_celltype(
    adata_path: Union[str, Path, AnnData],
    model_path: Union[str, Path],
    output_key: str = "cell_type",
    batch_size: int = 64,
    preprocess: bool = True,
    return_probabilities: bool = False,
    label_mapping: Optional[Dict[int, str]] = None,
    device: str = "auto",
) -> AnnData:
    """
    Annotate cell types in single-cell data using a finetuned foundation model.
    
    This is the main function for NeuroSC's primary workflow:
    1. Load your h5ad file
    2. Load your finetuned classification model
    3. Get cell type annotations
    
    Parameters
    ----------
    adata_path : str, Path, or AnnData
        Path to h5ad file or AnnData object containing single-cell data.
    model_path : str or Path
        Path to finetuned model checkpoint or HuggingFace model ID.
    output_key : str, optional (default: 'cell_type')
        Key to store predictions in adata.obs.
    batch_size : int, optional (default: 64)
        Batch size for inference.
    preprocess : bool, optional (default: True)
        Whether to apply preprocessing. Set to False if data is already preprocessed.
    return_probabilities : bool, optional (default: False)
        If True, also store prediction probabilities in adata.obsm.
    label_mapping : dict, optional
        Mapping from prediction indices to cell type names.
        e.g., {0: 'Neuron', 1: 'Astrocyte', 2: 'Oligodendrocyte'}
    device : str, optional (default: 'auto')
        Device to use ('cuda', 'cpu', or 'auto').
    
    Returns
    -------
    AnnData
        Input data with cell type annotations added to adata.obs[output_key].
    
    Examples
    --------
    Basic usage:
    
    >>> import neurosc as nsc
    >>> 
    >>> # Annotate cell types
    >>> adata = nsc.annotate_celltype(
    ...     adata_path="data/brain_cells.h5ad",
    ...     model_path="username/scgpt-brain-celltype-classifier"
    ... )
    >>> 
    >>> # View results
    >>> print(adata.obs['cell_type'].value_counts())
    
    With custom label mapping:
    
    >>> label_map = {
    ...     0: 'Excitatory Neuron',
    ...     1: 'Inhibitory Neuron',
    ...     2: 'Astrocyte',
    ...     3: 'Oligodendrocyte',
    ...     4: 'Microglia'
    ... }
    >>> 
    >>> adata = nsc.annotate_celltype(
    ...     adata_path="data/brain_cells.h5ad",
    ...     model_path="./my_finetuned_model/",
    ...     label_mapping=label_map,
    ...     return_probabilities=True
    ... )
    >>> 
    >>> # Access probabilities
    >>> probs = adata.obsm['cell_type_probabilities']
    
    Skip preprocessing if already done:
    
    >>> adata = nsc.annotate_celltype(
    ...     adata_path="preprocessed_data.h5ad",
    ...     model_path="username/model",
    ...     preprocess=False
    ... )
    """
    from .models import load_model
    from .inference import predict
    from .data import prepare_anndata
    import torch
    
    # 1. Load data
    print("=" * 60)
    print("NeuroSC Cell Type Annotation")
    print("=" * 60)
    
    if isinstance(adata_path, AnnData):
        adata = adata_path
        print(f"\n✓ Using provided AnnData: {adata.n_obs} cells x {adata.n_vars} genes")
    else:
        print(f"\n[1/4] Loading data from: {adata_path}")
        adata = sc.read_h5ad(adata_path)
        print(f"      ✓ Loaded: {adata.n_obs} cells x {adata.n_vars} genes")
    
    # 2. Preprocess if needed
    if preprocess:
        print(f"\n[2/4] Preprocessing data...")
        original_n_obs = adata.n_obs
        adata = prepare_anndata(
            adata,
            target_sum=1e4,
            min_genes=200,
            min_cells=3,
            copy=False
        )
        print(f"      ✓ Preprocessed: {adata.n_obs} cells (filtered {original_n_obs - adata.n_obs})")
    else:
        print(f"\n[2/4] Skipping preprocessing (preprocess=False)")
    
    # 3. Load model
    print(f"\n[3/4] Loading classification model from: {model_path}")
    
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = load_model(model_path, device=device)
    print(f"      ✓ Model loaded on {device}")
    
    # 4. Predict cell types
    print(f"\n[4/4] Predicting cell types...")
    predictions = predict(
        model=model,
        adata=adata,
        batch_size=batch_size,
        device=device,
        return_probs=return_probabilities,
    )
    
    # Store predictions
    if return_probabilities:
        pred_labels = predictions['labels']
        pred_probs = predictions['probabilities']
        adata.obsm[f'{output_key}_probabilities'] = pred_probs
        print(f"      ✓ Probabilities stored in adata.obsm['{output_key}_probabilities']")
    else:
        pred_labels = predictions
    
    # Apply label mapping if provided
    if label_mapping is not None:
        adata.obs[output_key] = [label_mapping.get(int(p), f"Unknown_{p}") for p in pred_labels]
        print(f"      ✓ Mapped predictions to cell type names")
    else:
        adata.obs[output_key] = pred_labels
    
    print(f"      ✓ Predictions stored in adata.obs['{output_key}']")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Annotation Complete!")
    print("=" * 60)
    print(f"\nCell type distribution:")
    print(adata.obs[output_key].value_counts().to_string())
    print(f"\nTotal cells annotated: {adata.n_obs:,}")
    print("=" * 60)
    
    return adata


def quick_annotate(
    h5ad_file: str,
    model: str,
    output_file: Optional[str] = None,
    **kwargs
) -> AnnData:
    """
    Quick one-liner for cell type annotation.
    
    Parameters
    ----------
    h5ad_file : str
        Path to input h5ad file.
    model : str
        Path to model or HuggingFace model ID.
    output_file : str, optional
        Path to save annotated h5ad file. If None, doesn't save.
    **kwargs
        Additional arguments passed to annotate_celltype.
    
    Returns
    -------
    AnnData
        Annotated data.
    
    Examples
    --------
    >>> import neurosc as nsc
    >>> 
    >>> # One-line annotation
    >>> adata = nsc.quick_annotate(
    ...     h5ad_file="data.h5ad",
    ...     model="username/brain-classifier",
    ...     output_file="annotated_data.h5ad"
    ... )
    """
    adata = annotate_celltype(h5ad_file, model, **kwargs)
    
    if output_file is not None:
        print(f"\nSaving annotated data to: {output_file}")
        adata.write_h5ad(output_file)
        print("✓ Saved successfully")
    
    return adata


def load_and_annotate(
    data_path: str,
    model_path: str,
    cell_type_key: str = "cell_type",
    save_path: Optional[str] = None,
    visualize: bool = True,
    **kwargs
) -> AnnData:
    """
    Complete workflow: load → annotate → visualize → save.
    
    Parameters
    ----------
    data_path : str
        Path to h5ad file.
    model_path : str
        Path to finetuned model.
    cell_type_key : str, optional (default: 'cell_type')
        Key to store cell type predictions.
    save_path : str, optional
        Path to save results.
    visualize : bool, optional (default: True)
        Whether to generate UMAP visualization.
    **kwargs
        Additional arguments for annotate_celltype.
    
    Returns
    -------
    AnnData
        Annotated data with UMAP coordinates (if visualize=True).
    
    Examples
    --------
    >>> import neurosc as nsc
    >>> 
    >>> # Complete workflow
    >>> adata = nsc.load_and_annotate(
    ...     data_path="brain_data.h5ad",
    ...     model_path="models/brain_classifier",
    ...     save_path="results/annotated.h5ad",
    ...     visualize=True
    ... )
    """
    # Annotate
    adata = annotate_celltype(
        adata_path=data_path,
        model_path=model_path,
        output_key=cell_type_key,
        **kwargs
    )
    
    # Visualize
    if visualize:
        print("\nGenerating UMAP visualization...")
        try:
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            sc.pl.umap(adata, color=[cell_type_key], save=f'_{cell_type_key}_annotation.png')
            print("✓ UMAP saved")
        except Exception as e:
            print(f"⚠ Visualization failed: {e}")
    
    # Save
    if save_path is not None:
        print(f"\nSaving to: {save_path}")
        adata.write_h5ad(save_path)
        print("✓ Saved")
    
    return adata

