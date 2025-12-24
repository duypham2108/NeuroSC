"""
Visualization utilities for single-cell data and model outputs.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
from anndata import AnnData


def plot_embeddings(
    adata: AnnData,
    basis: str = "X_umap",
    color: Optional[str] = None,
    **kwargs
):
    """
    Plot cell embeddings (wrapper around scanpy plotting).
    
    Parameters
    ----------
    adata : AnnData
        Annotated data with embeddings.
    basis : str, optional (default: 'X_umap')
        Embedding basis to plot (e.g., 'X_umap', 'X_pca', 'X_neurosc').
    color : str, optional
        Variable to color points by.
    **kwargs
        Additional arguments passed to scanpy plotting.
    
    Examples
    --------
    >>> import neurosc as nsc
    >>> import scanpy as sc
    >>> 
    >>> # After generating embeddings
    >>> sc.pp.neighbors(adata, use_rep="X_neurosc")
    >>> sc.tl.umap(adata)
    >>> nsc.utils.plot_embeddings(adata, color='cell_type')
    """
    try:
        import scanpy as sc
        
        if basis == "X_umap" and "X_umap" not in adata.obsm:
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
        
        sc.pl.embedding(adata, basis=basis.replace("X_", ""), color=color, **kwargs)
    
    except ImportError:
        print("scanpy is required for plotting. Install with: pip install scanpy")


def plot_gene_expression(
    adata: AnnData,
    genes: List[str],
    groupby: Optional[str] = None,
    **kwargs
):
    """
    Plot gene expression patterns.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data.
    genes : list
        List of gene names to plot.
    groupby : str, optional
        Variable to group cells by.
    **kwargs
        Additional arguments.
    
    Examples
    --------
    >>> nsc.utils.plot_gene_expression(
    ...     adata,
    ...     genes=['RBFOX3', 'GAD1', 'SLC17A7'],
    ...     groupby='cell_type'
    ... )
    """
    try:
        import scanpy as sc
        
        if groupby is not None:
            sc.pl.dotplot(adata, genes, groupby=groupby, **kwargs)
        else:
            sc.pl.violin(adata, genes, **kwargs)
    
    except ImportError:
        print("scanpy is required for plotting. Install with: pip install scanpy")


def plot_training_history(
    history: Dict[str, List[float]],
    figsize: tuple = (12, 4),
    save_path: Optional[str] = None,
):
    """
    Plot training history (loss curves).
    
    Parameters
    ----------
    history : dict
        Training history dictionary with 'train_loss' and optionally 'eval_loss'.
    figsize : tuple, optional (default: (12, 4))
        Figure size.
    save_path : str, optional
        Path to save the figure.
    
    Examples
    --------
    >>> history = trainer.train()
    >>> nsc.utils.plot_training_history(history)
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    
    if 'eval_loss' in history:
        ax.plot(epochs, history['eval_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training History', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

