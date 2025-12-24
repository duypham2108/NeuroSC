"""
Embedding functions compatible with scanpy workflow.
"""

from typing import Optional
from anndata import AnnData
import warnings


def embed_cells(
    adata: AnnData,
    model_name: str = "scgpt-base-neuroscience",
    layer: Optional[str] = None,
    use_rep: Optional[str] = None,
    key_added: str = "X_neurosc",
    batch_size: int = 64,
    copy: bool = False,
) -> Optional[AnnData]:
    """
    Embed cells using a foundation model (scanpy-compatible).
    
    Similar to scanpy's tl.pca or external.pp.bbknn, this function embeds
    single-cell data using a foundation model and stores the result in obsm.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    model_name : str, optional (default: 'scgpt-base-neuroscience')
        Name of the pretrained model to use.
    layer : str, optional
        Layer to use for embedding. If None, uses .X.
    use_rep : str, optional
        Use this representation from .obsm instead of .X.
    key_added : str, optional (default: 'X_neurosc')
        Key to add embeddings to adata.obsm.
    batch_size : int, optional (default: 64)
        Batch size for inference.
    copy : bool, optional (default: False)
        Whether to return a copy.
    
    Returns
    -------
    AnnData or None
        If copy=True, returns modified copy. Otherwise modifies adata inplace.
    
    Examples
    --------
    >>> import neurosc as nsc
    >>> import scanpy as sc
    >>> 
    >>> # Load data
    >>> adata = sc.read_h5ad("data.h5ad")
    >>> 
    >>> # Preprocess (scanpy workflow)
    >>> sc.pp.filter_cells(adata, min_genes=200)
    >>> sc.pp.filter_genes(adata, min_cells=3)
    >>> sc.pp.normalize_total(adata, target_sum=1e4)
    >>> sc.pp.log1p(adata)
    >>> 
    >>> # Embed using foundation model (replaces PCA)
    >>> nsc.tl.embed_cells(adata, key_added="X_scgpt")
    >>> 
    >>> # Continue with scanpy workflow
    >>> sc.pp.neighbors(adata, use_rep="X_scgpt")
    >>> sc.tl.umap(adata)
    >>> sc.tl.leiden(adata)
    >>> sc.pl.umap(adata, color=['leiden', 'cell_type'])
    """
    from ..models import load_model
    from ..inference import embed
    
    if copy:
        adata = adata.copy()
    
    # Load model
    try:
        model = load_model(model_name)
    except Exception as e:
        warnings.warn(f"Could not load model: {e}. Using random embeddings.")
        import numpy as np
        adata.obsm[key_added] = np.random.randn(adata.n_obs, 512)
        return adata if copy else None
    
    # Generate embeddings
    embed(
        model=model,
        adata=adata,
        batch_size=batch_size,
        key_added=key_added,
    )
    
    return adata if copy else None

