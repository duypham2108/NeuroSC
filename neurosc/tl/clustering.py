"""
Clustering functions using foundation model embeddings.
"""

from typing import Optional
from anndata import AnnData
import scanpy as sc


def cluster_cells(
    adata: AnnData,
    use_rep: str = "X_neurosc",
    method: str = "leiden",
    resolution: float = 1.0,
    key_added: Optional[str] = None,
    **kwargs
):
    """
    Cluster cells using foundation model embeddings.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data with embeddings in obsm.
    use_rep : str, optional (default: 'X_neurosc')
        Representation to use for clustering.
    method : str, optional (default: 'leiden')
        Clustering method: 'leiden' or 'louvain'.
    resolution : float, optional (default: 1.0)
        Resolution parameter for clustering.
    key_added : str, optional
        Key to add cluster labels to adata.obs.
    **kwargs
        Additional arguments passed to clustering function.
    
    Examples
    --------
    >>> import neurosc as nsc
    >>> 
    >>> # Embed and cluster
    >>> nsc.tl.embed_cells(adata)
    >>> nsc.tl.cluster_cells(adata, resolution=0.5)
    """
    # Compute neighbors if not already done
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, use_rep=use_rep)
    
    # Cluster
    if method == "leiden":
        sc.tl.leiden(adata, resolution=resolution, key_added=key_added, **kwargs)
    elif method == "louvain":
        sc.tl.louvain(adata, resolution=resolution, key_added=key_added, **kwargs)
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def leiden(
    adata: AnnData,
    use_rep: str = "X_neurosc",
    resolution: float = 1.0,
    **kwargs
):
    """
    Leiden clustering on foundation model embeddings.
    
    Wrapper around scanpy.tl.leiden that uses foundation model embeddings.
    """
    cluster_cells(adata, use_rep=use_rep, method="leiden", resolution=resolution, **kwargs)


def louvain(
    adata: AnnData,
    use_rep: str = "X_neurosc",
    resolution: float = 1.0,
    **kwargs
):
    """
    Louvain clustering on foundation model embeddings.
    
    Wrapper around scanpy.tl.louvain that uses foundation model embeddings.
    """
    cluster_cells(adata, use_rep=use_rep, method="louvain", resolution=resolution, **kwargs)

