"""
Batch integration using foundation models.
"""

from typing import Optional, List
from anndata import AnnData


def integrate_batches(
    adata: AnnData,
    batch_key: str,
    model_name: str = "scgpt-base-neuroscience",
    key_added: str = "X_integrated",
    **kwargs
) -> AnnData:
    """
    Integrate batches using foundation model embeddings.
    
    Foundation models can learn batch-invariant representations, enabling
    integration without explicit batch correction methods.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data with batch information.
    batch_key : str
        Key in adata.obs containing batch labels.
    model_name : str, optional
        Foundation model to use.
    key_added : str, optional (default: 'X_integrated')
        Key to add integrated embeddings to adata.obsm.
    **kwargs
        Additional arguments.
    
    Returns
    -------
    AnnData
        Data with integrated embeddings.
    
    Examples
    --------
    >>> import neurosc as nsc
    >>> import scanpy as sc
    >>> 
    >>> # Integrate multiple batches
    >>> nsc.tl.integrate_batches(adata, batch_key="batch")
    >>> 
    >>> # Visualize integration
    >>> sc.pp.neighbors(adata, use_rep="X_integrated")
    >>> sc.tl.umap(adata)
    >>> sc.pl.umap(adata, color=['batch', 'cell_type'])
    """
    from ..models import load_model
    from ..inference import embed
    
    # Load model
    model = load_model(model_name)
    
    # Generate batch-invariant embeddings
    embed(
        model=model,
        adata=adata,
        key_added=key_added,
        **kwargs
    )
    
    return adata

