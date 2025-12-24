"""
Pseudotime inference using foundation models.
"""

from typing import Optional
from anndata import AnnData
import warnings


def infer_pseudotime(
    adata: AnnData,
    model_name: str = "scgpt-base-neuroscience",
    root_cell: Optional[str] = None,
    use_rep: str = "X_neurosc",
    key_added: str = "neurosc_pseudotime",
    **kwargs
) -> AnnData:
    """
    Infer pseudotime using foundation model embeddings.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data.
    model_name : str, optional
        Foundation model to use.
    root_cell : str, optional
        Name or index of root cell.
    use_rep : str, optional (default: 'X_neurosc')
        Representation to use for pseudotime.
    key_added : str, optional (default: 'neurosc_pseudotime')
        Key to add pseudotime to adata.obs.
    **kwargs
        Additional arguments.
    
    Returns
    -------
    AnnData
        Data with pseudotime added.
    
    Examples
    --------
    >>> import neurosc as nsc
    >>> 
    >>> # Embed cells
    >>> nsc.tl.embed_cells(adata)
    >>> 
    >>> # Infer pseudotime
    >>> nsc.tl.infer_pseudotime(adata, root_cell=0)
    >>> 
    >>> # Visualize
    >>> sc.pl.umap(adata, color='neurosc_pseudotime')
    """
    import scanpy as sc
    
    # Generate embeddings if not present
    if use_rep not in adata.obsm:
        from .embedding import embed_cells
        embed_cells(adata, model_name=model_name, key_added=use_rep)
    
    # Compute diffusion pseudotime using embeddings
    warnings.warn(
        "Pseudotime inference uses diffusion pseudotime on foundation model embeddings"
    )
    
    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.diffmap(adata)
    sc.tl.dpt(adata, n_dcs=10)
    
    # Copy to custom key
    adata.obs[key_added] = adata.obs['dpt_pseudotime']
    
    return adata

