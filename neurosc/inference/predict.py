"""
Prediction and embedding utilities.
"""

import torch
import numpy as np
from typing import Optional, Union, Dict
from anndata import AnnData
from tqdm.auto import tqdm


def predict(
    model,
    adata: AnnData,
    gene_vocab: Optional[Dict[str, int]] = None,
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    return_probs: bool = False,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Make predictions on single-cell data.
    
    Parameters
    ----------
    model : nn.Module
        Trained model.
    adata : AnnData
        Input data.
    gene_vocab : dict, optional
        Gene vocabulary for tokenization.
    batch_size : int, optional (default: 64)
        Batch size for inference.
    device : str, optional
        Device to use for inference.
    return_probs : bool, optional (default: False)
        If True, return probabilities instead of class predictions.
    
    Returns
    -------
    np.ndarray or dict
        If return_probs=False: array of predicted labels
        If return_probs=True: dict with 'labels' and 'probabilities'
    
    Examples
    --------
    >>> import neurosc as nsc
    >>> import scanpy as sc
    >>> 
    >>> # Load model and data
    >>> model = nsc.load_model("scgpt-base-neuroscience")
    >>> adata = sc.read_h5ad("data.h5ad")
    >>> adata = nsc.prepare_anndata(adata)
    >>> 
    >>> # Make predictions
    >>> predictions = nsc.predict(model, adata)
    >>> adata.obs['predicted_cell_type'] = predictions
    """
    from ..data import create_dataloader
    
    model.eval()
    model = model.to(device)
    
    # Create dataloader
    dataloader = create_dataloader(
        adata,
        batch_size=batch_size,
        shuffle=False,
        gene_vocab=gene_vocab,
    )
    
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            # Move to device
            gene_ids = batch['genes'].to(device) if 'genes' in batch else None
            expression = batch['expression'].to(device)
            
            # Forward pass
            if gene_ids is not None:
                outputs = model(gene_ids, expression)
            else:
                outputs = model.predict(expression, expression)
            
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            all_logits.append(logits.cpu().numpy())
    
    all_logits = np.concatenate(all_logits, axis=0)
    
    if return_probs:
        # Return both labels and probabilities
        from scipy.special import softmax
        probs = softmax(all_logits, axis=-1)
        labels = np.argmax(all_logits, axis=-1)
        return {
            'labels': labels,
            'probabilities': probs
        }
    else:
        # Return only labels
        return np.argmax(all_logits, axis=-1)


def embed(
    model,
    adata: AnnData,
    gene_vocab: Optional[Dict[str, int]] = None,
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    key_added: Optional[str] = "X_neurosc",
) -> Union[AnnData, np.ndarray]:
    """
    Generate cell embeddings.
    
    Parameters
    ----------
    model : nn.Module
        Trained model.
    adata : AnnData
        Input data.
    gene_vocab : dict, optional
        Gene vocabulary for tokenization.
    batch_size : int, optional (default: 64)
        Batch size for inference.
    device : str, optional
        Device to use for inference.
    key_added : str, optional (default: 'X_neurosc')
        Key to store embeddings in adata.obsm. If None, returns embeddings as array.
    
    Returns
    -------
    AnnData or np.ndarray
        If key_added is specified, returns adata with embeddings added to obsm.
        Otherwise, returns embeddings as numpy array.
    
    Examples
    --------
    >>> import neurosc as nsc
    >>> import scanpy as sc
    >>> 
    >>> # Load model and data
    >>> model = nsc.load_model("scgpt-base-neuroscience")
    >>> adata = sc.read_h5ad("data.h5ad")
    >>> adata = nsc.prepare_anndata(adata)
    >>> 
    >>> # Generate embeddings
    >>> adata = nsc.embed(model, adata, key_added="X_scgpt")
    >>> 
    >>> # Use embeddings for downstream analysis
    >>> sc.pp.neighbors(adata, use_rep="X_scgpt")
    >>> sc.tl.umap(adata)
    >>> sc.pl.umap(adata, color='cell_type')
    """
    from ..data import create_dataloader
    
    model.eval()
    model = model.to(device)
    
    # Create dataloader
    dataloader = create_dataloader(
        adata,
        batch_size=batch_size,
        shuffle=False,
        gene_vocab=gene_vocab,
    )
    
    embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Embedding"):
            # Move to device
            gene_ids = batch['genes'].to(device) if 'genes' in batch else None
            expression = batch['expression'].to(device)
            
            # Get embeddings
            if gene_ids is not None:
                emb = model.embed(gene_ids, expression)
            else:
                outputs = model(expression, expression)
                emb = outputs['embeddings'] if isinstance(outputs, dict) else outputs
            
            embeddings.append(emb.cpu().numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    
    if key_added is not None:
        adata.obsm[key_added] = embeddings
        return adata
    else:
        return embeddings


def batch_predict(
    model,
    adata: AnnData,
    batch_key: str,
    **kwargs
) -> np.ndarray:
    """
    Make predictions with batch-aware processing.
    
    Parameters
    ----------
    model : nn.Module
        Trained model.
    adata : AnnData
        Input data with batch information.
    batch_key : str
        Key in adata.obs containing batch labels.
    **kwargs
        Additional arguments passed to predict().
    
    Returns
    -------
    np.ndarray
        Predictions for all cells.
    """
    batches = adata.obs[batch_key].unique()
    all_predictions = []
    
    for batch in tqdm(batches, desc="Processing batches"):
        batch_adata = adata[adata.obs[batch_key] == batch]
        preds = predict(model, batch_adata, **kwargs)
        all_predictions.append(preds)
    
    return np.concatenate(all_predictions, axis=0)


def batch_embed(
    model,
    adata: AnnData,
    batch_key: str,
    key_added: str = "X_neurosc",
    **kwargs
) -> AnnData:
    """
    Generate embeddings with batch-aware processing.
    
    Parameters
    ----------
    model : nn.Module
        Trained model.
    adata : AnnData
        Input data with batch information.
    batch_key : str
        Key in adata.obs containing batch labels.
    key_added : str, optional (default: 'X_neurosc')
        Key to store embeddings in adata.obsm.
    **kwargs
        Additional arguments passed to embed().
    
    Returns
    -------
    AnnData
        Data with embeddings added.
    """
    batches = adata.obs[batch_key].unique()
    all_embeddings = []
    
    for batch in tqdm(batches, desc="Processing batches"):
        batch_adata = adata[adata.obs[batch_key] == batch]
        emb = embed(model, batch_adata, key_added=None, **kwargs)
        all_embeddings.append(emb)
    
    adata.obsm[key_added] = np.concatenate(all_embeddings, axis=0)
    return adata

