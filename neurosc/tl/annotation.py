"""
Cell type annotation using foundation models.
"""

from typing import Optional, Dict
from anndata import AnnData
import warnings


def annotate_cells(
    adata: AnnData,
    model_name: str = "scgpt-base-neuroscience",
    reference: Optional[AnnData] = None,
    label_key: str = "cell_type",
    key_added: str = "predicted_cell_type",
    batch_size: int = 64,
) -> AnnData:
    """
    Automatically annotate cell types using a foundation model.
    
    Parameters
    ----------
    adata : AnnData
        Query data to annotate.
    model_name : str, optional
        Name of pretrained model.
    reference : AnnData, optional
        Reference dataset with known labels. If None, uses model's internal knowledge.
    label_key : str, optional (default: 'cell_type')
        Key in reference.obs containing cell type labels.
    key_added : str, optional (default: 'predicted_cell_type')
        Key to add predictions to adata.obs.
    batch_size : int, optional (default: 64)
        Batch size for inference.
    
    Returns
    -------
    AnnData
        Data with predictions added.
    
    Examples
    --------
    >>> import neurosc as nsc
    >>> 
    >>> # Annotate using pretrained model
    >>> nsc.tl.annotate_cells(adata, model_name="scgpt-base-neuroscience")
    >>> 
    >>> # View predictions
    >>> print(adata.obs['predicted_cell_type'].value_counts())
    """
    from ..models import load_model
    from ..inference import predict
    
    # Load model
    model = load_model(model_name)
    
    # If reference provided, finetune on reference first
    if reference is not None:
        warnings.warn(
            "Reference-based annotation not yet implemented. "
            "Using pretrained model predictions."
        )
    
    # Predict
    predictions = predict(
        model=model,
        adata=adata,
        batch_size=batch_size,
    )
    
    adata.obs[key_added] = predictions
    
    return adata


def transfer_labels(
    adata_ref: AnnData,
    adata_query: AnnData,
    label_key: str = "cell_type",
    model_name: str = "scgpt-base-neuroscience",
    key_added: str = "transferred_labels",
    **kwargs
) -> AnnData:
    """
    Transfer labels from reference to query dataset.
    
    Parameters
    ----------
    adata_ref : AnnData
        Reference dataset with known labels.
    adata_query : AnnData
        Query dataset to transfer labels to.
    label_key : str, optional (default: 'cell_type')
        Key in adata_ref.obs containing labels.
    model_name : str, optional
        Foundation model to use for transfer.
    key_added : str, optional (default: 'transferred_labels')
        Key to add transferred labels to adata_query.obs.
    **kwargs
        Additional arguments.
    
    Returns
    -------
    AnnData
        Query data with transferred labels.
    
    Examples
    --------
    >>> import neurosc as nsc
    >>> 
    >>> # Transfer labels from reference to query
    >>> nsc.tl.transfer_labels(
    ...     adata_ref=reference_data,
    ...     adata_query=new_data,
    ...     label_key="cell_type"
    ... )
    """
    from ..models import load_model
    from ..training import finetune_model
    from ..data import create_dataloader
    
    # Load pretrained model
    model = load_model(model_name)
    
    # Finetune on reference
    num_classes = adata_ref.obs[label_key].nunique()
    train_loader = create_dataloader(
        adata_ref,
        batch_size=32,
        return_labels=True,
        label_key=label_key,
    )
    
    # Quick finetune
    model = finetune_model(
        model=model,
        train_dataloader=train_loader,
        num_classes=num_classes,
        strategy="lora",
        output_dir="./temp_label_transfer",
        num_epochs=3,
        **kwargs
    )
    
    # Predict on query
    from ..inference import predict
    predictions = predict(model, adata_query)
    
    # Map predictions to labels
    label_map = {i: label for i, label in enumerate(adata_ref.obs[label_key].cat.categories)}
    adata_query.obs[key_added] = [label_map.get(p, "Unknown") for p in predictions]
    
    return adata_query

