"""
High-level workflows for common tasks.
"""

from typing import Optional
from anndata import AnnData


def quick_finetune(
    adata_train: AnnData,
    adata_eval: Optional[AnnData] = None,
    model_name: str = "scgpt-base-neuroscience",
    num_classes: Optional[int] = None,
    label_key: str = "cell_type",
    output_dir: str = "./output",
    num_epochs: int = 5,
    batch_size: int = 32,
    strategy: str = "lora",
    **kwargs
):
    """
    Quick finetuning workflow with sensible defaults.
    
    Parameters
    ----------
    adata_train : AnnData
        Training data.
    adata_eval : AnnData, optional
        Evaluation data.
    model_name : str, optional (default: 'scgpt-base-neuroscience')
        Pretrained model to finetune.
    num_classes : int, optional
        Number of classes. If None, inferred from label_key.
    label_key : str, optional (default: 'cell_type')
        Key in adata.obs for labels.
    output_dir : str, optional (default: './output')
        Output directory for checkpoints.
    num_epochs : int, optional (default: 5)
        Number of training epochs.
    batch_size : int, optional (default: 32)
        Training batch size.
    strategy : str, optional (default: 'lora')
        Finetuning strategy ('full', 'lora', 'freeze_backbone').
    **kwargs
        Additional arguments.
    
    Returns
    -------
    model
        Finetuned model.
    
    Examples
    --------
    >>> import neurosc as nsc
    >>> import scanpy as sc
    >>> 
    >>> # Load and prepare data
    >>> adata = sc.read_h5ad("neuron_data.h5ad")
    >>> adata = nsc.prepare_anndata(adata)
    >>> 
    >>> # Quick finetune
    >>> model = nsc.tools.quick_finetune(
    ...     adata,
    ...     label_key="cell_type",
    ...     num_epochs=10,
    ...     strategy="lora"
    ... )
    """
    from ..models import load_model
    from ..data import create_dataloader
    from ..training import finetune_model, LoRAConfig
    
    print("="*60)
    print("NeuroSC Quick Finetune")
    print("="*60)
    
    # Load pretrained model
    print(f"\n1. Loading pretrained model: {model_name}")
    model = load_model(model_name)
    
    # Infer number of classes
    if num_classes is None and label_key in adata_train.obs:
        num_classes = adata_train.obs[label_key].nunique()
        print(f"   Detected {num_classes} classes from '{label_key}'")
    
    # Create dataloaders
    print(f"\n2. Creating dataloaders")
    train_loader = create_dataloader(
        adata_train,
        batch_size=batch_size,
        shuffle=True,
        return_labels=True,
        label_key=label_key,
    )
    
    eval_loader = None
    if adata_eval is not None:
        eval_loader = create_dataloader(
            adata_eval,
            batch_size=batch_size,
            shuffle=False,
            return_labels=True,
            label_key=label_key,
        )
    
    # Finetune
    print(f"\n3. Starting finetuning with strategy: {strategy}")
    
    lora_config = LoRAConfig() if strategy == "lora" else None
    
    model = finetune_model(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        num_classes=num_classes,
        strategy=strategy,
        lora_config=lora_config,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        **kwargs
    )
    
    print("\n" + "="*60)
    print("✓ Finetuning complete!")
    print(f"  Model saved to: {output_dir}")
    print("="*60)
    
    return model


def quick_predict(
    adata: AnnData,
    model,
    output_key: str = "predicted_labels",
    embedding_key: str = "X_neurosc",
    batch_size: int = 64,
    run_umap: bool = True,
):
    """
    Quick prediction workflow.
    
    Parameters
    ----------
    adata : AnnData
        Input data.
    model : nn.Module or str
        Model or path to model.
    output_key : str, optional (default: 'predicted_labels')
        Key to store predictions in adata.obs.
    embedding_key : str, optional (default: 'X_neurosc')
        Key to store embeddings in adata.obsm.
    batch_size : int, optional (default: 64)
        Batch size for inference.
    run_umap : bool, optional (default: True)
        Whether to compute UMAP on embeddings.
    
    Returns
    -------
    AnnData
        Data with predictions and embeddings added.
    
    Examples
    --------
    >>> import neurosc as nsc
    >>> 
    >>> # Load model and data
    >>> model = nsc.load_model("scgpt-base-neuroscience")
    >>> adata = sc.read_h5ad("new_data.h5ad")
    >>> adata = nsc.prepare_anndata(adata)
    >>> 
    >>> # Quick predict
    >>> adata = nsc.tools.quick_predict(adata, model)
    >>> 
    >>> # Visualize
    >>> sc.pl.umap(adata, color='predicted_labels')
    """
    from ..models import load_model
    from ..inference import predict, embed
    
    print("="*60)
    print("NeuroSC Quick Predict")
    print("="*60)
    
    # Load model if path provided
    if isinstance(model, str):
        print(f"\n1. Loading model: {model}")
        model = load_model(model)
    else:
        print(f"\n1. Using provided model")
    
    # Generate embeddings
    print(f"\n2. Generating embeddings")
    adata = embed(
        model,
        adata,
        batch_size=batch_size,
        key_added=embedding_key,
    )
    
    # Make predictions
    print(f"\n3. Making predictions")
    predictions = predict(
        model,
        adata,
        batch_size=batch_size,
    )
    adata.obs[output_key] = predictions
    
    # Run UMAP
    if run_umap:
        print(f"\n4. Computing UMAP")
        import scanpy as sc
        sc.pp.neighbors(adata, use_rep=embedding_key)
        sc.tl.umap(adata)
    
    print("\n" + "="*60)
    print("✓ Prediction complete!")
    print(f"  Predictions stored in: adata.obs['{output_key}']")
    print(f"  Embeddings stored in: adata.obsm['{embedding_key}']")
    print("="*60)
    
    return adata

