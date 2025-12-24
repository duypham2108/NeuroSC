"""
Finetuning Example - Finetune a Foundation Model
=================================================

This script demonstrates how to finetune a foundation model on
your own neuroscience scRNA-seq data.
"""

import neurosc as nsc
import scanpy as sc
import numpy as np

print("=" * 60)
print("NeuroSC Finetuning Example")
print("=" * 60)

# 1. Load and prepare data
print("\n1. Loading and preparing data...")
adata = sc.datasets.pbmc3k()

# Add cell type labels (in practice, these would be your true labels)
# For demonstration, we'll use Leiden clusters as pseudo-labels
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.leiden(adata)
adata.obs['cell_type'] = adata.obs['leiden']

print(f"   Data: {adata.n_obs} cells x {adata.n_vars} genes")
print(f"   Cell types: {adata.obs['cell_type'].nunique()}")

# 2. Split data into train/test
print("\n2. Splitting data into train/test...")
from sklearn.model_selection import train_test_split

train_idx, test_idx = train_test_split(
    range(adata.n_obs),
    test_size=0.2,
    random_state=42
)

adata_train = adata[train_idx].copy()
adata_test = adata[test_idx].copy()

print(f"   Train: {adata_train.n_obs} cells")
print(f"   Test: {adata_test.n_obs} cells")

# 3. Quick finetuning with LoRA
print("\n3. Finetuning model with LoRA strategy...")
print("   Note: This example uses quick_finetune for simplicity")
print("   For production, see the full training example below")

try:
    model = nsc.tools.quick_finetune(
        adata_train,
        adata_eval=adata_test,
        model_name="scgpt-base-neuroscience",
        label_key="cell_type",
        num_epochs=3,  # Use more epochs in production
        batch_size=32,
        strategy="lora",  # Parameter-efficient finetuning
        output_dir="./output/finetuned_model",
        learning_rate=1e-4,
    )
    print("   ✓ Finetuning complete!")
    
    # 4. Evaluate on test set
    print("\n4. Evaluating on test set...")
    predictions = nsc.predict(model, adata_test)
    
    from neurosc.utils import compute_metrics
    metrics = compute_metrics(
        adata_test.obs['cell_type'].astype(int).values,
        predictions
    )
    
    print(f"   Test Accuracy: {metrics['accuracy']:.3f}")
    print(f"   Test F1 Score: {metrics['f1']:.3f}")
    
    # 5. Upload to HuggingFace (optional)
    print("\n5. Upload to HuggingFace Hub (optional)...")
    print("   To upload your finetuned model:")
    print("   >>> nsc.setup_huggingface(token='hf_...')")
    print("   >>> nsc.upload_to_hub(model, 'your-username/model-name')")
    
except Exception as e:
    print(f"   ⚠ Finetuning failed: {e}")
    print("   This is expected if pretrained models are not yet available")

print("\n" + "=" * 60)
print("Finetuning example complete!")
print("\nFull Training Example (Advanced):")
print("""
from neurosc.training import Trainer, TrainingArguments, LoRAConfig, finetune_model
from neurosc.data import create_dataloader

# Create dataloaders
train_loader = create_dataloader(
    adata_train,
    batch_size=32,
    return_labels=True,
    label_key="cell_type"
)

# Configure LoRA
lora_config = LoRAConfig(r=8, alpha=16, dropout=0.1)

# Finetune
model = finetune_model(
    model=model,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    num_classes=num_classes,
    strategy="lora",
    lora_config=lora_config,
    output_dir="./output",
    num_epochs=10,
    learning_rate=1e-4,
    eval_steps=100,
    save_steps=500,
)
""")
print("=" * 60)

