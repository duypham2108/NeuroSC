# NeuroSC Quick Start Guide

## The Main Workflow

NeuroSC's primary use case is simple:

1. **Load** your h5ad single-cell data file
2. **Load** a finetuned classification model
3. **Annotate** cell types automatically

## Installation

```bash
pip install neurosc
```

## Basic Usage

### Method 1: Simple Annotation

```python
import neurosc as nsc

# Annotate cell types in your data
adata = nsc.annotate_celltype(
    adata_path="brain_data.h5ad",
    model_path="username/brain-classifier"
)

# View results
print(adata.obs['cell_type'].value_counts())
```

### Method 2: With Custom Labels

```python
# Your cell type mapping (from when you trained the model)
cell_types = {
    0: 'Excitatory Neuron',
    1: 'Inhibitory Neuron',
    2: 'Astrocyte',
    3: 'Oligodendrocyte',
    4: 'Microglia'
}

# Annotate with named labels
adata = nsc.annotate_celltype(
    adata_path="brain_data.h5ad",
    model_path="username/brain-classifier",
    label_mapping=cell_types,
    return_probabilities=True
)

# Check confidence
import numpy as np
probs = adata.obsm['cell_type_probabilities']
avg_confidence = np.max(probs, axis=1).mean()
print(f"Average prediction confidence: {avg_confidence:.1%}")
```

### Method 3: Complete Workflow

```python
# Load → Annotate → Visualize → Save
adata = nsc.load_and_annotate(
    data_path="brain_data.h5ad",
    model_path="username/brain-classifier",
    save_path="results/annotated.h5ad",
    visualize=True
)
```

## Where to Get Models?

### Option 1: Use Community Models

Browse available models on HuggingFace:
```python
# Use a community model
adata = nsc.annotate_celltype(
    "data.h5ad",
    "username/brain-cell-classifier"  # From HuggingFace
)
```

### Option 2: Train Your Own

```python
import neurosc as nsc

# Prepare your training data (with known labels)
adata_train = nsc.prepare_anndata(adata_train)

# Quick finetune
model = nsc.tools.quick_finetune(
    adata_train,
    label_key="cell_type",  # Column with your labels
    num_epochs=10,
    strategy="lora"  # Fast, memory-efficient
)

# Upload to share (FREE on HuggingFace!)
nsc.setup_huggingface(token="hf_...")
nsc.upload_to_hub(model, "your-username/model-name")
```

## Common Workflows

### Just Want Predictions?

```python
adata = nsc.annotate_celltype("data.h5ad", "model-name")
print(adata.obs['cell_type'].value_counts())
```

### Need Confidence Scores?

```python
adata = nsc.annotate_celltype(
    "data.h5ad",
    "model-name",
    return_probabilities=True
)

# Access probabilities
probs = adata.obsm['cell_type_probabilities']
```

### Want to Visualize?

```python
adata = nsc.load_and_annotate(
    "data.h5ad",
    "model-name",
    visualize=True  # Creates UMAP plot
)
```

### Skip Preprocessing?

```python
# If your data is already preprocessed
adata = nsc.annotate_celltype(
    "preprocessed_data.h5ad",
    "model-name",
    preprocess=False
)
```

## Example: End-to-End

```python
import neurosc as nsc

# 1. Load and annotate
adata = nsc.annotate_celltype(
    adata_path="brain_sample.h5ad",
    model_path="neurosc/brain-cell-classifier",
    label_mapping={
        0: 'Excitatory Neuron',
        1: 'Inhibitory Neuron',
        2: 'Astrocyte',
        3: 'Oligodendrocyte',
        4: 'Microglia'
    },
    return_probabilities=True
)

# 2. Check results
print(f"Annotated {adata.n_obs:,} cells")
print("\nCell type distribution:")
print(adata.obs['cell_type'].value_counts())

# 3. Check confidence
import numpy as np
probs = adata.obsm['cell_type_probabilities']
confidence = np.max(probs, axis=1)
print(f"\nAverage confidence: {confidence.mean():.1%}")
print(f"Low confidence cells (<50%): {(confidence < 0.5).sum()}")

# 4. Visualize
import scanpy as sc
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=['cell_type'], save='_cell_types.png')

# 5. Save
adata.write_h5ad("annotated_brain_cells.h5ad")
```

## Next Steps

- **Train a model**: See `examples/02_finetuning.py`
- **Upload to share**: See `examples/06_upload_model.py`
- **Advanced usage**: See `examples/04_scanpy_integration.py`
- **Full documentation**: See `README.md`

## Troubleshooting

### Model Not Found?
Make sure the model exists on HuggingFace or use a local path:
```python
# HuggingFace model
adata = nsc.annotate_celltype("data.h5ad", "username/model-name")

# Local model
adata = nsc.annotate_celltype("data.h5ad", "./my_model/")
```

### Memory Issues?
Reduce batch size:
```python
adata = nsc.annotate_celltype(
    "data.h5ad",
    "model-name",
    batch_size=32  # Default is 64
)
```

### GPU Not Detected?
Specify device:
```python
adata = nsc.annotate_celltype(
    "data.h5ad",
    "model-name",
    device="cpu"  # or "cuda"
)
```

## Getting Help

- **Documentation**: See full [README.md](README.md)
- **Examples**: Check `examples/` directory
- **Issues**: https://github.com/yourusername/NeuroSC/issues

---

**That's it! You're ready to annotate cell types with NeuroSC! 🧠**

