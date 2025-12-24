# NeuroSC Examples

This directory contains example scripts demonstrating various use cases of NeuroSC.

## Example Scripts

### 0. Simple Workflow (`00_simple_workflow.py`) ⭐ START HERE
**The main use case** - Load data, load model, annotate cell types:
- All-in-one annotation function
- Custom label mapping
- Prediction probabilities
- Different annotation methods

**Run:**
```bash
python examples/00_simple_workflow.py
```

### 1. Quick Start (`01_quick_start.py`)
Basic workflow showing how to:
- Load and preprocess data
- Load pretrained models
- Generate embeddings
- Perform downstream analysis

**Run:**
```bash
python examples/01_quick_start.py
```

### 2. Finetuning (`02_finetuning.py`)
Demonstrates model finetuning with:
- Data preparation for training
- LoRA (parameter-efficient) finetuning
- Model evaluation
- Uploading to HuggingFace

**Run:**
```bash
python examples/02_finetuning.py
```

### 3. Cell Type Annotation (`03_cell_type_annotation.py`)
Shows automatic cell type annotation:
- Using pretrained models
- Label transfer from reference datasets
- Visualization of results

**Run:**
```bash
python examples/03_cell_type_annotation.py
```

### 4. Scanpy Integration (`04_scanpy_integration.py`)
Demonstrates seamless integration with scanpy:
- Standard scanpy preprocessing
- Replacing PCA with foundation model embeddings
- Using scanpy's downstream analysis tools

**Run:**
```bash
python examples/04_scanpy_integration.py
```

### 5. Batch Integration (`05_batch_integration.py`)
Batch integration using foundation models:
- Handling multiple batches
- Integration without explicit correction
- Quality metrics

**Run:**
```bash
python examples/05_batch_integration.py
```

### 6. Upload Model (`06_upload_model.py`)
Share your finetuned model on HuggingFace:
- Setting up HuggingFace authentication
- Uploading models (FREE hosting!)
- Creating model cards
- Sharing with the community

**Run:**
```bash
python examples/06_upload_model.py
```

## Jupyter Notebooks

For interactive exploration, check out the notebooks:
- `tutorial_basic.ipynb` - Basic tutorial
- `tutorial_advanced.ipynb` - Advanced features

## Requirements

All examples require the base NeuroSC installation:
```bash
pip install neurosc
```

Some examples may require additional packages:
```bash
pip install jupyter  # For notebooks
```

## Notes

- Some examples may show warnings about pretrained models not being available yet. This is expected during initial setup.
- The examples use the PBMC3k dataset from scanpy for demonstration purposes.
- For production use, replace the example data with your own neuroscience scRNA-seq datasets.

## Getting Help

If you encounter issues:
1. Check the main [README](../README.md)
2. Visit our [documentation](https://neurosc.readthedocs.io)
3. Open an [issue](https://github.com/yourusername/NeuroSC/issues)

