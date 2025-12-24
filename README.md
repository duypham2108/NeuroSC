# NeuroSC 🧠

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Models%20on%20Hub-orange)](https://huggingface.co/neurosc)

**NeuroSC** (Neuroscience Single-Cell) is an open-source Python package for finetuning and deploying foundation single-cell models specifically tailored for neuroscience scRNA-seq data.

Built on top of foundation models like [scGPT](https://github.com/bowang-lab/scGPT), NeuroSC provides:

- 🚀 **Easy finetuning** of foundation models on your neuroscience datasets
- 🔗 **Scanpy-compatible API** for seamless integration into existing workflows  
- 🤗 **HuggingFace Hub integration** for free model hosting and sharing
- 📦 **Pre-trained models** optimized for brain scRNA-seq data
- 🧰 **Comprehensive tools** for embedding, clustering, annotation, and more

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [Usage Examples](#usage-examples)
  - [Data Preprocessing](#data-preprocessing)
  - [Loading Pretrained Models](#loading-pretrained-models)
  - [Generating Embeddings](#generating-embeddings)
  - [Finetuning Models](#finetuning-models)
  - [Cell Type Annotation](#cell-type-annotation)
  - [Batch Integration](#batch-integration)
- [Scanpy-Compatible Workflow](#scanpy-compatible-workflow)
- [HuggingFace Integration](#huggingface-integration)
- [Available Pretrained Models](#available-pretrained-models)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Installation

### From PyPI (recommended)

```bash
pip install neurosc
```

### From source

```bash
git clone https://github.com/yourusername/NeuroSC.git
cd NeuroSC
pip install -e .
```

### Dependencies

NeuroSC requires Python 3.8+ and the following core dependencies:
- `scanpy` - Single-cell analysis
- `anndata` - Annotated data structures
- `torch` - Deep learning framework
- `transformers` - Foundation model utilities
- `huggingface-hub` - Model hosting and sharing

All dependencies are automatically installed with the package.

---

## Quick Start

```python
import neurosc as nsc
import scanpy as sc

# Load your data
adata = sc.read_h5ad("your_neuroscience_data.h5ad")

# Preprocess with NeuroSC
adata = nsc.prepare_anndata(adata, min_genes=200, highly_variable_genes=3000)

# Load a pretrained model
model = nsc.load_model("scgpt-base-neuroscience")

# Generate embeddings
adata = nsc.embed(model, adata, key_added="X_scgpt")

# Continue with scanpy workflow
sc.pp.neighbors(adata, use_rep="X_scgpt")
sc.tl.umap(adata)
sc.tl.leiden(adata)
sc.pl.umap(adata, color=['leiden', 'cell_type'])
```

---

## Features

### 🎯 Foundation Model Support

- **scGPT**: State-of-the-art transformer model for single-cell data
- **Custom models**: Easy integration of new foundation models
- **Pretrained weights**: Access neuroscience-optimized checkpoints

### 🔧 Finetuning Strategies

- **Full finetuning**: Update all model parameters
- **LoRA** (Low-Rank Adaptation): Parameter-efficient finetuning
- **Freeze backbone**: Only train classification head

### 🧬 Scanpy Integration

NeuroSC provides a `tl` (tools) module with scanpy-style functions:

```python
import neurosc as nsc

# Use like scanpy
nsc.tl.embed_cells(adata)              # Instead of sc.tl.pca
nsc.tl.cluster_cells(adata)            # Clustering with embeddings
nsc.tl.annotate_cells(adata)           # Automatic cell type annotation
nsc.tl.integrate_batches(adata, "batch")  # Batch integration
```

### 🤗 HuggingFace Hub Integration

Upload and share your finetuned models for free:

```python
# Finetune your model
model = nsc.load_model("scgpt-base-neuroscience")
# ... finetuning code ...

# Upload to HuggingFace (free hosting!)
nsc.upload_to_hub(
    model, 
    repo_id="your-username/my-neuroscience-model",
    commit_message="Finetuned on cortical neurons"
)

# Anyone can now use your model:
model = nsc.load_model("your-username/my-neuroscience-model")
```

---

## Usage Examples

### Data Preprocessing

NeuroSC provides scanpy-compatible preprocessing with optimizations for foundation models:

```python
import neurosc as nsc
import scanpy as sc

# Load data
adata = sc.read_h5ad("brain_data.h5ad")

# Preprocess for foundation models
adata = nsc.prepare_anndata(
    adata,
    target_sum=1e4,              # Normalize to 10k counts
    min_genes=200,               # Filter low-quality cells
    min_cells=3,                 # Filter rare genes
    pct_counts_mt_threshold=20,  # Filter high-mt% cells
    highly_variable_genes=3000,  # Select HVGs
    batch_key="batch"            # For batch-aware processing
)

# Validate data
nsc.data.validate_anndata(adata, verbose=True)
```

### Loading Pretrained Models

```python
import neurosc as nsc

# List available models
models = nsc.list_pretrained_models()
print(models)
# ['scgpt-base-neuroscience', 'scgpt-large-neuroscience', ...]

# Load a pretrained model
model = nsc.load_model("scgpt-base-neuroscience")

# Or load from local path
model = nsc.load_model("./my_model_checkpoint/")

# Or from HuggingFace Hub
model = nsc.load_model("username/model-name")
```

### Generating Embeddings

```python
import neurosc as nsc
import scanpy as sc

# Generate embeddings with foundation model
adata = nsc.embed(model, adata, key_added="X_scgpt")

# Use embeddings for downstream analysis
sc.pp.neighbors(adata, use_rep="X_scgpt")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)

# Visualize
sc.pl.umap(adata, color=['leiden', 'cell_type'])
```

### Finetuning Models

#### Full Finetuning

```python
from neurosc.training import finetune_model
from neurosc.data import create_dataloader

# Create dataloaders
train_loader = create_dataloader(
    adata_train,
    batch_size=32,
    return_labels=True,
    label_key="cell_type"
)

eval_loader = create_dataloader(
    adata_eval,
    batch_size=64,
    return_labels=True,
    label_key="cell_type"
)

# Finetune
model = finetune_model(
    model=model,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    num_classes=10,
    strategy="full",
    output_dir="./finetuned_model",
    num_epochs=10,
    learning_rate=1e-4
)
```

#### LoRA Finetuning (Parameter-Efficient)

```python
from neurosc.training import finetune_model, LoRAConfig

# Configure LoRA
lora_config = LoRAConfig(
    r=8,              # Rank
    alpha=16,         # Scaling
    dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

# Finetune with LoRA (much faster, less memory)
model = finetune_model(
    model=model,
    train_dataloader=train_loader,
    strategy="lora",
    lora_config=lora_config,
    output_dir="./lora_model",
    num_epochs=5
)
```

#### Quick Finetuning

```python
import neurosc as nsc

# One-line finetuning with sensible defaults
model = nsc.tools.quick_finetune(
    adata_train,
    adata_eval=adata_eval,
    label_key="cell_type",
    num_epochs=10,
    strategy="lora"
)
```

### Cell Type Annotation

```python
import neurosc as nsc

# Automatic cell type annotation
nsc.tl.annotate_cells(adata, model_name="scgpt-base-neuroscience")

# View predictions
print(adata.obs['predicted_cell_type'].value_counts())

# Transfer labels from reference to query
nsc.tl.transfer_labels(
    adata_ref=reference_data,
    adata_query=new_data,
    label_key="cell_type"
)
```

### Batch Integration

```python
import neurosc as nsc
import scanpy as sc

# Integrate batches using foundation model embeddings
nsc.tl.integrate_batches(adata, batch_key="batch")

# Visualize integration
sc.pp.neighbors(adata, use_rep="X_integrated")
sc.tl.umap(adata)
sc.pl.umap(adata, color=['batch', 'cell_type'])
```

---

## Scanpy-Compatible Workflow

NeuroSC is designed to integrate seamlessly with scanpy workflows:

```python
import scanpy as sc
import neurosc as nsc

# Standard scanpy preprocessing
adata = sc.read_h5ad("data.h5ad")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Use NeuroSC for embedding (replaces PCA)
nsc.tl.embed_cells(adata, key_added="X_scgpt")

# Continue with scanpy
sc.pp.neighbors(adata, use_rep="X_scgpt")
sc.tl.umap(adata)
sc.tl.leiden(adata)

# Or use NeuroSC's clustering
nsc.tl.cluster_cells(adata, use_rep="X_scgpt", resolution=0.5)

# Automatic annotation
nsc.tl.annotate_cells(adata)

# Standard scanpy plotting
sc.pl.umap(adata, color=['leiden', 'predicted_cell_type'])
```

---

## HuggingFace Integration

NeuroSC leverages HuggingFace Hub for **free model hosting and sharing**:

### Setup

```python
import neurosc as nsc

# Authenticate with HuggingFace
nsc.setup_huggingface(token="hf_...")

# Or use environment variable
# export HF_TOKEN=hf_...
```

### Upload Your Model

```python
# After finetuning
nsc.upload_to_hub(
    model,
    repo_id="your-username/scgpt-cortical-neurons",
    commit_message="Finetuned on mouse cortical neurons",
    private=False  # Make it public for the community!
)
```

### Download and Use Community Models

```python
# Load from HuggingFace
model = nsc.load_model("username/awesome-neuroscience-model")

# Or download first
model_path = nsc.download_pretrained("username/awesome-neuroscience-model")
model = nsc.load_model(model_path)
```

---

## Available Pretrained Models

| Model Name | Parameters | Description | Tissue | Species |
|------------|-----------|-------------|---------|---------|
| `scgpt-base-neuroscience` | 90M | Base scGPT finetuned on neuroscience data | Brain | Human/Mouse |
| `scgpt-large-neuroscience` | 300M | Large scGPT for neuroscience | Brain | Human/Mouse |
| `scgpt-base-general` | 90M | General purpose scGPT | Multi-tissue | Human |

### Downloading Pretrained Models

```python
import neurosc as nsc

# List available models
models = nsc.list_pretrained_models()

# Download a specific model
model_path = nsc.download_pretrained("scgpt-base-neuroscience")

# Load the model
model = nsc.load_model(model_path)
```

**Note**: Some pretrained models are hosted on HuggingFace Hub and require internet connection to download. Once downloaded, they are cached locally.

---

## API Reference

### Core Functions

- `nsc.load_model(model_name)` - Load a pretrained model
- `nsc.list_pretrained_models()` - List available models
- `nsc.prepare_anndata(adata, ...)` - Preprocess data for models
- `nsc.embed(model, adata, ...)` - Generate cell embeddings
- `nsc.predict(model, adata, ...)` - Make predictions

### Tools Module (`nsc.tl`)

Scanpy-compatible high-level functions:

- `nsc.tl.embed_cells(adata, ...)` - Embed cells (like `sc.tl.pca`)
- `nsc.tl.cluster_cells(adata, ...)` - Cluster with embeddings
- `nsc.tl.annotate_cells(adata, ...)` - Automatic cell type annotation
- `nsc.tl.integrate_batches(adata, ...)` - Batch integration
- `nsc.tl.transfer_labels(...)` - Transfer labels between datasets

### Training Module (`nsc.training`)

- `finetune_model(...)` - Finetune a foundation model
- `Trainer` - Custom trainer class
- `TrainingArguments` - Training configuration
- `LoRAConfig` - LoRA configuration

### HuggingFace Module (`nsc.tools`)

- `nsc.setup_huggingface(token)` - Setup HF authentication
- `nsc.upload_to_hub(model, repo_id, ...)` - Upload model to HF Hub
- `nsc.download_from_hub(repo_id, ...)` - Download from HF Hub
- `nsc.download_pretrained(model_name, ...)` - Download pretrained model

### Data Module (`nsc.data`)

- `prepare_anndata(adata, ...)` - Preprocessing pipeline
- `create_dataloader(adata, ...)` - Create PyTorch DataLoader
- `SingleCellDataset` - PyTorch Dataset for single-cell data
- `tokenize_genes(adata, gene_vocab, ...)` - Gene tokenization
- `create_gene_vocabulary(gene_names, ...)` - Create gene vocabulary

---

## Project Structure

```
NeuroSC/
├── neurosc/
│   ├── __init__.py
│   ├── data/              # Data preprocessing and datasets
│   │   ├── preprocessing.py
│   │   └── dataset.py
│   ├── models/            # Foundation models
│   │   ├── base.py
│   │   ├── scgpt_wrapper.py
│   │   └── model_registry.py
│   ├── training/          # Training utilities
│   │   ├── trainer.py
│   │   ├── finetune.py
│   │   └── callbacks.py
│   ├── inference/         # Inference and prediction
│   │   ├── predict.py
│   │   └── interpret.py
│   ├── tl/                # Scanpy-compatible tools
│   │   ├── embedding.py
│   │   ├── clustering.py
│   │   ├── annotation.py
│   │   └── integration.py
│   ├── tools/             # High-level tools and workflows
│   │   ├── huggingface.py
│   │   ├── download.py
│   │   └── workflows.py
│   └── utils/             # Utilities
│       ├── metrics.py
│       └── visualization.py
├── examples/              # Example scripts and notebooks
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── setup.py
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/yourusername/NeuroSC.git
cd NeuroSC
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/
```

---

## Citation

If you use NeuroSC in your research, please cite:

```bibtex
@software{neurosc2025,
  title={NeuroSC: Foundation Models for Neuroscience Single-Cell Analysis},
  author={NeuroSC Contributors},
  year={2025},
  url={https://github.com/yourusername/NeuroSC}
}
```

Please also cite the underlying foundation models you use (e.g., scGPT).

---

## License

NeuroSC is released under the [MIT License](LICENSE).

---

## Acknowledgments

- [scGPT](https://github.com/bowang-lab/scGPT) - Foundation model architecture
- [scanpy](https://github.com/scverse/scanpy) - Single-cell analysis framework
- [HuggingFace](https://huggingface.co/) - Model hosting infrastructure

---

## Contact

- GitHub Issues: [https://github.com/yourusername/NeuroSC/issues](https://github.com/yourusername/NeuroSC/issues)
- Email: your.email@example.com

---

**Happy analyzing! 🧬🧠**