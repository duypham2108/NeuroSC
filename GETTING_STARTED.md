# Getting Started with NeuroSC

Welcome to NeuroSC! This guide will help you get started with the package.

## Installation

### Option 1: Install from source (current setup)

Since you've just initialized the project, install in development mode:

```bash
cd NeuroSC
pip install -e .
```

This will install NeuroSC along with all required dependencies.

### Option 2: Install from PyPI (once published)

```bash
pip install neurosc
```

## Quick Test

Test your installation:

```python
import neurosc as nsc
print(f"NeuroSC version: {nsc.__version__}")
print("Available modules:", dir(nsc))
```

## Running Examples

The `examples/` directory contains ready-to-run scripts:

```bash
# Quick start
python examples/01_quick_start.py

# Finetuning tutorial
python examples/02_finetuning.py

# Cell type annotation
python examples/03_cell_type_annotation.py

# Scanpy integration
python examples/04_scanpy_integration.py

# Batch integration
python examples/05_batch_integration.py
```

## Basic Usage

### 1. Load and Preprocess Data

```python
import neurosc as nsc
import scanpy as sc

# Load your data
adata = sc.read_h5ad("your_data.h5ad")

# Preprocess
adata = nsc.prepare_anndata(
    adata,
    min_genes=200,
    highly_variable_genes=3000
)
```

### 2. Load Model and Generate Embeddings

```python
# Load pretrained model
model = nsc.load_model("scgpt-base-neuroscience")

# Generate embeddings
adata = nsc.embed(model, adata, key_added="X_scgpt")
```

### 3. Downstream Analysis

```python
# Use with scanpy
sc.pp.neighbors(adata, use_rep="X_scgpt")
sc.tl.umap(adata)
sc.tl.leiden(adata)
sc.pl.umap(adata, color='leiden')
```

## Next Steps

### For Users

1. **Try the examples** - Run the example scripts to see NeuroSC in action
2. **Read the documentation** - Check the comprehensive README.md
3. **Explore the API** - See API_REFERENCE.md for detailed function documentation
4. **Join the community** - Share your models on HuggingFace Hub

### For Developers

1. **Set up development environment**:
   ```bash
   pip install -e ".[dev]"
   ```

2. **Run tests** (once test suite is added):
   ```bash
   pytest tests/
   ```

3. **Format code**:
   ```bash
   black neurosc/
   flake8 neurosc/
   ```

4. **Contribute** - See CONTRIBUTING.md for guidelines

## Project Roadmap

### Phase 1: Initial Release ✓
- [x] Core package structure
- [x] Data preprocessing utilities
- [x] Model wrappers (scGPT)
- [x] Training and finetuning
- [x] Inference utilities
- [x] Scanpy-compatible API
- [x] HuggingFace integration
- [x] Documentation and examples

### Phase 2: Model Hosting (Next Steps)
- [ ] Upload pretrained models to HuggingFace Hub
- [ ] Create model cards and documentation
- [ ] Benchmark on neuroscience datasets
- [ ] Publish model performance metrics

### Phase 3: Community & Extensions
- [ ] Add more foundation model wrappers (Geneformer, etc.)
- [ ] Interactive tutorials and notebooks
- [ ] Video tutorials
- [ ] Community model contributions
- [ ] Integration with more single-cell tools

### Phase 4: Advanced Features
- [ ] Multi-modal integration (ATAC-seq, proteomics)
- [ ] Spatial transcriptomics support
- [ ] Transfer learning utilities
- [ ] Model compression and optimization
- [ ] Web interface for model exploration

## Uploading Your First Pretrained Model

To make pretrained models available to the community:

1. **Train or finetune a model**:
   ```python
   model = nsc.tools.quick_finetune(
       adata_train,
       label_key="cell_type",
       num_epochs=10
   )
   ```

2. **Set up HuggingFace**:
   ```python
   nsc.setup_huggingface(token="hf_...")
   ```

3. **Upload to Hub** (FREE!):
   ```python
   nsc.upload_to_hub(
       model,
       repo_id="your-username/scgpt-neuroscience",
       commit_message="Pretrained on cortical neurons"
   )
   ```

4. **Update model registry**:
   - Add your model to `neurosc/models/model_registry.py`
   - Submit a PR to make it available to everyone!

## Common Issues

### Model Loading Fails
If you see warnings about models not being available:
- This is expected before pretrained models are uploaded
- The package structure is complete and ready to use
- Upload your first model following the guide above

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### CUDA/GPU Issues
For GPU support, install PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Resources

- **Documentation**: See README.md
- **Examples**: Check `examples/` directory
- **API Reference**: Browse the code or generate docs with Sphinx
- **Issues**: https://github.com/yourusername/NeuroSC/issues
- **Discussions**: https://github.com/yourusername/NeuroSC/discussions

## Support

Need help?
1. Check the documentation
2. Look at example scripts
3. Search existing issues
4. Ask in discussions
5. Open a new issue

---

**Welcome to the NeuroSC community! 🧠🧬**

