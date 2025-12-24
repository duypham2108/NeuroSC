"""
Quick Start Example - Basic NeuroSC Usage
==========================================

This script demonstrates the basic workflow of using NeuroSC for
single-cell analysis with foundation models.
"""

import neurosc as nsc
import scanpy as sc
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("NeuroSC Quick Start Example")
print("=" * 60)

# 1. Load example data
# For this example, we'll create synthetic data
# In practice, you would load your own data:
# adata = sc.read_h5ad("your_data.h5ad")

print("\n1. Loading example data...")
adata = sc.datasets.pbmc3k()
print(f"   Loaded data: {adata.n_obs} cells x {adata.n_vars} genes")

# 2. Preprocess data
print("\n2. Preprocessing data...")
adata = nsc.prepare_anndata(
    adata,
    target_sum=1e4,
    min_genes=200,
    min_cells=3,
    highly_variable_genes=2000
)
print(f"   After filtering: {adata.n_obs} cells x {adata.n_vars} genes")

# 3. Load pretrained model
print("\n3. Loading pretrained model...")
try:
    model = nsc.load_model("scgpt-base-neuroscience")
    print("   ✓ Model loaded successfully")
except Exception as e:
    print(f"   ⚠ Could not load pretrained model: {e}")
    print("   This is expected if models are not yet available online")
    print("   The package is ready to use once models are uploaded!")

# 4. Generate embeddings
print("\n4. Generating embeddings...")
# Note: This will use random embeddings if model loading failed
try:
    adata = nsc.embed(model, adata, key_added="X_scgpt")
    print("   ✓ Embeddings generated")
except Exception as e:
    print(f"   Using fallback method: {e}")
    # Fallback to PCA for demonstration
    sc.tl.pca(adata)
    print("   ✓ Using PCA embeddings as fallback")

# 5. Downstream analysis with scanpy
print("\n5. Running downstream analysis...")
sc.pp.neighbors(adata, use_rep="X_pca" if "X_scgpt" not in adata.obsm else "X_scgpt")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
print("   ✓ Clustering complete")

# 6. Visualization
print("\n6. Visualizing results...")
sc.pl.umap(adata, color=['leiden'], save='_neurosc_quick_start.png')
print("   ✓ UMAP plot saved")

print("\n" + "=" * 60)
print("Quick start complete!")
print("Next steps:")
print("  - Try finetuning: examples/02_finetuning.py")
print("  - Explore annotation: examples/03_cell_type_annotation.py")
print("=" * 60)

