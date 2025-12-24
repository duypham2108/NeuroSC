"""
Scanpy Integration Example
===========================

This script demonstrates how NeuroSC integrates seamlessly with
the standard scanpy workflow.
"""

import scanpy as sc
import neurosc as nsc
import numpy as np

print("=" * 60)
print("NeuroSC + Scanpy Integration Example")
print("=" * 60)

# Standard scanpy preprocessing workflow
print("\n1. Standard scanpy preprocessing...")
adata = sc.datasets.pbmc3k()

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Calculate QC metrics
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(
    adata,
    qc_vars=['mt'],
    percent_top=None,
    log1p=False,
    inplace=True
)

# Filter
adata = adata[adata.obs.pct_counts_mt < 20, :]

# Normalize
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Feature selection
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

print(f"   Preprocessed data: {adata.n_obs} cells x {adata.n_vars} genes")

# Replace PCA with NeuroSC embeddings
print("\n2. Generate embeddings with NeuroSC (replaces sc.tl.pca)...")
try:
    nsc.tl.embed_cells(adata, key_added="X_scgpt")
    use_rep = "X_scgpt"
    print("   ✓ Using foundation model embeddings")
except Exception as e:
    print(f"   Using PCA as fallback: {e}")
    sc.tl.pca(adata)
    use_rep = "X_pca"

# Continue with standard scanpy workflow
print("\n3. Continue with standard scanpy workflow...")
sc.pp.neighbors(adata, use_rep=use_rep)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)

# Alternative: Use NeuroSC clustering
print("\n4. Alternative: Use NeuroSC clustering...")
try:
    nsc.tl.cluster_cells(
        adata,
        use_rep=use_rep,
        method="leiden",
        resolution=0.5,
        key_added="neurosc_clusters"
    )
    print("   ✓ NeuroSC clustering complete")
except Exception as e:
    print(f"   Clustering: {e}")

# Visualization
print("\n5. Visualization with scanpy...")
sc.pl.umap(adata, color=['leiden'], save='_integration_leiden.png')
print("   ✓ Visualizations saved")

# Find marker genes (standard scanpy)
print("\n6. Find marker genes (standard scanpy)...")
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes=10, save='_markers.png')
print("   ✓ Marker genes identified")

print("\n" + "=" * 60)
print("Integration example complete!")
print("\nKey takeaway:")
print("  NeuroSC embeddings can replace PCA in your scanpy workflow")
print("  Everything else works exactly the same!")
print("=" * 60)

