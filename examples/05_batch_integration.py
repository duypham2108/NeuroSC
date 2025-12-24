"""
Batch Integration Example
==========================

This script demonstrates batch integration using foundation model
embeddings.
"""

import scanpy as sc
import neurosc as nsc
import numpy as np

print("=" * 60)
print("NeuroSC Batch Integration Example")
print("=" * 60)

# 1. Load data with batches
print("\n1. Creating synthetic batched data...")
adata = sc.datasets.pbmc3k()

# Add synthetic batch labels
np.random.seed(42)
n_obs = adata.n_obs
adata.obs['batch'] = np.random.choice(['Batch1', 'Batch2', 'Batch3'], size=n_obs)

print(f"   Data: {adata.n_obs} cells")
print(f"   Batches: {adata.obs['batch'].value_counts().to_dict()}")

# 2. Preprocess
print("\n2. Preprocessing...")
adata = nsc.prepare_anndata(
    adata,
    min_genes=200,
    highly_variable_genes=2000,
    batch_key='batch'
)

# 3. Integration with NeuroSC
print("\n3. Batch integration with foundation model...")
try:
    nsc.tl.integrate_batches(
        adata,
        batch_key='batch',
        key_added='X_integrated'
    )
    
    # Compute neighbors and UMAP on integrated embeddings
    sc.pp.neighbors(adata, use_rep='X_integrated')
    sc.tl.umap(adata)
    sc.tl.leiden(adata)
    
    print("   ✓ Integration complete!")
    
    # 4. Visualize integration
    print("\n4. Visualizing integration...")
    sc.pl.umap(adata, color=['batch', 'leiden'], save='_batch_integration.png')
    print("   ✓ Visualization saved")
    
    # 5. Quantify integration
    print("\n5. Integration quality metrics...")
    # Note: In practice, you would use metrics like:
    # - Batch mixing entropy
    # - k-nearest neighbor batch effect
    # - Silhouette score
    
    from sklearn.metrics import silhouette_score
    if 'X_integrated' in adata.obsm:
        # Calculate batch effect (lower is better)
        batch_labels = adata.obs['batch'].astype('category').cat.codes
        batch_silhouette = silhouette_score(
            adata.obsm['X_integrated'],
            batch_labels
        )
        print(f"   Batch silhouette score: {batch_silhouette:.3f}")
        print("   (Lower scores indicate better batch mixing)")
    
except Exception as e:
    print(f"   ⚠ Integration requires pretrained models: {e}")
    print("   Falling back to standard methods...")
    
    # Fallback to combat or harmony
    sc.pp.pca(adata)
    sc.external.pp.bbknn(adata, batch_key='batch')
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=['batch'], save='_batch_integration_fallback.png')

print("\n" + "=" * 60)
print("Batch integration example complete!")
print("\nNote:")
print("  Foundation models can learn batch-invariant representations")
print("  reducing the need for explicit batch correction methods")
print("=" * 60)

