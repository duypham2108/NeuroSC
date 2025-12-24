"""
Cell Type Annotation Example
=============================

This script demonstrates automatic cell type annotation using
foundation models and label transfer.
"""

import neurosc as nsc
import scanpy as sc

print("=" * 60)
print("NeuroSC Cell Type Annotation Example")
print("=" * 60)

# 1. Load query data (unlabeled)
print("\n1. Loading query data...")
adata_query = sc.datasets.pbmc3k()
adata_query = nsc.prepare_anndata(
    adata_query,
    min_genes=200,
    highly_variable_genes=2000
)
print(f"   Query data: {adata_query.n_obs} cells")

# 2. Automatic annotation using pretrained model
print("\n2. Automatic cell type annotation...")
try:
    nsc.tl.annotate_cells(
        adata_query,
        model_name="scgpt-base-neuroscience",
        key_added="predicted_cell_type"
    )
    
    print("   ✓ Annotation complete!")
    print("\n   Predicted cell types:")
    print(adata_query.obs['predicted_cell_type'].value_counts())
    
except Exception as e:
    print(f"   ⚠ Annotation with pretrained model not available: {e}")

# 3. Label transfer from reference (alternative approach)
print("\n3. Label transfer from reference dataset...")

# Load reference data (with known labels)
adata_ref = sc.datasets.pbmc3k()

# Add some pseudo-labels for demonstration
sc.pp.filter_cells(adata_ref, min_genes=200)
sc.pp.filter_genes(adata_ref, min_cells=3)
sc.pp.normalize_total(adata_ref, target_sum=1e4)
sc.pp.log1p(adata_ref)
sc.pp.highly_variable_genes(adata_ref, n_top_genes=2000)
sc.tl.pca(adata_ref)
sc.pp.neighbors(adata_ref)
sc.tl.leiden(adata_ref)

# Map clusters to cell types (pseudo-labels)
cluster_to_celltype = {
    '0': 'CD4 T cells',
    '1': 'CD8 T cells',
    '2': 'B cells',
    '3': 'NK cells',
    '4': 'Monocytes',
}
adata_ref.obs['cell_type'] = adata_ref.obs['leiden'].map(
    lambda x: cluster_to_celltype.get(x, 'Unknown')
)

print(f"   Reference data: {adata_ref.n_obs} cells")
print(f"   Cell types: {adata_ref.obs['cell_type'].unique().tolist()}")

try:
    # Transfer labels
    nsc.tl.transfer_labels(
        adata_ref=adata_ref,
        adata_query=adata_query,
        label_key="cell_type",
        key_added="transferred_labels"
    )
    
    print("   ✓ Label transfer complete!")
    print("\n   Transferred labels:")
    print(adata_query.obs['transferred_labels'].value_counts())
    
except Exception as e:
    print(f"   ⚠ Label transfer requires pretrained models: {e}")

# 4. Visualize results
print("\n4. Visualizing results...")
sc.pp.neighbors(adata_query)
sc.tl.umap(adata_query)

if 'predicted_cell_type' in adata_query.obs:
    sc.pl.umap(
        adata_query,
        color=['predicted_cell_type'],
        save='_annotation.png'
    )
    print("   ✓ Visualization saved")

print("\n" + "=" * 60)
print("Cell type annotation example complete!")
print("=" * 60)

