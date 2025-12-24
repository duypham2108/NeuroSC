"""
Simple Workflow - The Main Use Case
====================================

This example demonstrates the primary workflow of NeuroSC:
1. Load your h5ad single-cell data file
2. Load a finetuned classification model
3. Annotate cell types automatically

This is the simplest and most common use case.
"""

import neurosc as nsc

print("=" * 60)
print("NeuroSC Simple Workflow - Cell Type Annotation")
print("=" * 60)

# ============================================================================
# Method 1: All-in-one function (SIMPLEST)
# ============================================================================

print("\n" + "=" * 60)
print("Method 1: All-in-one annotation")
print("=" * 60)

try:
    adata = nsc.annotate_celltype(
        adata_path="data/brain_cells.h5ad",  # Your h5ad file
        model_path="username/scgpt-brain-classifier",  # Finetuned model
        output_key="cell_type",  # Where to store predictions
    )
    
    # That's it! Your data now has cell type annotations
    print(f"\nAnnotated {adata.n_obs} cells")
    print(f"Cell types: {adata.obs['cell_type'].unique().tolist()}")
    
except FileNotFoundError:
    print("\n⚠ Example data file not found. Using demo data...")
    
    # For demonstration, use built-in dataset
    import scanpy as sc
    adata = sc.datasets.pbmc3k()
    
    print("\nUsing PBMC3k demo data for illustration...")
    try:
        adata = nsc.annotate_celltype(
            adata_path=adata,  # Can also pass AnnData object directly
            model_path="username/scgpt-pbmc-classifier",  # Your model
        )
    except Exception as e:
        print(f"Model loading will work once you upload your finetuned model: {e}")
        print("\nSee examples/02_finetuning.py for how to create a model")


# ============================================================================
# Method 2: One-liner with file save
# ============================================================================

print("\n" + "=" * 60)
print("Method 2: Quick annotate with save")
print("=" * 60)

try:
    adata = nsc.quick_annotate(
        h5ad_file="data/brain_cells.h5ad",
        model="username/brain-classifier",
        output_file="results/annotated_cells.h5ad"  # Save results
    )
except Exception as e:
    print(f"Will work with real data: {e}")


# ============================================================================
# Method 3: Complete workflow with visualization
# ============================================================================

print("\n" + "=" * 60)
print("Method 3: Complete workflow (annotate + visualize)")
print("=" * 60)

try:
    adata = nsc.load_and_annotate(
        data_path="data/brain_cells.h5ad",
        model_path="username/brain-classifier",
        save_path="results/annotated.h5ad",
        visualize=True  # Creates UMAP plot
    )
except Exception as e:
    print(f"Will work with real data: {e}")


# ============================================================================
# Method 4: With custom cell type names
# ============================================================================

print("\n" + "=" * 60)
print("Method 4: Custom cell type labels")
print("=" * 60)

# Define your cell type mapping (from model training)
cell_type_mapping = {
    0: 'Excitatory Neuron',
    1: 'Inhibitory Neuron', 
    2: 'Astrocyte',
    3: 'Oligodendrocyte',
    4: 'Microglia',
    5: 'Endothelial',
    6: 'OPC'
}

try:
    adata = nsc.annotate_celltype(
        adata_path="data/brain_cells.h5ad",
        model_path="username/brain-classifier",
        label_mapping=cell_type_mapping,  # Map indices to names
        return_probabilities=True,  # Also get prediction confidence
    )
    
    # Access probabilities
    probs = adata.obsm['cell_type_probabilities']
    print(f"\nPrediction probabilities shape: {probs.shape}")
    
    # Check confidence
    import numpy as np
    max_probs = np.max(probs, axis=1)
    print(f"Average prediction confidence: {np.mean(max_probs):.2%}")
    
except Exception as e:
    print(f"Will work with real data: {e}")


# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 60)
print("Summary - Choose Your Method:")
print("=" * 60)
print("""
1. SIMPLEST - Just annotate:
   >>> adata = nsc.annotate_celltype("data.h5ad", "model")

2. Quick with save:
   >>> nsc.quick_annotate("data.h5ad", "model", output_file="out.h5ad")

3. With visualization:
   >>> nsc.load_and_annotate("data.h5ad", "model", visualize=True)

4. Custom labels + probabilities:
   >>> adata = nsc.annotate_celltype(
   ...     "data.h5ad", "model",
   ...     label_mapping={0: 'Type1', 1: 'Type2'},
   ...     return_probabilities=True
   ... )
""")
print("=" * 60)

print("\n📚 Next steps:")
print("  - Have data but no model? See: examples/02_finetuning.py")
print("  - Want to upload your model? See: examples/06_upload_model.py")
print("  - Advanced workflows? See: examples/04_scanpy_integration.py")

