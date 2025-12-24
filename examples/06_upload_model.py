"""
Upload Model to HuggingFace - Share Your Finetuned Model
=========================================================

This example shows how to upload your finetuned classification model
to HuggingFace Hub for free hosting and easy sharing.
"""

import neurosc as nsc
import scanpy as sc

print("=" * 60)
print("Upload Your Model to HuggingFace Hub")
print("=" * 60)

# Scenario: You've finetuned a model and want to share it

# ============================================================================
# Step 1: Finetune your model (or load existing one)
# ============================================================================

print("\n[Step 1] Prepare your finetuned model")
print("-" * 60)

# Option A: You already have a trained model
# model = nsc.load_model("./my_finetuned_model/")

# Option B: Quick finetune for this example
print("Finetuning a model (for demonstration)...")

# Load sample data
adata = sc.datasets.pbmc3k()
adata = nsc.prepare_anndata(adata, min_genes=200, highly_variable_genes=2000)

# Add pseudo cell types
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.leiden(adata)
adata.obs["cell_type"] = adata.obs["leiden"]

try:
    # Quick finetune
    model = nsc.tools.quick_finetune(
        adata,
        label_key="cell_type",
        num_epochs=2,  # Short for demo
        strategy="lora",
        output_dir="./temp_model",
    )
    print("✓ Model finetuned")
except Exception as e:
    print(f"⚠ Finetuning requires base model: {e}")
    print("This will work once base models are available")
    model = None


# ============================================================================
# Step 2: Set up HuggingFace authentication
# ============================================================================

print("\n[Step 2] Authenticate with HuggingFace")
print("-" * 60)
print(
    """
Get your HuggingFace token:
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with 'write' access
3. Copy the token (starts with 'hf_...')
"""
)

# Set up authentication
# Option A: Provide token directly
# nsc.setup_huggingface(token="hf_your_token_here")

# Option B: Use environment variable (recommended)
# export HF_TOKEN=hf_your_token_here
# nsc.setup_huggingface()

print("Run: nsc.setup_huggingface(token='hf_...')")


# ============================================================================
# Step 3: Upload model to HuggingFace Hub
# ============================================================================

print("\n[Step 3] Upload to HuggingFace Hub (FREE!)")
print("-" * 60)

if model is not None:
    try:
        nsc.upload_to_hub(
            model,
            repo_id="your-username/scgpt-brain-celltype-classifier",
            commit_message="Finetuned scGPT for brain cell type classification",
            private=False,  # Make it public to share with community!
        )

        print("\n✓ Upload complete!")
        print("Your model is now hosted on HuggingFace for FREE")
        print("URL: https://huggingface.co/your-username/scgpt-brain-celltype-classifier")

    except Exception as e:
        print(f"Upload will work after authentication: {e}")
else:
    print("Upload example:")
    print(
        """
    nsc.upload_to_hub(
        model,
        repo_id="your-username/model-name",
        commit_message="Description of your model"
    )
    """
    )


# ============================================================================
# Step 4: Create model card (optional but recommended)
# ============================================================================

print("\n[Step 4] Create a Model Card (recommended)")
print("-" * 60)

model_card_template = """
# scGPT Brain Cell Type Classifier

This model is a finetuned version of scGPT for classifying brain cell types.

## Model Description

- **Base Model**: scGPT
- **Task**: Cell type classification
- **Species**: Mouse
- **Tissue**: Brain (cortex)
- **Cell Types**: 7 types (Excitatory neurons, Inhibitory neurons, Astrocytes, ...)

## Training Data

- Dataset: Allen Brain Atlas
- Cells: 50,000
- Genes: 3,000 highly variable genes

## Usage

```python
import neurosc as nsc

# Annotate your data
adata = nsc.annotate_celltype(
    adata_path="your_brain_data.h5ad",
    model_path="your-username/scgpt-brain-celltype-classifier"
)
```

## Performance

- Accuracy: 95.2%
- F1 Score: 0.94

## Citation

If you use this model, please cite...
"""

print("Save this as README.md in your HuggingFace repo:")
print(model_card_template)


# ============================================================================
# Step 5: Share label mapping
# ============================================================================

print("\n[Step 5] Share your label mapping")
print("-" * 60)

# Create a label mapping file
label_mapping = {
    0: "Excitatory Neuron",
    1: "Inhibitory Neuron",
    2: "Astrocyte",
    3: "Oligodendrocyte",
    4: "Microglia",
    5: "Endothelial",
    6: "OPC",
}

import json

mapping_file = "label_mapping.json"
with open(mapping_file, "w") as f:
    json.dump(label_mapping, f, indent=2)

print(f"✓ Created {mapping_file}")
print("Upload this to your HuggingFace repo so users know the mapping!")


# ============================================================================
# Step 6: Test your uploaded model
# ============================================================================

print("\n[Step 6] Test your uploaded model")
print("-" * 60)

test_code = """
# Anyone can now use your model!

import neurosc as nsc

# Load and annotate
adata = nsc.annotate_celltype(
    adata_path="data.h5ad",
    model_path="your-username/scgpt-brain-celltype-classifier",
    label_mapping={
        0: 'Excitatory Neuron',
        1: 'Inhibitory Neuron',
        # ... etc
    }
)

print(adata.obs['cell_type'].value_counts())
"""

print("Users can now use your model:")
print(test_code)


# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 60)
print("Summary - Sharing Your Model")
print("=" * 60)
print(
    """
1. Finetune your model:
   >>> model = nsc.tools.quick_finetune(adata, label_key="cell_type")

2. Authenticate:
   >>> nsc.setup_huggingface(token="hf_...")

3. Upload (FREE):
   >>> nsc.upload_to_hub(model, "username/model-name")

4. Create README.md with:
   - Model description
   - Usage instructions
   - Label mapping
   - Performance metrics

5. Share with community!
   URL: https://huggingface.co/username/model-name

Benefits:
✓ Free hosting
✓ Version control
✓ Easy sharing
✓ Community contributions
✓ Automatic model cards
"""
)
print("=" * 60)
