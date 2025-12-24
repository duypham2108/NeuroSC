"""
Inference utilities for making predictions with trained models.
"""

from .predict import predict, embed, batch_predict, batch_embed
from .interpret import compute_attention_weights, get_gene_importance

__all__ = [
    "predict",
    "embed",
    "batch_predict",
    "batch_embed",
    "compute_attention_weights",
    "get_gene_importance",
]

