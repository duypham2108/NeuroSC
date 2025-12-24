"""
Tools module - scanpy-compatible API for high-level analysis.

This module provides a scanpy-style interface for common single-cell analysis
tasks using foundation models.
"""

from .embedding import embed_cells
from .clustering import cluster_cells, leiden, louvain
from .annotation import annotate_cells, transfer_labels
from .integration import integrate_batches
from .pseudotime import infer_pseudotime

__all__ = [
    "embed_cells",
    "cluster_cells",
    "leiden",
    "louvain",
    "annotate_cells",
    "transfer_labels",
    "integrate_batches",
    "infer_pseudotime",
]

