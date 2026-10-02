"""Unified, device-agnostic TOP pre-training label computation.

See design/top_label_unified_gpu_design.md for the derivation.
"""
from top_label.unified import (
    TopLabelConfig,
    STATE_UNKNOWN,
    STATE_FREE,
    STATE_OCCUPIED,
    compute_top_labels,
    compute_frame_pair_labels,
    overlap_intervals,
    label_from_depth,
)

__all__ = [
    "TopLabelConfig",
    "STATE_UNKNOWN",
    "STATE_FREE",
    "STATE_OCCUPIED",
    "compute_top_labels",
    "compute_frame_pair_labels",
    "overlap_intervals",
    "label_from_depth",
]
