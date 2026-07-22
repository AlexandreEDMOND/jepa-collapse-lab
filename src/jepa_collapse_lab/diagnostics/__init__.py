"""Collapse diagnostics: variance, covariance, spectrum, effective rank, UMAP."""

from .collect import (
    collect_embeddings,
    collect_paired_embeddings,
    load_model_from_checkpoint,
    select_embeddings,
)
from .metrics import (
    correlation_matrix,
    covariance_matrix,
    cross_correlation_matrix,
    cross_correlation_summary,
    effective_rank,
    mean_per_dim_std,
    per_dim_std,
    singular_spectrum,
    summarize_embeddings,
)
from .plots import (
    plot_covariance_heatmap,
    plot_cross_correlation_heatmap,
    plot_per_dim_std,
    plot_projection,
    plot_singular_spectrum,
    plot_training_curves,
    run_all_diagnostics,
)

__all__ = [
    "collect_embeddings",
    "collect_paired_embeddings",
    "correlation_matrix",
    "covariance_matrix",
    "cross_correlation_matrix",
    "cross_correlation_summary",
    "effective_rank",
    "load_model_from_checkpoint",
    "mean_per_dim_std",
    "per_dim_std",
    "plot_covariance_heatmap",
    "plot_cross_correlation_heatmap",
    "plot_per_dim_std",
    "plot_projection",
    "plot_singular_spectrum",
    "plot_training_curves",
    "run_all_diagnostics",
    "select_embeddings",
    "singular_spectrum",
    "summarize_embeddings",
]
