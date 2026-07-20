"""Collapse diagnostics: variance, covariance, spectrum, effective rank, UMAP."""

from .collect import collect_embeddings, load_model_from_checkpoint, select_embeddings
from .metrics import (
    correlation_matrix,
    covariance_matrix,
    effective_rank,
    mean_per_dim_std,
    per_dim_std,
    singular_spectrum,
    summarize_embeddings,
)
from .plots import (
    plot_covariance_heatmap,
    plot_per_dim_std,
    plot_projection,
    plot_singular_spectrum,
    plot_training_curves,
    run_all_diagnostics,
)

__all__ = [
    "collect_embeddings",
    "correlation_matrix",
    "covariance_matrix",
    "effective_rank",
    "load_model_from_checkpoint",
    "mean_per_dim_std",
    "per_dim_std",
    "plot_covariance_heatmap",
    "plot_per_dim_std",
    "plot_projection",
    "plot_singular_spectrum",
    "plot_training_curves",
    "run_all_diagnostics",
    "select_embeddings",
    "singular_spectrum",
    "summarize_embeddings",
]
