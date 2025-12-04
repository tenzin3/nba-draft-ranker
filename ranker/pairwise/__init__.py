"""
Utilities and CLI entrypoint for pairwise and learning-to-rank models.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pairwise_rank import (
        PROJECT_ROOT,
        DEFAULT_DATA_PATHS,
        DEFAULT_OUTPUT_PATH,
        DEFAULT_HOLDOUT_SEASONS,
        MIN_FEATURE_COVERAGE,
        MAX_PAIRS_PER_SEASON,
        SYMMETRIC_PAIRS,
        WEIGHT_BY_PICK_GAP,
        RANDOM_STATE,
        load_and_clean,
        pick_feature_columns,
        select_top_features,
        evaluate_spearman_by_season,
        write_rankings_csv,
        ltr_fit_predict,
        fit_preprocessors,
        transform_features,
        build_pairwise_samples,
        fit_pairwise_models_and_weights,
        ensemble_probabilities,
        compute_player_scores_pairwise,
        pairwise_fit_predict,
        select_top_features,
        random_search_ltr,
        grid_search_ltr,
        tune_alpha_on_val,
        main,
    )

__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_DATA_PATHS",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_HOLDOUT_SEASONS",
    "MIN_FEATURE_COVERAGE",
    "MAX_PAIRS_PER_SEASON",
    "SYMMETRIC_PAIRS",
    "WEIGHT_BY_PICK_GAP",
    "RANDOM_STATE",
    "load_and_clean",
    "pick_feature_columns",
    "select_top_features",
    "evaluate_spearman_by_season",
    "write_rankings_csv",
    "ltr_fit_predict",
    "fit_preprocessors",
    "transform_features",
    "build_pairwise_samples",
    "fit_pairwise_models_and_weights",
    "ensemble_probabilities",
    "compute_player_scores_pairwise",
    "pairwise_fit_predict",
    "random_search_ltr",
    "grid_search_ltr",
    "tune_alpha_on_val",
    "main",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(".pairwise_rank", __name__)
    value = getattr(module, name)
    # cache on module for future lookups
    globals()[name] = value
    return value
