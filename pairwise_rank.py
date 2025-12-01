"""Train a pairwise ranking model and rank recent draft classes.

This script learns a Bradley-Terry style pairwise model on draft combine
measurements from 2000-2022, then uses it to score and rank players in the
2023-2025 classes. Pairwise samples are only formed within the same season and
position group so the model captures relative ordering inside each draft year
without comparing unlike positions.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

DATA_PATH = Path("preprocess/draft_with_position.csv")
OUTPUT_PATH = Path("outputs/pairwise_rankings.csv")
TRAIN_LAST_SEASON = 2022
MIN_FEATURE_COVERAGE = 0.5  # drop features that are mostly empty


def load_and_clean(path: Path) -> pd.DataFrame:
    """Load draft data and normalize columns/values."""
    df = pd.read_csv(path)
    df = df.rename(columns=lambda c: c.strip().lower())

    def parse_ft_in(value: object) -> float:
        """Parse a string like 6' 10'' into inches."""
        if isinstance(value, str) and "'" in value:
            try:
                feet, rest = value.split("'")
                inches = rest.replace('"', "").replace("''", "").replace(" ", "")
                feet_val = float(feet.strip()) if feet.strip() else 0.0
                inch_val = float(inches) if inches else 0.0
                return feet_val * 12 + inch_val
            except Exception:
                return np.nan
        return np.nan

    text_cols = {
        "first_name",
        "last_name",
        "player_name",
        "position",
        "wingspan_ft_in",
        "standing_reach_ft_in",
        "team_city",
        "team_name",
        "team_abbreviation",
        "organization",
        "organization_type",
        "position_clean",
        "position_group",
    }

    # Integer-encode only stable categorical columns (avoid names/teams to reduce leakage).
    categorical_to_encode = {"position_group", "position_clean", "position", "organization_type"}
    for col in categorical_to_encode:
        if col in df.columns:
            df[f"{col}_code"] = (
                df[col]
                .fillna("UNK")
                .astype(str)
                .str.strip()
                .astype("category")
                .cat.codes
                .astype(int)
            )

    # Convert everything that is not a text column into numeric values.
    for col in df.columns:
        if col in text_cols:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["season"] = df["season"].astype(int)
    df["overall_pick"] = df["overall_pick"].astype(int)
    if "position_group" in df.columns:
        df["position_group"] = df["position_group"].astype(str).str.strip()

    # Convert inch-based measurements to centimeters (floats).
    df["height"] = df["height"].astype(float) * 2.54
    df["height_cm"] = df["height"]

    wing_ft_in = df.get("wingspan_ft_in")
    if wing_ft_in is not None:
        wing_in = wing_ft_in.apply(parse_ft_in)
    else:
        wing_in = pd.Series(np.nan, index=df.index)
    df["wingspan"] = df["wingspan"].astype(float) * 2.54
    df["wingspan_cm"] = df["wingspan"]
    df.loc[df["wingspan_cm"].isna(), "wingspan_cm"] = wing_in * 2.54
    df.loc[df["wingspan"].isna(), "wingspan"] = df["wingspan_cm"]

    reach_ft_in = df.get("standing_reach_ft_in")
    if reach_ft_in is not None:
        reach_in = reach_ft_in.apply(parse_ft_in)
    else:
        reach_in = pd.Series(np.nan, index=df.index)
    df["standing_reach"] = df["standing_reach"].astype(float) * 2.54
    df["standing_reach_cm"] = df["standing_reach"]
    df.loc[df["standing_reach_cm"].isna(), "standing_reach_cm"] = reach_in * 2.54
    df.loc[df["standing_reach"].isna(), "standing_reach"] = df["standing_reach_cm"]

    return df


def pick_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], pd.Series]:
    """Select numeric feature columns with reasonable coverage."""
    drop_cols = {
        "overall_pick",
        "round_number",
        "round_pick",
        "season",
        "player_id",
        "team_id",
    }
    text_cols = {
        "first_name",
        "last_name",
        "player_name",
        "position",
        "wingspan_ft_in",
        "standing_reach_ft_in",
        "team_city",
        "team_name",
        "team_abbreviation",
        "organization",
        "organization_type",
        "position_clean",
        "position_group",
    }

    coverage = df.drop(columns=list(drop_cols | text_cols), errors="ignore").notna().mean()
    eligible = coverage[coverage >= MIN_FEATURE_COVERAGE].index.tolist()

    # Manually remove redundant or overly correlated columns to reduce noise.
    # Keep only one version of height / wingspan / standing reach.
    manual_exclude = {
        "height_cm",
        "wingspan_cm",
        "standing_reach_cm",
        # These encoded categorical columns are highly correlated with pos_num
        # and can confuse linear models if treated as ordered.
        "position_group_code",
        "position_clean_code",
        "position_code",
    }
    feature_cols = [c for c in eligible if c not in manual_exclude]

    dropped = [c for c in coverage.index if c not in feature_cols]
    return feature_cols, dropped, coverage


def build_pairwise_samples(
    df: pd.DataFrame, features: np.ndarray, within_position_only: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Create pairwise differences and labels within each season.

    If ``within_position_only`` is True, restrict pairs to players sharing the
    same ``position_group``. Otherwise, form all pairs inside a season so the
    model learns a true global ordering for each draft class.
    """
    x_rows: List[np.ndarray] = []
    y_rows: List[int] = []

    for season, idx in df.groupby("season").groups.items():
        season_idx = list(idx)
        if len(season_idx) < 2:
            continue

        season_features = features[season_idx]
        picks = df.loc[season_idx, "overall_pick"].to_numpy()
        pos_groups = df.loc[season_idx, "position_group"].to_numpy()

        for i, j in combinations(range(len(season_idx)), 2):
            if within_position_only:
                if (
                    pd.isna(pos_groups[i])
                    or pd.isna(pos_groups[j])
                    or pos_groups[i] != pos_groups[j]
                ):
                    continue
            if picks[i] == picks[j]:
                continue
            x_rows.append(season_features[i] - season_features[j])
            y_rows.append(1 if picks[i] < picks[j] else 0)

    if not x_rows:
        return np.empty((0, features.shape[1])), np.empty((0,), dtype=int)
    return np.vstack(x_rows), np.array(y_rows)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_pairwise_models(x_train: np.ndarray, y_train: np.ndarray) -> List[Tuple[str, object]]:
    """Fit several linear pairwise models for ensembling."""
    models: List[Tuple[str, object]] = [
        (
            "log_reg",
            LogisticRegression(
                max_iter=1000,
                fit_intercept=False,
                class_weight="balanced",
                n_jobs=-1,
                solver="lbfgs",
            ),
        ),
        (
            "linear_svc",
            LinearSVC(
                fit_intercept=False,
                class_weight="balanced",
                max_iter=5000,
            ),
        ),
        (
            "sgd_log",
            SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                max_iter=2000,
                tol=1e-4,
                fit_intercept=False,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        (
            "gboost",
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=2,
                subsample=0.8,
                random_state=42,
            ),
        ),
    ]

    for name, model in models:
        model.fit(x_train, y_train)
    return models


def ensemble_probabilities(models: List[Tuple[str, object]], x: np.ndarray) -> np.ndarray:
    """Average probabilities/decision scores across models."""
    probs: List[np.ndarray] = []
    for _, model in models:
        if hasattr(model, "predict_proba"):
            probs.append(model.predict_proba(x)[:, 1])  # type: ignore[attr-defined]
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(x)  # type: ignore[attr-defined]
            probs.append(_sigmoid(decision))
        else:
            probs.append(model.predict(x))  # type: ignore[attr-defined]
    return np.mean(probs, axis=0)


def compute_player_scores(
    df: pd.DataFrame,
    features: np.ndarray,
    models: List[Tuple[str, object]],
    within_position_only: bool = False,
) -> np.ndarray:
    """Estimate player skill as average win probability vs peers.

    If ``within_position_only`` is True, only compare players inside the same
    ``position_group``; otherwise, compare all players within a season.
    """
    scores = np.zeros(len(df))
    games = np.zeros(len(df))

    for season, idx in df.groupby("season").groups.items():
        season_idx = list(idx)
        if len(season_idx) < 2:
            continue

        positions = df.loc[season_idx, "position_group"].to_numpy()
        season_feats = features[season_idx]

        if within_position_only:
            # Only compare players within the same position_group.
            for pos in np.unique(positions):
                pos_mask = [k for k in range(len(season_idx)) if positions[k] == pos]
                if len(pos_mask) < 2:
                    continue
                for i, j in combinations(pos_mask, 2):
                    diff = season_feats[i] - season_feats[j]
                    prob = ensemble_probabilities(models, diff.reshape(1, -1))[0]
                    gi, gj = season_idx[i], season_idx[j]
                    scores[gi] += prob
                    scores[gj] += 1 - prob
                    games[gi] += 1
                    games[gj] += 1
        else:
            # Compare all players within the season (global ordering per season).
            for i, j in combinations(range(len(season_idx)), 2):
                diff = season_feats[i] - season_feats[j]
                prob = ensemble_probabilities(models, diff.reshape(1, -1))[0]
                gi, gj = season_idx[i], season_idx[j]
                scores[gi] += prob
                scores[gj] += 1 - prob
                games[gi] += 1
                games[gj] += 1

    games = np.maximum(games, 1)  # avoid divide by zero
    return scores / games


def count_pairs(df: pd.DataFrame) -> Tuple[int, List[Tuple[int, int]]]:
    """Count pairwise comparisons per season under the position constraint."""
    total = 0
    per_season: List[Tuple[int, int]] = []
    for season, g in df.groupby("season"):
        c = 0
        for _, gg in g.groupby("position_group"):
            n = len(gg)
            c += n * (n - 1) // 2
        per_season.append((int(season), c))
        total += c
    return total, per_season


def fit_preprocessors(train_df: pd.DataFrame, feature_cols: Iterable[str]) -> Dict[str, object]:
    """Fit mean imputer and scaler on training data."""
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    imputer.fit(train_df[feature_cols])
    scaler.fit(imputer.transform(train_df[feature_cols]))
    return {"imputer": imputer, "scaler": scaler}


def transform_features(
    df: pd.DataFrame, feature_cols: Iterable[str], imputer: SimpleImputer, scaler: StandardScaler
) -> np.ndarray:
    """Apply preprocessing to the given dataframe."""
    imputed = imputer.transform(df[feature_cols])
    scaled = scaler.transform(imputed)
    return scaled


def evaluate_spearman_by_season(
    df: pd.DataFrame,
    scores: np.ndarray,
    group_by_position: bool = False,
) -> List[Tuple[int, float]]:
    """Compute Spearman correlation between predicted score and draft order.

    If ``group_by_position`` is False, compute one Spearman per season using
    all players. If True, compute a weighted average of Spearman correlations
    over ``position_group`` within each season.
    """
    results: List[Tuple[int, float]] = []
    for season, idx in df.groupby("season").groups.items():
        season_idx = list(idx)
        season_scores = scores[season_idx]
        season_df = df.loc[season_idx]

        if not group_by_position:
            picks = season_df["overall_pick"].to_numpy()
            corr, _ = spearmanr(-picks, season_scores)  # lower pick = higher rank
            results.append((int(season), float(corr)))
        else:
            # Weighted average over position groups by group size.
            total_n = 0
            weighted_sum = 0.0
            for _, g in season_df.groupby("position_group"):
                if len(g) < 2:
                    continue
                g_idx = g.index.to_numpy()
                g_scores = scores[g_idx]
                g_picks = g["overall_pick"].to_numpy()
                corr, _ = spearmanr(-g_picks, g_scores)
                if np.isnan(corr):
                    continue
                n = len(g)
                total_n += n
                weighted_sum += corr * n
            if total_n == 0:
                corr_val = np.nan
            else:
                corr_val = weighted_sum / total_n
            results.append((int(season), float(corr_val)))
    return results


def evaluate_pairwise_accuracy(
    df: pd.DataFrame,
    features: np.ndarray,
    models: List[Tuple[str, object]],
    within_position_only: bool = False,
) -> float:
    """Compute pairwise prediction accuracy within each season.

    This directly measures how often the model's pairwise preferences agree
    with draft order, aligning evaluation with the pairwise training objective.
    """
    y_true: List[int] = []
    y_pred: List[int] = []

    for season, idx in df.groupby("season").groups.items():
        season_idx = list(idx)
        if len(season_idx) < 2:
            continue
        season_feats = features[season_idx]
        picks = df.loc[season_idx, "overall_pick"].to_numpy()
        pos_groups = df.loc[season_idx, "position_group"].to_numpy()

        for i, j in combinations(range(len(season_idx)), 2):
            if within_position_only:
                if (
                    pd.isna(pos_groups[i])
                    or pd.isna(pos_groups[j])
                    or pos_groups[i] != pos_groups[j]
                ):
                    continue
            if picks[i] == picks[j]:
                continue
            diff = season_feats[i] - season_feats[j]
            prob = ensemble_probabilities(models, diff.reshape(1, -1))[0]
            y_true.append(1 if picks[i] < picks[j] else 0)
            y_pred.append(1 if prob >= 0.5 else 0)

    if not y_true:
        return float("nan")
    return float(accuracy_score(y_true, y_pred))


def main() -> None:
    df = load_and_clean(DATA_PATH).reset_index(drop=True)
    feature_cols, dropped_cols, coverage = pick_feature_columns(df)

    train_df = df[df["season"] <= TRAIN_LAST_SEASON].reset_index(drop=True)
    test_df = df[df["season"] > TRAIN_LAST_SEASON].reset_index(drop=True)

    # Print data diagnostics.
    print(f"Feature coverage (kept >= {MIN_FEATURE_COVERAGE:.2f} non-null):")
    print(
        coverage.sort_values(ascending=True).to_string(float_format=lambda x: f"{x:.3f}")
    )
    print(f"Kept {len(feature_cols)} features: {feature_cols}")
    print(f"Dropped {len(dropped_cols)} sparse features: {dropped_cols}")
    train_pairs_total, train_pairs = count_pairs(train_df)
    test_pairs_total, test_pairs = count_pairs(test_df)
    print(
        f"Train players: {len(train_df)}, seasons: {train_df['season'].nunique()}, pairs: {train_pairs_total}"
    )
    print(
        f"Test players: {len(test_df)}, seasons: {test_df['season'].nunique()}, pairs: {test_pairs_total}"
    )
    print(f"Train pair counts by season: {train_pairs}")
    print(f"Test pair counts by season: {test_pairs}")

    preprocessors = fit_preprocessors(train_df, feature_cols)
    imputer: SimpleImputer = preprocessors["imputer"]  # type: ignore[assignment]
    scaler: StandardScaler = preprocessors["scaler"]  # type: ignore[assignment]

    train_features = transform_features(train_df, feature_cols, imputer, scaler)
    test_features = transform_features(test_df, feature_cols, imputer, scaler)

    # Train pairwise model on full-season pairs (global ordering within season).
    x_train, y_train = build_pairwise_samples(train_df, train_features, within_position_only=False)
    x_test, y_test = build_pairwise_samples(test_df, test_features, within_position_only=False)

    models = fit_pairwise_models(x_train, y_train)

    # Pairwise accuracy directly on held-out pairs.
    ensemble_probs = ensemble_probabilities(models, x_test)
    test_pred = (ensemble_probs >= 0.5).astype(int)
    pairwise_acc = accuracy_score(y_test, test_pred)

    # Player-level scores = average win probability vs peers in same season.
    test_scores_global = compute_player_scores(
        test_df, test_features, models, within_position_only=False
    )
    test_scores_by_pos = compute_player_scores(
        test_df, test_features, models, within_position_only=True
    )

    correlations_global = evaluate_spearman_by_season(
        test_df, test_scores_global, group_by_position=False
    )
    correlations_by_pos = evaluate_spearman_by_season(
        test_df, test_scores_by_pos, group_by_position=True
    )

    rankings: List[pd.DataFrame] = []
    for season, idx in test_df.groupby("season").groups.items():
        subset = test_df.loc[idx].copy()
        subset["pred_score"] = test_scores_global[list(idx)]
        subset["pred_rank"] = subset["pred_score"].rank(
            ascending=False, method="first"
        ).astype(int)
        subset = subset.sort_values("pred_rank")[
            ["season", "player_name", "overall_pick", "pred_score", "pred_rank"]
        ]
        rankings.append(subset)

    Path(OUTPUT_PATH.parent).mkdir(parents=True, exist_ok=True)
    pd.concat(rankings).to_csv(OUTPUT_PATH, index=False)

    print(f"Pairwise accuracy on 2023-2025 (full-season pairs): {pairwise_acc:.3f}")

    # Also report pairwise accuracy computed directly over all season pairs.
    full_pairwise_acc = evaluate_pairwise_accuracy(
        test_df, test_features, models, within_position_only=False
    )
    pos_pairwise_acc = evaluate_pairwise_accuracy(
        test_df, test_features, models, within_position_only=True
    )
    print(f"Pairwise accuracy (recomputed, full-season): {full_pairwise_acc:.3f}")
    print(f"Pairwise accuracy (within position groups): {pos_pairwise_acc:.3f}")

    print("Spearman correlations by season (global score vs. draft order):")
    for season, corr in sorted(correlations_global):
        print(f"  {season}: {corr:.3f}")

    print("Spearman correlations by season (position-group weighted):")
    for season, corr in sorted(correlations_by_pos):
        print(f"  {season}: {corr:.3f}")
    print(f"Wrote rankings to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
