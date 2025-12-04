# coding: utf-8
"""
pairwise_rank.py (Improved)

- LTR (LightGBM LambdaRank) with:
  * robust optional per-season z-score normalization (no warnings)
  * custom early-stopping metric: mean Spearman across query groups (aligns with your evaluation)
  * time-based validation (last K seasons in training) to select best_iteration
  * refit on ALL training seasons using best_iteration (no data wasted)
  * small ensemble (3 configs) averaged predictions
  * supports --mode ltr / pairwise / hybrid (blends pairwise+ltr with alpha tuned on held-out seasons)

- Pairwise (baseline) with:
  * stronger ensemble incl. HGB, ExtraTrees, Logistic, SGD, optional LGBMClassifier

Usage:
  python -m ranker.pairwise.pairwise_rank --data outputs/nba_draft_final.csv --train_last_season 2015 --mode ltr --ltr_season_zscore
  python -m ranker.pairwise.pairwise_rank --data outputs/nba_draft_final.csv --train_last_season 2015 --mode pairwise --max_pairs_per_season 30000
  python -m ranker.pairwise.pairwise_rank --data outputs/nba_draft_final.csv --train_last_season 2015 --mode hybrid --ltr_season_zscore --alpha_val_last_k 2
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_regression

import lightgbm as lgb

import re

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None  # type: ignore


# -----------------------------
# Defaults
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Only use the notebook's data source: outputs/nba_college_selected_features.csv at repo root.
DEFAULT_DATA_PATHS = [
    PROJECT_ROOT / "outputs" / "college_stats.csv",
    Path("outputs/college_stats.csv"),
]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "pairwise_rankings.csv"

DEFAULT_HOLDOUT_SEASONS = 4
MIN_FEATURE_COVERAGE = 0.50

MAX_PAIRS_PER_SEASON = 30000
SYMMETRIC_PAIRS = True
WEIGHT_BY_PICK_GAP = True
RANDOM_STATE = 42


# -----------------------------
# Helpers
# -----------------------------

def _resolve_default_data_path() -> Path:
    for p in DEFAULT_DATA_PATHS:
        if p.exists():
            return p
    return DEFAULT_DATA_PATHS[0]


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df
    dup_names = df.columns[df.columns.duplicated()].unique()
    out = df.copy()
    for name in dup_names:
        idxs = np.where(out.columns == name)[0]
        block = out.iloc[:, idxs]
        out[name] = block.bfill(axis=1).iloc[:, 0]
        out = out.drop(out.columns[idxs[1:]], axis=1)
    return out


def _derive_position_group(pos: object) -> str:
    if pos is None or (isinstance(pos, float) and np.isnan(pos)):
        return "UNK"
    s = str(pos).upper().replace(" ", "")
    has_g = "G" in s
    has_f = "F" in s
    has_c = "C" in s
    if has_g and has_f:
        return "GF"
    if has_f and has_c:
        return "FC"
    if has_g:
        return "G"
    if has_f:
        return "F"
    if has_c:
        return "C"
    return "UNK"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _predict_proba(model: object, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]  # type: ignore[attr-defined]
    if hasattr(model, "decision_function"):
        return _sigmoid(model.decision_function(X))  # type: ignore[attr-defined]
    return model.predict(X).astype(float)  # type: ignore[attr-defined]


def _infer_name_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["player_name", "name", "player", "full_name", "first_name"]:
        if c in df.columns:
            return c
    return None


def _sort_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["season", "overall_pick"]).reset_index(drop=True)


def _default_ltr_search_space() -> List[Dict]:
    # Modest grid sized for this dataset; early stopping will trim rounds.
    leaves = [31, 47, 63]
    lrs = [0.04, 0.05]
    mins = [20, 30]
    rounds = [2000, 3000]
    space: List[Dict] = []
    for n in leaves:
        for lr in lrs:
            for m in mins:
                for r in rounds:
                    space.append(
                        {
                            "num_leaves": n,
                            "learning_rate": lr,
                            "min_child_samples": m,
                            "num_boost_round": r,
                            "colsample_bytree": 0.85,
                            "subsample": 0.9,
                        }
                    )
    return space


def _add_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create per-minute rates from totals for richer signal on this dataset."""
    rate_cols = ["totals_fg", "totals_ft", "totals_trb", "totals_blk", "totals_stl", "totals_tov", "totals_pf"]
    out = df.copy()
    mp = out.get("mp")
    if mp is None:
        return out
    mp = pd.to_numeric(mp, errors="coerce").replace(0, np.nan)
    for col in rate_cols:
        if col in out.columns:
            out[f"{col}_per_min"] = out[col] / mp
    return out


def _score_features(train_df: pd.DataFrame, feature_cols: List[str]) -> List[Tuple[str, float, float, float]]:
    """
    Score features using Spearman correlation (absolute) and mutual information
    against -overall_pick (higher is better pick).
    Returns list of (col, spearman, mi, combined_score).
    """
    y = -train_df["overall_pick"].to_numpy()
    scores: List[Tuple[str, float, float, float]] = []
    for col in feature_cols:
        s = pd.to_numeric(train_df[col], errors="coerce")
        mask = s.notna()
        if mask.sum() < 10:
            continue
        sp = float(spearmanr(y[mask], s[mask]).correlation)
        try:
            mi = float(mutual_info_regression(s[mask].to_frame(), y[mask], random_state=RANDOM_STATE)[0])
        except Exception:
            mi = 0.0
        scores.append((col, sp, mi, 0.0))

    if not scores:
        return []

    max_mi = max(abs(mi) for _, _, mi, _ in scores) or 1.0
    out: List[Tuple[str, float, float, float]] = []
    for col, sp, mi, _ in scores:
        combined = abs(sp) + 0.5 * (mi / max_mi)
        out.append((col, sp, mi, combined))
    out.sort(key=lambda x: x[3], reverse=True)
    return out


def select_top_features(train_df: pd.DataFrame, feature_cols: List[str], top_k: int) -> Tuple[List[str], List[Tuple[str, float, float, float]]]:
    scores = _score_features(train_df, feature_cols)
    if not scores:
        return feature_cols, []
    selected = [c for c, _, _, _ in scores[:max(1, top_k)]]
    return selected, scores


# -----------------------------
# Load / clean
# -----------------------------

def load_and_clean(path: Path, *, dedupe_by_pick: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = df.rename(columns=lambda c: str(c).strip().lower())
    df = _coalesce_duplicate_columns(df)

    if "season" not in df.columns and "draft_year" in df.columns:
        df["season"] = df["draft_year"]

    if "overall_pick" not in df.columns:
        raise ValueError("Missing required column: overall_pick")
    if "season" not in df.columns:
        raise ValueError("Missing required column: season (or draft_year)")

    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["overall_pick"] = pd.to_numeric(df["overall_pick"], errors="coerce")
    df = df.dropna(subset=["season", "overall_pick"]).copy()
    df["season"] = df["season"].astype(int)
    df["overall_pick"] = df["overall_pick"].astype(int)
    # ---- dedupe: (season, overall_pick) should be unique ----
    if dedupe_by_pick:
        before = len(df)
        dup = int(df.duplicated(subset=["season", "overall_pick"]).sum())
        if dup > 0:
            # keep the row with the largest MP (more complete season stats)
            if "mp" in df.columns:
                df = df.sort_values("mp", ascending=False)
            df = df.drop_duplicates(subset=["season", "overall_pick"], keep="first").copy()
            after = len(df)
            print(f"[Clean] Deduped (season, overall_pick): removed {before-after} rows (had {dup} duplicates).")
        assert int(df.duplicated(subset=["season", "overall_pick"]).sum()) == 0

    # position -> position_group one-hot
    if "position_group" not in df.columns:
        if "position" in df.columns:
            df["position_group"] = df["position"].apply(_derive_position_group)
        else:
            df["position_group"] = "UNK"
    else:
        df["position_group"] = df["position_group"].astype(str).str.strip()

    cat_cols = [c for c in ["position_group", "organization_type"] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols, dummy_na=True)

    # try numeric coercion for non-text object cols
    text_cols = {
        "player_name", "first_name", "last_name", "name", "player", "full_name",
        "team_name", "team_city", "team_abbreviation", "college", "position",
    }
    for col in df.columns:
        if col in text_cols:
            continue
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _add_rate_features(df)

    return df.reset_index(drop=True)


def pick_feature_columns(df: pd.DataFrame, *, feature_set: str = "per_min") -> Tuple[List[str], List[str]]:


    feature_set = str(feature_set).strip().lower()
    if feature_set not in {"per_min", "totals", "both"}:
        raise ValueError(f"Unknown feature_set={feature_set!r}. Choose from: per_min, totals, both.")

    curated_base = [
        "shooting_fg%",
        "mp",
        "age",
    ]
    curated_per_min = [
        "totals_fg_per_min",
        "totals_ft_per_min",
        "totals_trb_per_min",
        "totals_blk_per_min",
        "totals_stl_per_min",
        "totals_tov_per_min",
        "totals_pf_per_min",
    ]
    curated_totals = [
        "totals_fg",
        "totals_ft",
        "totals_trb",
        "totals_blk",
        "totals_stl",
        "totals_tov",
        "totals_pf",
    ]

    # Default (per_min): avoid collinearity between totals and per-minute versions.
    curated_order: List[str] = list(curated_base)
    if feature_set == "per_min":
        curated_order += curated_per_min
    elif feature_set == "totals":
        curated_order += curated_totals
    else:  # both
        curated_order += curated_per_min + curated_totals



    drop_cols = {
        "overall_pick",
        "round_number",
        "round_pick",
        "season",
        "draft_year",
        "person_id",
        "player_profile_flag",
        "player_name",
    }
    object_cols = set(df.select_dtypes(include=["object"]).columns.tolist())
    candidate = df.drop(columns=list(drop_cols | object_cols), errors="ignore")
    coverage = candidate.notna().mean()
    eligible = [c for c in curated_order if c in candidate.columns and coverage.get(c, 0.0) >= MIN_FEATURE_COVERAGE]

    for col in candidate.columns:
        if col not in eligible and coverage.get(col, 0.0) >= MIN_FEATURE_COVERAGE:
            eligible.append(col)

    dropped = sorted(set(candidate.columns) - set(eligible))
    return eligible, dropped


# -----------------------------
# Evaluation / output
# -----------------------------

def evaluate_spearman_by_season(df: pd.DataFrame, scores: np.ndarray) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    for season, idx in df.groupby("season").groups.items():
        season_idx = list(idx)
        picks = df.loc[season_idx, "overall_pick"].to_numpy()
        corr = spearmanr(-picks, scores[season_idx]).correlation
        out.append((int(season), float(corr)))
    return out


def write_rankings_csv(out_path: Path, df: pd.DataFrame, scores: np.ndarray) -> None:
    name_col = _infer_name_col(df)
    chunks: List[pd.DataFrame] = []
    for season, idx in df.groupby("season").groups.items():
        season_idx = list(idx)
        sub = df.loc[season_idx].copy()
        sub["pred_score"] = scores[season_idx]
        sub["pred_rank"] = sub["pred_score"].rank(ascending=False, method="first").astype(int)
        keep_cols = ["season"] + ([name_col] if name_col else []) + ["overall_pick", "pred_score", "pred_rank"]
        chunks.append(sub.sort_values("pred_rank")[keep_cols])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(chunks, ignore_index=True).to_csv(out_path, index=False)
    print(f"Wrote rankings to {out_path}")


# ============================================================
# LTR (LightGBM LambdaRank) – with Spearman early stopping
# ============================================================

def _safe_group_zscore(Xg: np.ndarray) -> np.ndarray:
    """
    Z-score per feature within a season group, ignoring NaNs, NO warnings.
    If a feature is all-NaN in a group -> mean=0, std=1 (keeps NaNs).
    """
    X = Xg.astype(float, copy=True)
    mask = ~np.isnan(X)
    cnt = mask.sum(axis=0).astype(float)  # [d]
    sum_ = np.where(mask, X, 0.0).sum(axis=0)
    mean = np.divide(sum_, cnt, out=np.zeros_like(sum_), where=cnt > 0)

    centered = X - mean
    var_sum = np.where(mask, centered * centered, 0.0).sum(axis=0)
    var = np.divide(var_sum, cnt, out=np.ones_like(var_sum), where=cnt > 0)
    std = np.sqrt(var)
    std[std < 1e-6] = 1.0
    return (X - mean) / std


def _make_ltr_arrays(
    df: pd.DataFrame,
    feature_cols: List[str],
    *,
    season_zscore: bool,
) -> Tuple[pd.DataFrame, np.ndarray, List[int], pd.DataFrame, List[int]]:
    """
    Returns (X_df, y, group_sizes, df_sorted, season_order)
    - df_sorted: season,pick sorted
    - y: per-season reverse rank in [0..n-1]
    """
    df_sorted = _sort_df(df)
    seasons = df_sorted["season"].to_numpy()
    X = df_sorted[feature_cols].to_numpy(dtype=float)

    y = np.zeros(len(df_sorted), dtype=np.int32)
    group_sizes: List[int] = []
    season_order: List[int] = []

    start = 0
    while start < len(df_sorted):
        s = seasons[start]
        end = start
        while end < len(df_sorted) and seasons[end] == s:
            end += 1
        n = end - start
        group_sizes.append(n)
        season_order.append(int(s))

        if season_zscore:
            X[start:end] = _safe_group_zscore(X[start:end])

        # best pick -> highest relevance
        y[start:end] = np.arange(n - 1, -1, -1, dtype=np.int32)
        start = end

    X_df = pd.DataFrame(X, columns=feature_cols)
    return X_df, y, group_sizes, df_sorted, season_order


def _split_groups_last_k(season_order: List[int], group_sizes: List[int], k: int) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """
    Split row indices into train/val by taking last k seasons as val.
    Returns (train_idx, val_idx, group_tr, group_val)
    """
    k = max(1, min(k, len(season_order) - 1)) if len(season_order) > 1 else 1
    val_seasons = set(season_order[-k:])

    train_rows: List[np.ndarray] = []
    val_rows: List[np.ndarray] = []
    group_tr: List[int] = []
    group_val: List[int] = []

    start = 0
    for s, g in zip(season_order, group_sizes):
        idxs = np.arange(start, start + g)
        if s in val_seasons:
            val_rows.append(idxs)
            group_val.append(g)
        else:
            train_rows.append(idxs)
            group_tr.append(g)
        start += g

    train_idx = np.concatenate(train_rows) if train_rows else np.array([], dtype=int)
    val_idx = np.concatenate(val_rows) if val_rows else np.array([], dtype=int)
    return train_idx, val_idx, group_tr, group_val


def _feval_mean_spearman(preds: np.ndarray, dataset: lgb.Dataset) -> Tuple[str, float, bool]:
    """
    Custom metric aligned with user's evaluation:
    mean Spearman correlation between preds and labels within each query group.
    (labels are per-season reverse rank)
    """
    y = dataset.get_label()
    group = dataset.get_group()
    idx = 0
    cors: List[float] = []
    for g in group:
        g = int(g)
        p = preds[idx:idx + g]
        t = y[idx:idx + g]
        idx += g
        if g < 2:
            continue
        c = spearmanr(p, t).correlation
        if c is not None and not np.isnan(c):
            cors.append(float(c))
    return "mean_spearman", float(np.mean(cors)) if cors else 0.0, True


def _guess_monotone_for_feature(name: str) -> int:
    """
    Heuristic monotonic constraints for NBA draft features.
    +1 means larger feature => larger relevance (earlier pick).
    -1 means larger feature => smaller relevance.
     0 means unconstrained.
    """
    n = name.lower()
    # categorical one-hots: don't constrain
    if n.startswith("position_group_") or n.startswith("organization_type_"):
        return 0
    # negative signals
    if "age" in n:
        return -1
    if "tov" in n or "turnover" in n:
        return -1
    if re.search(r"pf", n) or "totals_pf" in n or "foul" in n:
        return -1
    # positive signals (availability & skill)
    if n in {"mp"} or "minutes" in n:
        return +1
    if "fg%" in n or "shooting_fg" in n:
        return +1
    if "fg" in n or "ft" in n or "trb" in n or "reb" in n or "blk" in n or "stl" in n:
        return +1
    # default: leave unconstrained
    return 0


def _build_monotone_constraints(feature_cols: List[str]) -> List[int]:
    return [_guess_monotone_for_feature(c) for c in feature_cols]



def _train_ltr_booster_with_refit(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    *,
    season_zscore: bool,
    val_last_k: int,
    cfg: Dict,
    monotone_constraints: Optional[List[int]] = None,
) -> lgb.Booster:
    """
    1) Train on train-part, validate on last K seasons => find best_iteration via early stop on mean_spearman
    2) Refit on ALL seasons in train_df using best_iteration (no data wasted)
    """
    X_all, y_all, group_all, _, season_order = _make_ltr_arrays(train_df, feature_cols, season_zscore=season_zscore)

    tr_idx, va_idx, group_tr, group_val = _split_groups_last_k(season_order, group_all, val_last_k)
    use_val = len(va_idx) > 0 and len(group_val) > 0 and len(tr_idx) > 0

    params = {
        "objective": "lambdarank",
        "boosting_type": "gbdt",
        "learning_rate": cfg.get("learning_rate", 0.03),
        "num_leaves": cfg.get("num_leaves", 63),
        "min_data_in_leaf": cfg.get("min_child_samples", 30),
        "feature_fraction": cfg.get("colsample_bytree", 0.85),
        "bagging_fraction": cfg.get("subsample", 0.85),
        "bagging_freq": 1,
        "lambda_l2": cfg.get("reg_lambda", 1.0),
        "seed": cfg.get("seed", RANDOM_STATE),
        "verbosity": -1,
        "metric": "None",
        "label_gain": list(range(int(np.max(y_all)) + 1)),
    }

    if monotone_constraints is not None:
        if len(monotone_constraints) != len(feature_cols):
            raise ValueError(
                f"monotone_constraints length {len(monotone_constraints)} != num_features {len(feature_cols)}"
            )
        params["monotone_constraints"] = monotone_constraints

    num_boost_round = int(cfg.get("num_boost_round", 8000))
    early_stopping_rounds = int(cfg.get("early_stopping_rounds", 250))
    log_every = int(cfg.get("log_every", 250))

    if use_val:
        dtrain = lgb.Dataset(X_all.iloc[tr_idx], label=y_all[tr_idx])
        dtrain.set_group(group_tr)
        dvalid = lgb.Dataset(X_all.iloc[va_idx], label=y_all[va_idx], reference=dtrain)
        dvalid.set_group(group_val)

        booster = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            feval=_feval_mean_spearman,
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False),
                lgb.log_evaluation(log_every),
            ],
        )
        best_iter = booster.best_iteration or num_boost_round
    else:
        best_iter = int(min(num_boost_round, 2000))

    dtrain_full = lgb.Dataset(X_all, label=y_all)
    dtrain_full.set_group(group_all)

    booster_full = lgb.train(
        params=params,
        train_set=dtrain_full,
        num_boost_round=int(best_iter),
    )
    return booster_full


def ltr_fit_predict(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_cols: List[str],
    *,
    season_zscore: bool,
    val_last_k: int,
    ensemble_cfgs: List[Dict],
    monotone: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Train an ensemble of boosters on train_df, predict scores on pred_df.
    Returns scores aligned with pred_sorted (season,pick sorted).
    """
    pred_X, _, _, pred_sorted, _ = _make_ltr_arrays(pred_df, feature_cols, season_zscore=season_zscore)

    monotone_constraints: Optional[List[int]] = None
    if monotone:
        monotone_constraints = _build_monotone_constraints(feature_cols)


    scores_sum = np.zeros(len(pred_X), dtype=float)
    for cfg in ensemble_cfgs:
        booster = _train_ltr_booster_with_refit(
            train_df, feature_cols,
            season_zscore=season_zscore,
            val_last_k=val_last_k,
            cfg=cfg,
            monotone_constraints=monotone_constraints,
        )
        scores_sum += booster.predict(pred_X)

    scores = scores_sum / max(1, len(ensemble_cfgs))
    return scores, pred_sorted


# ============================================================
# Pairwise baseline
# ============================================================

def fit_preprocessors(train_df: pd.DataFrame, feature_cols: Iterable[str]) -> Tuple[SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    imputer.fit(train_df[list(feature_cols)])
    scaler.fit(imputer.transform(train_df[list(feature_cols)]))
    return imputer, scaler


def transform_features(df: pd.DataFrame, feature_cols: Iterable[str], imputer: SimpleImputer, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(imputer.transform(df[list(feature_cols)]))


def build_pairwise_samples(
    df: pd.DataFrame,
    features: np.ndarray,
    within_position_only: bool,
    *,
    max_pairs_per_season: Optional[int],
    symmetric_pairs: bool,
    weight_by_pick_gap: bool,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    x_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    g_rows: List[int] = []
    w_rows: List[float] = []

    for season, idx in df.groupby("season").groups.items():
        season_idx = list(idx)
        if len(season_idx) < 2:
            continue

        picks = df.loc[season_idx, "overall_pick"].to_numpy()
        season_feats = features[season_idx]

        if within_position_only:
            pos_cols = [c for c in df.columns if c.startswith("position_group_")]
            if pos_cols:
                pos_mat = df.loc[season_idx, pos_cols].to_numpy()
                pos_key = pos_mat.argmax(axis=1)
            else:
                pos_key = np.zeros(len(season_idx), dtype=int)
        else:
            pos_key = None

        pairs: List[Tuple[int, int]] = []
        for i, j in combinations(range(len(season_idx)), 2):
            if picks[i] == picks[j]:
                continue
            if within_position_only and pos_key is not None and pos_key[i] != pos_key[j]:
                continue
            pairs.append((i, j))

        if max_pairs_per_season and max_pairs_per_season > 0 and len(pairs) > max_pairs_per_season:
            sel = rng.choice(len(pairs), size=max_pairs_per_season, replace=False)
            pairs = [pairs[k] for k in sel]

        for i, j in pairs:
            diff = season_feats[i] - season_feats[j]
            label = 1 if picks[i] < picks[j] else 0
            weight = float(abs(int(picks[i]) - int(picks[j]))) if weight_by_pick_gap else 1.0

            x_rows.append(diff)
            y_rows.append(label)
            g_rows.append(int(season))
            w_rows.append(weight)

            if symmetric_pairs:
                x_rows.append(-diff)
                y_rows.append(1 - label)
                g_rows.append(int(season))
                w_rows.append(weight)

    if not x_rows:
        d = features.shape[1]
        return np.empty((0, d)), np.empty((0,), int), np.empty((0,), int), np.empty((0,), float)

    X = np.vstack(x_rows)
    y = np.array(y_rows, dtype=int)
    groups = np.array(g_rows, dtype=int)
    weights = np.array(w_rows, dtype=float)
    weights = weights / np.mean(weights) if weights.size else weights
    return X, y, groups, weights


def _build_pairwise_model_factories() -> List[Tuple[str, Callable[[], object]]]:
    factories: List[Tuple[str, Callable[[], object]]] = [
        ("log_reg", lambda: LogisticRegression(
            max_iter=5000, fit_intercept=False, class_weight="balanced", n_jobs=-1, solver="lbfgs", C=1.0
        )),
        ("sgd_log", lambda: SGDClassifier(
            loss="log_loss", penalty="l2", alpha=5e-5, max_iter=8000, tol=1e-4,
            fit_intercept=False, class_weight="balanced", random_state=RANDOM_STATE
        )),
        ("hgb", lambda: HistGradientBoostingClassifier(
            learning_rate=0.06, max_depth=6, max_leaf_nodes=31, min_samples_leaf=30,
            l2_regularization=1e-2, early_stopping=True, random_state=RANDOM_STATE
        )),
        ("extra_trees", lambda: ExtraTreesClassifier(
            n_estimators=900, max_depth=None, min_samples_leaf=8, max_features="sqrt",
            n_jobs=-1, class_weight="balanced", random_state=RANDOM_STATE
        )),
    ]
    if LGBMClassifier is not None:
        factories.append(("lgbm", lambda: LGBMClassifier(
            n_estimators=2500, learning_rate=0.03, num_leaves=63,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0, min_child_samples=30,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1
        )))
    return factories


def fit_pairwise_models_and_weights(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sample_weight: np.ndarray,
    *,
    n_splits: int = 5,
) -> Tuple[List[Tuple[str, object]], np.ndarray]:
    uniq = np.unique(groups)
    n_splits = int(min(max(2, n_splits), len(uniq)))
    factories = _build_pairwise_model_factories()
    gkf = GroupKFold(n_splits=n_splits)

    mean_losses: List[float] = []
    for name, make in factories:
        fold_losses = []
        for tr, va in gkf.split(X, y, groups=groups):
            m = make()
            m.fit(X[tr], y[tr], sample_weight=sample_weight[tr])
            p = _predict_proba(m, X[va])
            ll = log_loss(y[va], np.clip(p, 1e-6, 1 - 1e-6), sample_weight=sample_weight[va])
            fold_losses.append(float(ll))
        mean_ll = float(np.mean(fold_losses)) if fold_losses else 0.69
        mean_losses.append(mean_ll)
        print(f"[CV] {name:>12s}  logloss={mean_ll:.4f}")

    losses = np.array(mean_losses, dtype=float)
    w = np.exp(-(losses - losses.min()))
    w = np.maximum(w, 1e-3)
    w = w / w.sum()

    fitted: List[Tuple[str, object]] = []
    for (name, make) in factories:
        m = make()
        m.fit(X, y, sample_weight=sample_weight)
        fitted.append((name, m))

    print("Ensemble weights:", {name: float(wi) for (name, _), wi in zip(fitted, w)})
    return fitted, w


def ensemble_probabilities(models: List[Tuple[str, object]], weights: np.ndarray, X: np.ndarray) -> np.ndarray:
    P = np.vstack([_predict_proba(m, X) for _, m in models])
    return (weights.reshape(-1, 1) * P).sum(axis=0)


def compute_player_scores_pairwise(
    df: pd.DataFrame,
    features: np.ndarray,
    models: List[Tuple[str, object]],
    model_weights: np.ndarray,
    *,
    within_position_only: bool,
) -> np.ndarray:
    scores = np.zeros(len(df), dtype=float)
    games = np.zeros(len(df), dtype=float)

    for season, idx in df.groupby("season").groups.items():
        season_idx = list(idx)
        if len(season_idx) < 2:
            continue
        season_feats = features[season_idx]

        if within_position_only:
            pos_cols = [c for c in df.columns if c.startswith("position_group_")]
            if pos_cols:
                pos_mat = df.loc[season_idx, pos_cols].to_numpy()
                pos_key = pos_mat.argmax(axis=1)
            else:
                pos_key = np.zeros(len(season_idx), dtype=int)
        else:
            pos_key = None

        diffs: List[np.ndarray] = []
        pair_i: List[int] = []
        pair_j: List[int] = []
        for i, j in combinations(range(len(season_idx)), 2):
            if within_position_only and pos_key is not None and pos_key[i] != pos_key[j]:
                continue
            diffs.append(season_feats[i] - season_feats[j])
            pair_i.append(i)
            pair_j.append(j)

        if not diffs:
            continue
        X_pairs = np.vstack(diffs)
        probs = ensemble_probabilities(models, model_weights, X_pairs)

        for k, p in enumerate(probs):
            gi = season_idx[pair_i[k]]
            gj = season_idx[pair_j[k]]
            scores[gi] += p
            scores[gj] += 1.0 - p
            games[gi] += 1.0
            games[gj] += 1.0

    games = np.maximum(games, 1.0)
    return scores / games


def pairwise_fit_predict(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_cols: List[str],
    *,
    within_position_pairs: bool,
    max_pairs_per_season: int,
    n_splits: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    train_sorted = _sort_df(train_df)
    pred_sorted = _sort_df(pred_df)

    monotone_constraints: Optional[List[int]] = None
    monotone_constraints = _build_monotone_constraints(feature_cols)


    imputer, scaler = fit_preprocessors(train_sorted, feature_cols)
    train_features = transform_features(train_sorted, feature_cols, imputer, scaler)
    pred_features = transform_features(pred_sorted, feature_cols, imputer, scaler)

    X_train, y_train, g_train, w_train = build_pairwise_samples(
        train_sorted,
        train_features,
        within_position_pairs,
        max_pairs_per_season=(None if max_pairs_per_season <= 0 else max_pairs_per_season),
        symmetric_pairs=SYMMETRIC_PAIRS,
        weight_by_pick_gap=WEIGHT_BY_PICK_GAP,
        random_state=RANDOM_STATE,
    )
    if X_train.shape[0] == 0:
        raise RuntimeError("No training pairs were constructed. Check filters or data.")

    models, weights = fit_pairwise_models_and_weights(X_train, y_train, g_train, w_train, n_splits=n_splits)
    scores = compute_player_scores_pairwise(
        pred_sorted, pred_features, models, weights, within_position_only=within_position_pairs
    )
    return scores, pred_sorted


# ============================================================
# Hybrid (proper alpha tuning on held-out seasons)
# ============================================================

def _split_alpha_val(train_df: pd.DataFrame, alpha_val_last_k: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seasons = np.sort(train_df["season"].unique())
    if len(seasons) <= 2:
        return train_df, train_df.iloc[0:0].copy()
    k = max(1, min(alpha_val_last_k, len(seasons) - 1))
    alpha_val_seasons = set(seasons[-k:])
    alpha_val = train_df[train_df["season"].isin(alpha_val_seasons)].copy()
    train_sub = train_df[~train_df["season"].isin(alpha_val_seasons)].copy()
    return train_sub.reset_index(drop=True), alpha_val.reset_index(drop=True)


def _avg_spearman(df_sorted: pd.DataFrame, scores: np.ndarray) -> float:
    corrs = evaluate_spearman_by_season(df_sorted, scores)
    vals = [c for _, c in corrs if not np.isnan(c)]
    return float(np.mean(vals)) if vals else float("nan")


def grid_search_ltr(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    *,
    season_zscore: bool,
    val_last_k: int,
    search_space: Optional[List[Dict]] = None,
    log_every: int = 200,
) -> Tuple[Dict, float, List[Tuple[Dict, float, int]]]:
    """
    Grid search LTR hyperparameters using last-k seasons as validation.
    Returns (best_cfg, best_score, results).
    """
    space = search_space if search_space is not None else _default_ltr_search_space()
    if not space:
        raise ValueError("LTR search space is empty.")

    X_df, y, group_sizes, df_sorted, season_order = _make_ltr_arrays(
        train_df, feature_cols, season_zscore=season_zscore
    )
    tr_idx, va_idx, group_tr, group_val = _split_groups_last_k(season_order, group_sizes, val_last_k)
    if len(tr_idx) == 0 or len(group_tr) == 0:
        raise RuntimeError("Not enough seasons left for training after validation split. Decrease ltr_val_last_k.")
    if len(va_idx) == 0 or len(group_val) == 0:
        raise RuntimeError("Not enough seasons to perform validation-based grid search.")

    results: List[Tuple[Dict, float, int]] = []
    best_cfg: Optional[Dict] = None
    best_score = -1e9

    for i, cfg in enumerate(space, 1):
        params = {
            "objective": "lambdarank",
            "boosting_type": "gbdt",
            "learning_rate": cfg.get("learning_rate", 0.04),
            "num_leaves": cfg.get("num_leaves", 31),
            "min_data_in_leaf": cfg.get("min_child_samples", 20),
            "feature_fraction": cfg.get("colsample_bytree", 0.85),
            "bagging_fraction": cfg.get("subsample", 0.85),
            "bagging_freq": 1,
            "lambda_l2": cfg.get("reg_lambda", 1.0),
            "seed": cfg.get("seed", RANDOM_STATE),
            "verbosity": -1,
            "metric": "None",
            "label_gain": list(range(int(np.max(y)) + 1)),
        }
        num_boost_round = int(cfg.get("num_boost_round", 2000))
        early_stopping_rounds = int(cfg.get("early_stopping_rounds", 200))

        dtrain = lgb.Dataset(X_df.iloc[tr_idx], label=y[tr_idx])
        dtrain.set_group(group_tr)
        dvalid = lgb.Dataset(X_df.iloc[va_idx], label=y[va_idx], reference=dtrain)
        dvalid.set_group(group_val)

        booster = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            feval=_feval_mean_spearman,
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False),
                lgb.log_evaluation(log_every),
            ],
        )

        best_iter = booster.best_iteration or num_boost_round
        preds = booster.predict(X_df.iloc[va_idx], num_iteration=best_iter)
        val_df = df_sorted.iloc[va_idx].reset_index(drop=True)
        score = _avg_spearman(val_df, preds)
        results.append((cfg, score, int(best_iter)))

        if not np.isnan(score) and score > best_score:
            best_score = score
            best_cfg = cfg

        print(f"[Grid LTR] {i}/{len(space)} cfg={cfg}  val_spearman={score:.4f}  best_iter={best_iter}")

    if best_cfg is None:
        raise RuntimeError("Grid search failed to find a valid configuration.")

    print("[Grid LTR] Best cfg:", best_cfg, f"val_spearman={best_score:.4f}")
    return best_cfg, best_score, results


def random_search_ltr(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    *,
    season_zscore: bool,
    val_last_k: int,
    n_trials: int = 15,
    seed: int = RANDOM_STATE,
    log_every: int = 200,
) -> Tuple[Dict, float, List[Tuple[Dict, float, int]]]:
    """
    Random search over a wider hyperparam space for LTR.
    Returns (best_cfg, best_score, results).
    """
    rng = np.random.default_rng(seed)
    space: List[Dict] = []
    for _ in range(max(1, n_trials)):
        cfg = {
            "num_leaves": int(rng.choice([15, 31, 47, 63, 95])),
            "learning_rate": float(rng.uniform(0.025, 0.08)),
            "min_child_samples": int(rng.choice([15, 20, 25, 30, 40])),
            "colsample_bytree": float(rng.uniform(0.7, 0.95)),
            "subsample": float(rng.uniform(0.7, 0.95)),
            "reg_lambda": float(rng.uniform(0.0, 1.5)),
            "num_boost_round": int(rng.integers(1500, 4500)),
        }
        space.append(cfg)

    print(f"[Random LTR] Trials: {len(space)}")
    return grid_search_ltr(
        train_df,
        feature_cols,
        season_zscore=season_zscore,
        val_last_k=val_last_k,
        search_space=space,
        log_every=log_every,
    )


def tune_alpha_on_val(
    val_sorted: pd.DataFrame,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    step: float = 0.05,
) -> float:
    best_alpha = 0.0
    best = -1e9
    alphas = np.arange(0.0, 1.0 + 1e-9, step)
    for a in alphas:
        s = a * scores_a + (1.0 - a) * scores_b
        m = _avg_spearman(val_sorted, s)
        if not np.isnan(m) and m > best:
            best = m
            best_alpha = float(a)
    print(f"[Hybrid] alpha tuning on held-out train seasons: best_alpha={best_alpha:.2f}, val_avg_spearman={best:.3f}")
    return best_alpha


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(_resolve_default_data_path()))
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUTPUT_PATH))
    ap.add_argument("--mode", type=str, default="ltr", choices=["ltr", "pairwise", "hybrid"])

    ap.add_argument("--train_last_season", type=int, default=None, help="train on seasons <= this")
    ap.add_argument("--holdout_seasons", type=int, default=DEFAULT_HOLDOUT_SEASONS)

    # LTR knobs
    ap.add_argument("--ltr_season_zscore", action="store_true",
                    help="Enable per-season z-score normalization for LTR (recommended on your data).")
    ap.add_argument("--ltr_val_last_k", type=int, default=2,
                    help="Use last K seasons inside training for early-stopping selection.")
    ap.add_argument("--ltr_grid_search", action="store_true",
                    help="Run small grid search on train split to pick LTR hyperparameters.")
    ap.add_argument("--ltr_random_search_trials", type=int, default=0,
                    help="If >0, run random search with this many trials to pick LTR hyperparameters.")
    ap.add_argument("--feature_select_top_k", type=int, default=None,
                    help="If set, select top K features using Spearman+mutual-info on train split.")
    ap.add_argument("--feature_set", type=str, default="per_min",
                    choices=["per_min", "totals", "both"],
                    help="Feature set to use. per_min (default) avoids collinearity between totals and per-minute stats.")
    ap.add_argument("--ltr_monotone", action="store_true",
                    help="Enable monotone constraints for LTR (helps against overfitting on small feature sets).")
    ap.add_argument("--no_dedupe_by_pick", action="store_true",
                    help="Disable deduping duplicated (season, overall_pick) rows (not recommended).")
    # Pairwise knobs
    ap.add_argument("--within_position_pairs", action="store_true")
    ap.add_argument("--max_pairs_per_season", type=int, default=MAX_PAIRS_PER_SEASON)
    ap.add_argument("--n_splits", type=int, default=5)

    # Hybrid knobs
    ap.add_argument("--alpha_val_last_k", type=int, default=2,
                    help="In HYBRID mode, hold out last K seasons inside training to tune blend alpha (no leakage).")
    ap.add_argument("--alpha_step", type=float, default=0.05)

    args = ap.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)

    df = load_and_clean(data_path, dedupe_by_pick=not args.no_dedupe_by_pick)
    feature_cols, dropped_cols = pick_feature_columns(df, feature_set=args.feature_set)

    selected_scores: List[Tuple[str, float, float, float]] = []

    seasons = np.sort(df["season"].unique())
    if args.train_last_season is None:
        if len(seasons) <= args.holdout_seasons:
            train_last = int(seasons.max())
        else:
            train_last = int(seasons[-(args.holdout_seasons + 1)])
    else:
        train_last = int(args.train_last_season)

    train_df = df[df["season"] <= train_last].reset_index(drop=True)
    test_df = df[df["season"] > train_last].reset_index(drop=True)

    if args.feature_select_top_k is not None:
        feature_cols, selected_scores = select_top_features(train_df, feature_cols, args.feature_select_top_k)
        if selected_scores:
            print(f"[FeatureSelect] top {len(feature_cols)} features (combined Spearman/MI):")
            for col, sp, mi, score in selected_scores[:args.feature_select_top_k]:
                print(f"  {col:20s}  spearman={sp:+.3f}  mi={mi:.4f}  combined={score:.3f}")

    print(f"Loaded {len(df)} players from {data_path}")
    print(f"Seasons: {int(seasons.min())}–{int(seasons.max())} (n={len(seasons)}); train_last_season={train_last}")
    print(f"Train players: {len(train_df)} | Test players: {len(test_df)}")
    print(f"Kept {len(feature_cols)} features (coverage >= {MIN_FEATURE_COVERAGE:.2f}). Dropped {len(dropped_cols)} sparse features.")

    if len(feature_cols) == 0:
        raise RuntimeError("No usable features after coverage filtering.")
    if len(test_df) == 0:
        print("No held-out seasons to score (test_df is empty). Exiting.")
        return

    season_zscore = bool(args.ltr_season_zscore)

    # LTR ensemble configs
    ensemble_cfgs = [
        {"seed": RANDOM_STATE, "num_leaves": 31, "learning_rate": 0.05, "num_boost_round": 3000, "min_child_samples": 20},
        {"seed": RANDOM_STATE + 101, "num_leaves": 47, "learning_rate": 0.04, "num_boost_round": 3500, "min_child_samples": 25},
        {"seed": RANDOM_STATE + 202, "num_leaves": 63, "learning_rate": 0.035, "num_boost_round": 4000, "min_child_samples": 30},
    ]
    if args.ltr_random_search_trials and args.ltr_random_search_trials > 0:
        best_cfg, best_score, _ = random_search_ltr(
            train_df,
            feature_cols,
            season_zscore=season_zscore,
            val_last_k=args.ltr_val_last_k,
            n_trials=args.ltr_random_search_trials,
        )
        seeds = [RANDOM_STATE, RANDOM_STATE + 101, RANDOM_STATE + 202]
        ensemble_cfgs = [{**best_cfg, "seed": s} for s in seeds]
        print(f"[Random LTR] Using best cfg for ensemble: val_spearman={best_score:.4f}")
    elif args.ltr_grid_search:
        best_cfg, best_score, _ = grid_search_ltr(
            train_df,
            feature_cols,
            season_zscore=season_zscore,
            val_last_k=args.ltr_val_last_k,
        )
        seeds = [RANDOM_STATE, RANDOM_STATE + 101, RANDOM_STATE + 202]
        ensemble_cfgs = [{**best_cfg, "seed": s} for s in seeds]
        print(f"[Grid LTR] Using best cfg for ensemble: val_spearman={best_score:.4f}")

    if args.mode == "ltr":
        scores, test_sorted = ltr_fit_predict(
            train_df,
            test_df,
            feature_cols,
            season_zscore=season_zscore,
            val_last_k=args.ltr_val_last_k,
            ensemble_cfgs=ensemble_cfgs,
            monotone=args.ltr_monotone,
        )
        write_rankings_csv(out_path, test_sorted, scores)
        corrs = evaluate_spearman_by_season(test_sorted, scores)
        print("Spearman correlations by season (pred_score vs. draft order):")
        for s, c in sorted(corrs):
            print(f"  {s}: {c:.3f}")
        vals = [c for _, c in corrs if not np.isnan(c)]
        if vals:
            print(f"Avg Spearman over test seasons: {float(np.mean(vals)):.3f}")
        return

    if args.mode == "pairwise":
        scores, test_sorted = pairwise_fit_predict(
            train_df,
            test_df,
            feature_cols,
            within_position_pairs=args.within_position_pairs,
            max_pairs_per_season=args.max_pairs_per_season,
            n_splits=args.n_splits,
        )
        write_rankings_csv(out_path, test_sorted, scores)
        corrs = evaluate_spearman_by_season(test_sorted, scores)
        print("Spearman correlations by season (pred_score vs. draft order):")
        for s, c in sorted(corrs):
            print(f"  {s}: {c:.3f}")
        vals = [c for _, c in corrs if not np.isnan(c)]
        if vals:
            print(f"Avg Spearman over test seasons: {float(np.mean(vals)):.3f}")
        return

    # HYBRID: tune alpha on held-out seasons within training (no leakage)
    train_sub, alpha_val = _split_alpha_val(train_df, args.alpha_val_last_k)
    if len(alpha_val) == 0:
        print("[Hybrid] Not enough seasons to create alpha validation split; falling back to pairwise.")
        scores, test_sorted = pairwise_fit_predict(
            train_df, test_df, feature_cols,
            within_position_pairs=args.within_position_pairs,
            max_pairs_per_season=args.max_pairs_per_season,
            n_splits=args.n_splits,
        )
        write_rankings_csv(out_path, test_sorted, scores)
        return

    # predictions on alpha-val from models trained on train_sub
    ltr_val_scores, val_sorted = ltr_fit_predict(
        train_sub, alpha_val, feature_cols,
        season_zscore=season_zscore,
        val_last_k=max(1, min(args.ltr_val_last_k, max(1, len(np.sort(train_sub["season"].unique())) - 1))),
        ensemble_cfgs=ensemble_cfgs,
            monotone=args.ltr_monotone,
    )
    pw_val_scores, _ = pairwise_fit_predict(
        train_sub, alpha_val, feature_cols,
        within_position_pairs=args.within_position_pairs,
        max_pairs_per_season=args.max_pairs_per_season,
        n_splits=args.n_splits,
    )

    alpha = tune_alpha_on_val(val_sorted, pw_val_scores, ltr_val_scores, step=args.alpha_step)

    # fit final models on full train_df and predict test_df
    ltr_test_scores, test_sorted = ltr_fit_predict(
        train_df, test_df, feature_cols,
        season_zscore=season_zscore,
        val_last_k=args.ltr_val_last_k,
        ensemble_cfgs=ensemble_cfgs,
        monotone=args.ltr_monotone,
    )
    pw_test_scores, test_sorted2 = pairwise_fit_predict(
        train_df, test_df, feature_cols,
        within_position_pairs=args.within_position_pairs,
        max_pairs_per_season=args.max_pairs_per_season,
        n_splits=args.n_splits,
    )

    # ensure same order
    if not np.array_equal(test_sorted["season"].to_numpy(), test_sorted2["season"].to_numpy()) or \
       not np.array_equal(test_sorted["overall_pick"].to_numpy(), test_sorted2["overall_pick"].to_numpy()):
        key = ["season", "overall_pick"]
        a = test_sorted[key].copy()
        a["_idxA"] = np.arange(len(a))
        b = test_sorted2[key].copy()
        b["_idxB"] = np.arange(len(b))
        m = a.merge(b, on=key, how="inner")
        ltr_test_scores = ltr_test_scores[m["_idxA"].to_numpy()]
        pw_test_scores = pw_test_scores[m["_idxB"].to_numpy()]
        test_sorted = test_sorted.iloc[m["_idxA"].to_numpy()].reset_index(drop=True)

    hybrid_scores = alpha * pw_test_scores + (1.0 - alpha) * ltr_test_scores

    write_rankings_csv(out_path, test_sorted, hybrid_scores)

    corrs = evaluate_spearman_by_season(test_sorted, hybrid_scores)
    print("Spearman correlations by season (pred_score vs. draft order):")
    for s, c in sorted(corrs):
        print(f"  {s}: {c:.3f}")
    vals = [c for _, c in corrs if not np.isnan(c)]
    if vals:
        print(f"Avg Spearman over test seasons: {float(np.mean(vals)):.3f}")


if __name__ == "__main__":
    main()
