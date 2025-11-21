"""
Utility to aggregate college stats for a specific season across draft classes.

Usage examples:
    python proccess_pipline/pipline.py --season 1999-00
    python proccess_pipline/pipline.py --season 2000-01 --draft-years 2001
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


# Try the sibling project layout first (../nba-draft-ranker/college); fall back to ../college.
_BASE_DIR = Path(__file__).resolve().parent.parent
_PREFERRED_COLLEGE = _BASE_DIR / "nba-draft-ranker" / "college"
COLLEGE_ROOT = _PREFERRED_COLLEGE if _PREFERRED_COLLEGE.exists() else _BASE_DIR / "college"


def _flatten_columns(column_names: Iterable[str]) -> List[str]:
    """Convert tuple-like column headers to simple strings."""
    flattened: List[str] = []
    for raw in column_names:
        text = str(raw).strip()
        if text.startswith("(") and text.endswith(")"):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                flattened.append(text)
                continue

            if isinstance(parsed, Tuple) and len(parsed) == 2:
                left, right = parsed
                if str(left).startswith("Unnamed"):
                    flattened.append(str(right))
                else:
                    flattened.append(f"{left}_{right}")
                continue

        flattened.append(text)

    return flattened


def _load_player_row(player_dir: Path, season: str) -> Dict[str, object] | None:
    csv_path = player_dir / "all_college_stats.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    df.columns = _flatten_columns(df.columns)
    if "Season" not in df.columns:
        return None

    season_matches = df[df["Season"].astype(str) == season]
    if season_matches.empty:
        return None

    return season_matches.iloc[0].to_dict()


def _iter_draft_dirs(filters: List[int] | None) -> List[Tuple[int, Path]]:
    draft_dirs: List[Tuple[int, Path]] = []
    for path in COLLEGE_ROOT.iterdir():
        if not path.is_dir():
            continue
        name = path.name
        if not (name.startswith("bbr_") and name.endswith("_players")):
            continue

        try:
            year = int(name.split("_")[1])
        except (IndexError, ValueError):
            continue

        if filters and year not in filters:
            continue

        draft_dirs.append((year, path))

    return sorted(draft_dirs, key=lambda item: item[0])


def collect_season_rows(season: str, draft_years: List[int] | None = None) -> Tuple[pd.DataFrame, Dict[int, List[str]]]:
    missing: Dict[int, List[str]] = {}
    rows: List[Dict[str, object]] = []

    for draft_year, draft_dir in _iter_draft_dirs(draft_years):
        summary_path = draft_dir / "_scrape_summary.json"
        if not summary_path.exists():
            continue

        summary = json.loads(summary_path.read_text())
        for player in summary:
            slug = player.get("slug")
            if not slug:
                continue

            player_dir = draft_dir / slug
            row = _load_player_row(player_dir, season)
            if row is None:
                missing.setdefault(draft_year, []).append(slug)
                continue

            row.update(
                {
                    "draft_year": draft_year,
                    "player_slug": slug,
                    "player_name": player.get("name"),
                }
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    preferred = ["draft_year", "player_slug", "player_name", "Season"]
    ordered = preferred + [col for col in df.columns if col not in preferred]
    df = df.reindex(columns=ordered)
    return df, missing


def collect_all_rows(draft_years: List[int] | None = None) -> pd.DataFrame:
    """Return all seasons for every player in the selected draft years."""
    rows: List[Dict[str, object]] = []

    for draft_year, draft_dir in _iter_draft_dirs(draft_years):
        summary_path = draft_dir / "_scrape_summary.json"
        if not summary_path.exists():
            continue

        summary = json.loads(summary_path.read_text())
        for player in summary:
            slug = player.get("slug")
            if not slug:
                continue

            csv_path = draft_dir / slug / "all_college_stats.csv"
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            df.columns = _flatten_columns(df.columns)
            for record in df.to_dict(orient="records"):
                record.update(
                    {
                        "draft_year": draft_year,
                        "player_slug": slug,
                        "player_name": player.get("name"),
                    }
                )
                rows.append(record)

    df = pd.DataFrame(rows)
    preferred = ["draft_year", "player_slug", "player_name", "Season"]
    ordered = preferred + [col for col in df.columns if col not in preferred]
    return df.reindex(columns=ordered)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate college stats.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--season", help="Season label to extract, e.g. 1999-00")
    mode.add_argument("--all", dest="all_seasons", action="store_true", help="Collect all seasons for each player.")
    parser.add_argument(
        "--draft-years",
        nargs="*",
        type=int,
        default=None,
        help="Restrict to specific draft years (e.g. 2000 2001). Defaults to all available.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to college/college_stats_<season|all>.csv",
    )
    args = parser.parse_args()

    if args.all_seasons:
        df = collect_all_rows(args.draft_years)
        missing = None
        default_name = "college_stats_all_seasons.csv"
    else:
        df, missing = collect_season_rows(args.season, args.draft_years)
        default_name = f"college_stats_{args.season.replace('/', '-')}.csv"

    output_path = Path(args.output) if args.output else COLLEGE_ROOT / default_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} rows to {output_path}")
    if missing:
        for draft_year, slugs in missing.items():
            print(f"[{draft_year}] missing season '{args.season}' for: {', '.join(slugs)}")


if __name__ == "__main__":
    main()
