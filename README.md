# NBA Draft Ranker

End-to-end pipeline to scrape, preprocess, and rank NBA draft prospects using pairwise and learning-to-rank (LambdaRank) approaches. The project combines college stats scraped from Basketball-Reference with NBA draft and combine data from the nba_api to produce season-by-season prospect rankings and evaluation reports.

## Repository Structure

- scrapper/
  - college.py — Scrapes Basketball-Reference player pages for a given draft class and stores per-player tables as CSVs.
  - nba.py — Uses nba_api to download draft history and draft combine data.
- preprocess/
  - Notebooks for cleaning/feature engineering: college.ipynb, nba_combine.ipynb, undrafted.ipynb.
- classifier/ and ranker/
  - Notebooks and code for modeling. Main script: ranker/pairwise/pairwise_rank.py.
- data/
  - Raw: NBA draft, combine, and college stats.
  - Cleaned: College stats, NBA draft, and combine.
  - Result: Rankings, and classifier results.
- requirements.txt — Python dependencies.

## Setup

1) Python 3.10+ recommended. Create and activate a virtual environment.
2) Install dependencies:

```bash
pip install -r requirements.txt
```

Packages used include: pandas, numpy, scikit-learn, scipy, torch==2.5.0, tqdm, requests, nba_api, urllib3<2, and LightGBM (imported in the ranker). If LightGBM is not installed, LambdaRank and optional LGBMClassifier parts will require lightgbm:

```bash
pip install lightgbm
```

## Data Sources

- Basketball-Reference (college player pages) — scraped via scrapper/college.py.
- nba_api
  - Draft history via nba_api.stats.endpoints.drafthistory.
  - Draft combine via nba_api.stats.endpoints.draftcombinestats.

Please respect the target sites’ Terms of Service and rate limits.

## Scraping College Player Tables (Basketball-Reference)

Script: scrapper/college.py

- Config at top of file:
  - DRAFT_URL — e.g., https://www.basketball-reference.com/draft/NBA_2025.html
  - OUT_DIR — e.g., bbr_2025_players

Running the script downloads per-player pages for the specified draft class, parses visible and commented tables, and writes per-player CSVs plus a _scrape_summary.json file.

Example:

```bash
python scrapper/college.py
```

Output structure (example for 2025):

```
data/
  college/
    bbr_2025_players/
      _scrape_summary.json
      jamesle01/
        jamesle01.html
        <table_id>.csv
        metadata.json
      ...
```

If you run scrapper/college.py from the repo root, move or symlink the resulting bbr_<YEAR>_players folder under data/college/ so that downstream steps can find it.

## Scraping NBA Draft and Combine (nba_api)

Script: scrapper/nba.py

- Draft history: Srapper.scrap_draft(year) returns a DataFrame; scrap_all_drafts() would write to drafts/draft_<year>.csv if used similarly to the combine flow.
- Draft combine: Srapper.scrap_draft_combine(year); scrap_all_draft_combine() writes to draft_combine/draft_combine_<year>.csv.

Example (combine):

```bash
python scrapper/nba.py
```

Note: You may need to adapt years or call the class methods directly from a notebook/script based on your needs.

## Aggregating College Stats

Script: extract/pipeline.py

This aggregates the per-player scraped tables into a single CSV. It detects data/college/bbr_<YEAR>_players folders and reads each player’s all_college_stats.csv (as produced by your scraping/processing flow) to build an analysis table.

- Single season example:

```bash
python extract/pipeline.py --season 1999-00
```

Writes: data/college/college_stats_1999-00.csv and reports any players missing that season.

- All seasons for all scraped players:

```bash
python extract/pipeline.py --all
```

Writes: data/college/college_stats_all_seasons.csv.

Optional filters:

```bash
python extract/pipeline.py --season 2019-20 --draft-years 2020 2021
python extract/pipeline.py --all --draft-years 2000 2001 2002
python extract/pipeline.py --all --output data/college/custom.csv
```

## Modeling: Pairwise and LambdaRank

Script: ranker/pairwise/pairwise_rank.py

Key features:

- Pairwise baseline with an ensemble of Logistic, SGD, HistGradientBoosting, ExtraTrees, and optional LGBMClassifier.
- LambdaRank (LightGBM) with per-season z-score, custom early stopping on mean Spearman per season, time-based validation on last-K seasons, and ensemble of configs.
- Hybrid mode blends pairwise and LTR, with alpha tuned on held-out seasons.

Expected input: a CSV with at least season and overall_pick columns plus numeric features. The script will infer known name columns (e.g., player_name) for nicer outputs when present.

Common invocations (from the repo root):

```bash
# LTR with per-season z-scoring
python -m ranker.pairwise.pairwise_rank \
  --data outputs/nba_draft_final.csv \
  --train_last_season 2015 \
  --mode ltr \
  --ltr_season_zscore

# Pairwise baseline
python -m ranker.pairwise.pairwise_rank \
  --data outputs/nba_draft_final.csv \
  --train_last_season 2015 \
  --mode pairwise \
  --max_pairs_per_season 30000

# Hybrid with alpha tuned on last K seasons
python -m ranker.pairwise.pairwise_rank \
  --data outputs/nba_draft_final.csv \
  --train_last_season 2015 \
  --mode hybrid \
  --ltr_season_zscore \
  --alpha_val_last_k 2
```

Outputs include a season-wise ranking CSV (default outputs/pairwise_rankings.csv) with pred_score and pred_rank.

### Feature engineering

The script will:

- Normalize columns and coalesce duplicates.
- Create position_group dummies where available (from position or position_group).
- Add per-minute rate features from totals (e.g., totals_trb_per_min) when mp is available.
- Automatically drop text/object columns and low-coverage numeric columns.

You can control the feature set via --feature_set (per_min, totals, or both).

## Repro/Workflow Tips

1) Scrape a draft class from Basketball-Reference with scrapper/college.py and place the output under data/college/bbr_<YEAR>_players.
2) Optionally, use scrapper/nba.py to fetch draft history and combine data.
3) Aggregate college stats with extract/pipeline.py (per season or all seasons).
4) Prepare a modeling table (via notebooks or merges in your pipeline) that contains season, overall_pick, and your chosen features.
5) Train/evaluate rankings with ranker/pairwise/pairwise_rank.py in ltr, pairwise, or hybrid mode.

## Notes

- Respect robots.txt and rate limits. scrapper/college.py includes polite randomized backoff and handles HTTP 429 Retry-After headers.
- If LightGBM is not installed, LambdaRank and the optional LGBMClassifier components will be unavailable.
- Large data/ directories are expected; outputs are written to outputs/ by default.

## License

For internal/research use. Review data source licenses and terms before redistribution.
