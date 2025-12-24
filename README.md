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

# Season-Wise NBA Draft Ranking Using Collegiate Performance

This repository contains the code and data pipeline for **season-wise NBA Draft prediction** formulated as a **learning-to-rank (LTR)** problem. We build an end-to-end workflow that uses **final-season collegiate box-score statistics** to reconstruct NBA draft order within each draft class, and we additionally study a **drafted vs. undrafted** classification task.

The accompanying paper is titled:  
**“Season-Wise NBA Draft Ranking Using Collegiate Performance” (December 2025)**  
Authors: Tenzin Tsundue, Aryaman Sharma, Chuyang Ye (NYU Courant)

---

## Project Overview

### Task A — Draft Order Prediction (Learning-to-Rank)
We treat each **draft year as a ranking group** and learn a scoring function that orders prospects **within the same season**.  
- **Train:** Draft years **2000–2024**  
- **Test:** Held-out **2025** draft class (temporal split)

We compare three ranking paradigms:
- **Pointwise:** Ridge, Random Forest, ExtraTrees, HistGB, MLP (regression → sort)
- **Pairwise:** RankSVM, Pairwise Logistic, RankNet (learn preferences from within-year pairs)
- **Listwise:** ListNet, ListMLE, **LambdaMART** (optimize list-level ranking structure)

**Metrics (computed within a season):**
- **Spearman’s rank correlation (ρ)**
- **Pairwise accuracy** (fraction of correctly ordered player pairs)

**Key Result (2025 test class):**
- **LambdaMART** achieved the best overall ordering quality (highest pairwise accuracy) while maintaining strong rank correlation.

### Task B — Draftability Prediction (Binary Classification)
We also train a classifier to predict whether a prospect is **drafted (pick ≤ 60)** or **undrafted**, using the same college-only feature set.
- Evaluated under both:
  - **Natural (imbalanced) distribution**
  - **Balanced (50/50 under-sampled) test set**
- Observed **moderate ROC-AUC (~0.56–0.58)** and persistent difficulty identifying undrafted players, consistent with limited features and coverage gaps.

---

## Data Sources

We integrate multiple public sources:

- **NBA Draft + Combine (nba.com via `nba_api`)**
- **College stats (drafted players):** Basketball-Reference
- **College stats (undrafted combine participants):** Sports-Reference

> Note: **NBA Combine features are intentionally excluded from the main ranking feature set** due to high missingness and inconsistent protocols across years. The pipeline focuses on **final-season college box-score stats** for temporal consistency.

---

## Feature Set (College-only)

Final compact feature set used throughout:
- `Totals_FG`, `Totals_FT`, `Totals_TRB`, `Totals_STL`, `Totals_BLK`,
- `Totals_TOV`, `Totals_PF`, `Shooting_FG%`, `MP`, `Age`

All features are **z-score standardized** using training-only statistics (per split) to avoid leakage.

---

## Repository Structure (Suggested)

Your repo may look like this (adjust to match your actual folders):

