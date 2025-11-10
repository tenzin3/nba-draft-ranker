import requests
from pathlib import Path 
from tqdm import tqdm
from bs4 import BeautifulSoup
from nba_api.stats.endpoints import drafthistory
import pandas as pd
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
import time
import random

DRAFT_DIR = Path("drafts")
DRAFT_DIR.mkdir(exist_ok=True)

COLLEGE_STATS_DIR = Path("college_stats")
COLLEGE_STATS_DIR.mkdir(exist_ok=True)

class Srapper:
    def __init__(self, draft_dir:Path = DRAFT_DIR, college_stats_dir:Path = COLLEGE_STATS_DIR):
        self.draft_dir = draft_dir
        self.college_stats_dir = college_stats_dir
        self.college_stat_website = "https://www.sports-reference.com/cbb/players/"
        self.session = requests.Session()
        retry = Retry(
            total=5,
            read=5,
            connect=3,
            backoff_factor=0.8,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9"
        }

    def scrap(self, year:int):
        # nba_api expects a string year via season_year_nullable
        draft = drafthistory.DraftHistory(season_year_nullable=str(year)) 
        return draft.get_data_frames()[0]

    def scrap_all(self):
        for year in tqdm(range(2000, 2026), desc="Scraping drafts History"):
            df = self.scrap(year)
            df.to_csv(self.draft_dir / f"draft_{year}.csv", index=False)

if __name__ == "__main__":
    scraper = Srapper()
    scraper.scrap_all()
