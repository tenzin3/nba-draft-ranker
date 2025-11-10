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

    def scrap_college_stats(self, first_name:str, last_name:str, year:int):
        url = f"{self.college_stat_website}/{first_name.lower()}-{last_name.lower()}-1.html"
        try:
            resp = self.session.get(url, headers=self.headers, timeout=20)
        except requests.exceptions.ReadTimeout:
            print(f"Timeout for {url}")
            return
        except requests.exceptions.RequestException as e:
            print(f"Request error for {url}: {e}")
            return
        if not resp.ok:
            print(f"HTTP {resp.status_code} for {url}")
            return

        soup = BeautifulSoup(resp.text, 'html.parser')

        output_path = self.college_stats_dir / f"{year}/{first_name.lower()}-{last_name.lower()}.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(soup.prettify())


    def scrap_season_college_stats(self, year:int):
        # Read the draft file
        df = pd.read_csv(self.draft_dir / f"draft_{year}.csv")

        # Iterate over the rows to get first and last name
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Scraping college stats for {year}"):
            name = str(row.get('PLAYER_NAME', '')).strip()
            parts = name.split()
            if len(parts) < 2:
                continue
            first_name, last_name = parts[0], parts[-1]
            self.scrap_college_stats(first_name, last_name, year)
            time.sleep(random.uniform(0.5, 1.5))
        
        

if __name__ == "__main__":
    scraper = Srapper()
    scraper.scrap_season_college_stats(2000)
