from pathlib import Path 
from tqdm import tqdm

from nba_api.stats.endpoints import drafthistory

DRAFT_DIR = Path("drafts")
DRAFT_DIR.mkdir(exist_ok=True)

class Srapper:
    def __init__(self, draft_dir:Path = DRAFT_DIR):
        self.draft_dir = draft_dir

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