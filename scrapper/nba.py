from pathlib import Path 
from tqdm import tqdm
from nba_api.stats.endpoints import drafthistory, draftcombinestats

DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_DIR.mkdir(exist_ok=True)

DRAFT_DIR = RAW_DATA_DIR / "drafts"
DRAFT_DIR.mkdir(exist_ok=True)

DRAFT_COMBINE_DIR = RAW_DATA_DIR / "draft_combine"
DRAFT_COMBINE_DIR.mkdir(exist_ok=True)


class NBASrapper:
    def __init__(self, draft_dir:Path = DRAFT_DIR, draft_combine_dir:Path = DRAFT_COMBINE_DIR):
        self.draft_dir = draft_dir
        self.draft_combine_dir = draft_combine_dir

    def scrap_draft(self, year:int):
        draft = drafthistory.DraftHistory(season_year_nullable=str(year)) 
        return draft.get_data_frames()[0]

    def scrap_all_drafts(self):
        for year in tqdm(range(2000, 2026), desc="Scraping drafts History"):
            df = self.scrap(year)
            df.to_csv(self.draft_dir / f"draft_{year}.csv", index=False)

    def scrap_draft_combine(self, year:int):
        draft_combine = draftcombinestats.DraftCombineStats(season_all_time=str(year))
        return draft_combine.get_data_frames()[0]

    def scrap_all_draft_combine(self):
        for year in tqdm(range(2016, 2026), desc="Scraping drafts History"):
            df = self.scrap_draft_combine(year)
            df.to_csv(self.draft_combine_dir / f"draft_combine_{year}.csv", index=False)

if __name__ == "__main__":
    scraper = NBASrapper()
    scraper.scrap_all_drafts()