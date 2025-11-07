from nba_api.stats.endpoints import drafthistory

class Srapper:
    def __init__(self):
        pass 

    def scrap(self, year:int):
        # nba_api expects a string year via season_year_nullable
        draft = drafthistory.DraftHistory(season_year_nullable=str(year)) 
        return draft.get_data_frames()[0]

if __name__ == "__main__":
    scraper = Srapper()
    df_2025 = scraper.scrap(year=2025)
    df_2025.to_csv("draft_2025.csv", index=False)