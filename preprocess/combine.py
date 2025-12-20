import pandas as pd
from pathlib import Path

current_dir = Path(__file__).parent


def combine_college_drafted_and_nba_combine():
    college_df = pd.read_csv(current_dir.parent/ "data"/ "cleaned"/ "college_drafted"/'college_drafted_last_season.csv')
    draft_df = pd.read_csv(current_dir.parent/ "data" / "cleaned" / "draft_combine.csv")

    merged_df = pd.merge(
        draft_df,
        college_df,              
        left_on=["PLAYER_NAME", "SEASON"],
        right_on=["player_name", "draft_year"],
        how="inner",                     # ⬅️ keep ALL rows from draft_df
        suffixes=("_DRAFT", "_COMBINE")
    )

    # Remove duplicates if any
    merged_df = merged_df.drop_duplicates()

    # Sort by season and draft order
    merged_df = merged_df.sort_values(by=["SEASON", "OVERALL_PICK"])
    print(f"Shape of merged_df: {merged_df.shape}")
    merged_df.to_csv(current_dir.parent/ "data" / "cleaned"/ "college_drafted_&_nba_combine_merged.csv", index=False)


if __name__ == "__main__":
    combine_college_drafted_and_nba_combine()
