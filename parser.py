import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup, Comment
import re

COLLEGE_STATS_DIR = Path("college_stats")


def write_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


class CollegeStatsParser:
    def __init__(self):
        pass 

    def _find_table(self, soup: BeautifulSoup, table_id_candidates: List[str]) -> Optional[BeautifulSoup]:
        # try direct lookup first
        for tid in table_id_candidates:
            table = soup.find("table", id=tid)
            if table is not None:
                return table
        # sports-reference often hides tables inside HTML comments
        for c in soup.find_all(string=lambda text: isinstance(text, Comment)):
            try:
                sub = BeautifulSoup(c, "html.parser")
            except Exception:
                continue
            for tid in table_id_candidates:
                table = sub.find("table", id=tid)
                if table is not None:
                    return table
        return None

    def _parse_table(self, table: BeautifulSoup) -> Dict[str, Any]:
        # Build column order from header using data-stat
        columns: List[str] = []
        thead = table.find("thead")
        if thead:
            header_row = thead.find("tr")
            if header_row:
                for th in header_row.find_all(["th","td"]):
                    key = th.get("data-stat") or th.get_text(strip=True)
                    if key:
                        columns.append(key)
        # Fallback: read from first body row
        if not columns:
            first = table.find("tbody").find("tr") if table.find("tbody") else None
            if first:
                for td in first.find_all(["th","td"]):
                    key = td.get("data-stat") or td.get_text(strip=True)
                    columns.append(key)

        def row_to_dict(tr) -> Dict[str, Any]:
            data: Dict[str, Any] = {}
            cells = tr.find_all(["th","td"])
            for i, cell in enumerate(cells):
                key = cell.get("data-stat") or (columns[i] if i < len(columns) else f"col_{i}")
                text = cell.get_text(strip=True)
                data[key] = text
            return data

        body_rows: List[Dict[str, Any]] = []
        tbody = table.find("tbody")
        if tbody:
            for tr in tbody.find_all("tr"):
                if "thead" in tr.get("class", []):
                    continue
                # only keep season rows that have a season label in th with scope=row
                if tr.find("th", {"scope": "row"}) is None:
                    continue
                body_rows.append(row_to_dict(tr))

        career: Optional[Dict[str, Any]] = None
        tfoot = table.find("tfoot")
        if tfoot:
            foot_tr = tfoot.find("tr")
            if foot_tr:
                career = row_to_dict(foot_tr)

        return {"rows": body_rows, "career": career}

    def parse(self, path: Path) -> Dict[str, Any]:
        """
        Parse the html file and return a json dictionary
        """
        with open(path, "r") as f:
            content = f.read()
        soup = BeautifulSoup(content, "html.parser")
        # Player name (best-effort)
        name = None
        h1 = soup.find("h1")
        if h1:
            name = h1.get_text(strip=True)
        if not name:
            title = soup.find("title")
            if title:
                name = re.sub(r"\s*-\s*College Basketball.*$", "", title.get_text(strip=True))
        # Extract per-game college table
        table = self._find_table(soup, ["players_per_game", "per_game", "player_per_game", "per_game_college"])
        per_game: Dict[str, Any] = {}
        if table:
            parsed = self._parse_table(table)
            per_game = {
                "seasons": parsed["rows"],
                "career": parsed["career"],
            }
        # Optionally extract advanced as well if present
        adv_table = self._find_table(soup, ["players_advanced", "advanced"])
        advanced: Optional[Dict[str, Any]] = None
        if adv_table:
            parsed_adv = self._parse_table(adv_table)
            advanced = {
                "seasons": parsed_adv["rows"],
                "career": parsed_adv["career"],
            }
        result: Dict[str, Any] = {
            "player": name,
            "per_game": per_game or None,
            "advanced": advanced,
            "source_path": str(path),
        }
        return result


if __name__ == "__main__":
    
    parser = CollegeStatsParser()
    result = parser.parse(COLLEGE_STATS_DIR / "2000" / "chris-mihm.html")
    write_json("chris-mihm.json", result)
