import os
import re
import time
import json
import random
from urllib.parse import urljoin, urlparse
import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment
from io import StringIO
from pathlib import Path
BASE = "https://www.basketball-reference.com"
DRAFT_URL = "https://www.basketball-reference.com/draft/NBA_2025.html"

DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_DIR.mkdir(exist_ok=True)

OUT_DIR = RAW_DATA_DIR / "college"/ "2025"
OUT_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

os.makedirs(OUT_DIR, exist_ok=True)

def get_soup(url, *, retries=6, backoff=2.0, timeout=40):
    """GET a page with retry/backoff and handle 429 Retry-After politely."""
    last_exc = None
    for i in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            # Handle 429 without throwing to compute custom delay
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                try:
                    delay = int(retry_after) if retry_after is not None else None
                except ValueError:
                    delay = None
                if delay is None:
                    delay = backoff * (2 ** i) + random.uniform(0.5, 1.5)
                print(f"HTTP 429 received. Sleeping for {delay:.1f}s before retrying {url}")
                time.sleep(delay)
                continue
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except requests.exceptions.HTTPError as e:
            # If 429 surfaced via HTTPError, honor Retry-After as well
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            if status == 429:
                retry_after = getattr(e.response, 'headers', {}).get("Retry-After") if getattr(e, 'response', None) else None
                try:
                    delay = int(retry_after) if retry_after is not None else None
                except ValueError:
                    delay = None
                if delay is None:
                    delay = backoff * (2 ** i) + random.uniform(0.5, 1.5)
                print(f"HTTPError 429. Sleeping for {delay:.1f}s before retrying {url}")
                time.sleep(delay)
                last_exc = e
                continue
            last_exc = e
        except Exception as e:
            last_exc = e
        # polite randomized backoff for other failures
        sleep_for = backoff * (2 ** i) + random.uniform(0.5, 1.5)
        time.sleep(sleep_for)
    raise last_exc

def slug_from_url(url: str) -> str:
    """
    Convert a Basketball-Reference player URL to a stable slug.
    e.g. https://www.basketball-reference.com/players/j/jamesle01.html -> jamesle01
    """
    path = urlparse(url).path
    base = os.path.basename(path)  # jamesle01.html
    return re.sub(r"\.html$", "", base)

def get_drafted_player_links(draft_url=DRAFT_URL):
    """
    Parse the draft page for player profile links.
    Looks for <td data-stat='player'><a href='...'></a></td>.
    """
    soup = get_soup(draft_url)
    links = []
    for td in soup.select("td[data-stat='player'] a"):
        href = td.get("href")
        if href and href.startswith("/players/"):
            links.append(urljoin(BASE, href))
    # de-dup while preserving order
    seen = set()
    unique_links = []
    for u in links:
        if u not in seen:
            unique_links.append(u)
            seen.add(u)
    return unique_links

def parse_all_tables_from_player_page(soup: BeautifulSoup):
    """
    Basketball-Reference often stores tables inside HTML comments.
    We'll collect:
      1) normal (visible) <table> elements
      2) tables embedded within <!-- ... -->
    Returns dict: {table_id_or_name: DataFrame}
    """
    tables = {}

    def add_table(tag):
        # table id (preferred) or a fallback name
        tid = tag.get("id")
        key = tid if tid else f"table_{len(tables)+1}"
        try:
            # read_html returns a list; wrap tag into a string
            dfs = pd.read_html(StringIO(str(tag)))
            if dfs:
                tables[key] = dfs[0]
        except ValueError:
            # no table found by pandas in this tag
            pass

    # 1) visible tables
    for t in soup.find_all("table"):
        add_table(t)

    # 2) commented tables (inside <!-- ... -->)
    for c in soup.find_all(string=lambda text: isinstance(text, Comment)):
        # if a comment chunk contains a table, parse it
        if "<table" in c and "</table>" in c:
            # make a new soup from the comment text
            hidden_soup = BeautifulSoup(c, "lxml")
            for t in hidden_soup.find_all("table"):
                add_table(t)

    return tables

def extract_player_name(soup: BeautifulSoup) -> str:
    """
    Try to extract the player's name from the page header.
    Basketball-Reference usually has: <h1 itemprop="name"><span>LeBron James</span></h1>
    """
    h1 = soup.find("h1")
    if h1:
        span = h1.find("span")
        if span and span.get_text(strip=True):
            return span.get_text(strip=True)
        return h1.get_text(strip=True)
    # fallback from title
    if soup.title and soup.title.string:
        return soup.title.string.split(" Stats")[0].strip()
    return "Unknown Player"

def save_player_data(player_url: str):
    """
    Download player page, parse all tables, and save:
      - raw html
      - each table as CSV under a per-player folder
      - metadata.json (name, url, tables present)
    """
    soup = get_soup(player_url)
    player_name = extract_player_name(soup)
    slug = slug_from_url(player_url)
    player_dir = os.path.join(OUT_DIR, slug)
    os.makedirs(player_dir, exist_ok=True)

    # Save raw HTML (optional but useful for debugging)
    raw_html_path = os.path.join(player_dir, f"{slug}.html")
    with open(raw_html_path, "w", encoding="utf-8") as f:
        f.write(str(soup))

    # Parse tables
    tables = parse_all_tables_from_player_page(soup)

    saved_tables = []
    for key, df in tables.items():
        # Clean column names a bit (optional)
        df.columns = [str(c).strip() for c in df.columns]
        csv_path = os.path.join(player_dir, f"{key}.csv")
        # Some tables include multi-index headers; flatten if necessary
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [" ".join([str(x) for x in tup]).strip() for tup in df.columns.values]
        df.to_csv(csv_path, index=False)
        saved_tables.append({"table_id": key, "csv": os.path.relpath(csv_path, start=OUT_DIR)})

    # Save small metadata
    metadata = {
        "name": player_name,
        "slug": slug,
        "url": player_url,
        "num_tables": len(saved_tables),
        "tables": saved_tables,
    }
    with open(os.path.join(player_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return slug, player_name, len(saved_tables)

def get_already_scraped_players():
    """
    Check which players have already been scraped by looking at existing directories.
    Returns a set of slugs that have been completed.
    """
    already_scraped = set()
    if os.path.exists(OUT_DIR):
        for item in os.listdir(OUT_DIR):
            item_path = os.path.join(OUT_DIR, item)
            # Check if it's a directory and has metadata.json (indicates complete scrape)
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, "metadata.json")
                if os.path.exists(metadata_path):
                    already_scraped.add(item)
    return already_scraped

def main():
    print(f"Fetching drafted players from: {DRAFT_URL}")
    player_links = get_drafted_player_links(DRAFT_URL)
    print(f"Found {len(player_links)} player pages.")
    
    # Check for already-scraped players
    already_scraped = get_already_scraped_players()
    if already_scraped:
        print(f"Found {len(already_scraped)} already-scraped players. Will skip them.")
    
    results = []
    skipped_count = 0
    
    for i, url in enumerate(player_links, start=1):
        slug = slug_from_url(url)
        
        # Skip if already scraped
        if slug in already_scraped:
            print(f"[{i}/{len(player_links)}] Skipping {slug} (already scraped)")
            skipped_count += 1
            # Load existing metadata for the summary
            try:
                metadata_path = os.path.join(OUT_DIR, slug, "metadata.json")
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    results.append({
                        "slug": slug, 
                        "name": metadata.get("name", "Unknown"), 
                        "url": url, 
                        "tables": metadata.get("num_tables", 0),
                        "skipped": True
                    })
            except Exception:
                pass
            continue
        
        print(f"[{i}/{len(player_links)}] Scraping {url} ...")
        try:
            slug, name, n_tables = save_player_data(url)
            results.append({"slug": slug, "name": name, "url": url, "tables": n_tables, "skipped": False})
        except Exception as e:
            print(f"  ERROR for {url}: {e}")
        
        # Polite delay to respect the site
        time.sleep(3.0 + random.uniform(1.0, 2.0))
        
        # Extra delay every 10 players
        if (i - skipped_count) % 10 == 0 and (i - skipped_count) > 0:
            print(f"  Taking a longer break after scraping {i - skipped_count} new players...")
            time.sleep(10.0)

    # Save run summary
    with open(os.path.join(OUT_DIR, "_scrape_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Output in: {OUT_DIR}")
    print(f"Total players: {len(player_links)}")
    print(f"Skipped (already scraped): {skipped_count}")
    print(f"Newly scraped: {len(player_links) - skipped_count}")
    if results:
        print(f"\nExample player folder: {OUT_DIR}/{results[0]['slug']}/")

if __name__ == "__main__":
    main()