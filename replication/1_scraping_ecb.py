# 1_scraping_ecb.py
# Scrape ECB monetary-policy press conference statement URLs and their publication dates.
# I/O:
#   Inputs : Optional env vars ECB_SCROLL_STEPS, ECB_SCROLL_PX, ECB_SCROLL_SLEEP to control Selenium scrolling.
#   Outputs: data_raw/ecb_speech_urls.csv with columns [date, url].
# Notes:
#   The script first collects statement links from the ECB index page (dynamic content via scrolling),
#   then visits each statement page to extract its publication date from the Open Graph meta tag.

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from tqdm import tqdm


INDEX_URL = "https://www.ecb.europa.eu/press/pressconf/html/index.en.html"
BASE_URL = "https://www.ecb.europa.eu"
UA = {"User-Agent": "Mozilla/5.0"}


def get_project_root() -> Path:
    """Return repository root (project root is parent of replication/)."""
    return Path(__file__).resolve().parents[1]


def get_scroll_params() -> tuple[int, int, float]:
    """Return (scroll_steps, scroll_px, sleep_s) from env (or defaults)."""
    scroll_steps = int(os.getenv("ECB_SCROLL_STEPS", "200"))
    scroll_px = int(os.getenv("ECB_SCROLL_PX", "400"))
    sleep_s = float(os.getenv("ECB_SCROLL_SLEEP", "0.3"))
    return scroll_steps, scroll_px, sleep_s


def scrape_statement_urls(scroll_steps: int = 200, scroll_px: int = 400, sleep_s: float = 0.3) -> pd.DataFrame:
    """Collect statement relative links from the ECB press conference index page."""
    driver = webdriver.Firefox()
    try:
        driver.get(INDEX_URL)
        for _ in range(scroll_steps):
            driver.execute_script(f"window.scrollBy(0, {scroll_px});")
            time.sleep(sleep_s)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        links = sorted({a["href"] for a in soup.find_all("a", href=True)})

        df = pd.DataFrame(links, columns=["link"])
        df = df.loc[
            df["link"].str.contains("press_conference/monetary-policy", na=False)
            & df["link"].str.contains(r"\.en\.html$", na=False)
        ].reset_index(drop=True)
        return df
    finally:
        driver.quit()


def fetch_publication_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Fetch each statement page and extract its publication date from the 'article:published_time' meta tag."""
    session = requests.Session()
    session.headers.update(UA)

    dates = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching dates"):
        url = BASE_URL + row["link"]
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            page = BeautifulSoup(resp.content, "html.parser")
            meta = page.find("meta", attrs={"property": "article:published_time"})
            dates.append(meta["content"] if meta and meta.get("content") else None)
        except requests.exceptions.RequestException:
            dates.append(None)

    out = df.copy()
    out["date"] = pd.Series(dates, dtype="object")
    out["url"] = BASE_URL + out["link"]

    out["date"] = out["date"].astype(str).str.slice(0, 10)
    out = out[
        out["url"]
        != "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/html/index.en.html"
    ].reset_index(drop=True)

    return out[["date", "url"]]


def write_output(df: pd.DataFrame, out_path: Path) -> None:
    """Write the scraped dataset to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path}")


def main() -> None:
    """Execute the scraping pipeline and save the final [date, url] dataset."""
    project_root = get_project_root()
    out_path = project_root / "data_raw" / "ecb_speech_urls.csv"

    scroll_steps, scroll_px, sleep_s = get_scroll_params()

    df_links = scrape_statement_urls(scroll_steps=scroll_steps, scroll_px=scroll_px, sleep_s=sleep_s)
    print(f"TOTAL DOCUMENTS (links): {len(df_links)}")

    df_out = fetch_publication_dates(df_links)
    print(df_out.head())

    write_output(df_out, out_path)


if __name__ == "__main__":
    main()