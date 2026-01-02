# 1_scraping_ecb.py
# Scrape ECB monetary-policy press conference statement URLs and their publication dates.
# I/O:
#   Inputs: Optional env vars (ECB_SCROLL_STEPS, ECB_SCROLL_PX, ECB_SCROLL_SLEEP) to control page scrolling.
#   Outputs: CSV file "data_raw/ecb_speech_urls.csv" containing columns [date, url].

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


def scrape_statement_urls(scroll_steps: int = 200, scroll_px: int = 400, sleep_s: float = 0.3) -> pd.DataFrame:
    """Collect relative URLs for ECB monetary-policy statements from the press conference index page."""
    driver = webdriver.Firefox()
    try:
        driver.get(INDEX_URL)

        # Scroll to load additional items rendered dynamically on the page.
        for _ in range(scroll_steps):
            driver.execute_script(f"window.scrollBy(0, {scroll_px});")
            time.sleep(sleep_s)

        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Extract all unique hrefs, then filter to statement pages in English.
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
    """Fetch each statement page and extract its publication date from the Open Graph meta tag."""
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

    # Standardize date format to YYYY-MM-DD and drop the non-statement index URL.
    out["date"] = out["date"].astype(str).str.slice(0, 10)
    out = out[
        out["url"]
        != "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/html/index.en.html"
    ].reset_index(drop=True)

    return out[["date", "url"]]


def main() -> None:
    """Run the end-to-end scraping pipeline and write the resulting dataset."""
    # IMPORTANT: project root is the parent of the replication/ folder.
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "data_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ecb_speech_urls.csv"

    # Read optional tuning parameters from environment variables for flexible execution.
    scroll_steps = int(os.getenv("ECB_SCROLL_STEPS", "200"))
    scroll_px = int(os.getenv("ECB_SCROLL_PX", "400"))
    sleep_s = float(os.getenv("ECB_SCROLL_SLEEP", "0.3"))

    df_links = scrape_statement_urls(scroll_steps=scroll_steps, scroll_px=scroll_px, sleep_s=sleep_s)
    print(f"TOTAL DOCUMENTS (links): {len(df_links)}")

    df_out = fetch_publication_dates(df_links)
    print(df_out.head())

    df_out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
