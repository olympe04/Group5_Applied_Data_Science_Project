# 01_collect_urls_parallel.py
# Alternative scraping approach (fallback): use this version if the Selenium-based scraper fails.
# Collect ECB monetary-policy statement URLs by probing daily URL patterns in parallel.
# I/O:
#   Inputs : hard-coded date range (START, END) and concurrency params (MAX_WORKERS, TIMEOUT)
#   Outputs: data_raw/ecb_speech_urls.csv with columns [date, url]
# Notes:
#   The script generates the expected daily statement URL, checks existence in parallel (with retries),
#   keeps only successful hits, and saves a deduplicated, date-sorted URL list.

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://www.ecb.europa.eu"
UA = {"User-Agent": "Mozilla/5.0"}

START = date(1999, 1, 1)
END = date(2025, 12, 31)

MAX_WORKERS = 6   # higher values may trigger 429/503
TIMEOUT = 25

_tls = threading.local()  # thread-local storage for per-thread requests.Session


def get_project_root() -> Path:
    """Return repository root (script is in extra_code/)."""
    scripts_dir = Path(__file__).resolve().parent
    return scripts_dir.parent


def resolve_output_path(project_root: Path) -> Path:
    """Return the output CSV path (data_raw/ecb_speech_urls.csv) and ensure the directory exists."""
    out_dir = project_root / "data_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "ecb_speech_urls.csv"


def url_for(d: date) -> str:
    """Build the expected ECB statement URL for date d."""
    yy = d.year % 100
    return (
        f"{BASE_URL}/press/press_conference/monetary-policy-statement/"
        f"{d.year}/html/is{yy:02d}{d.month:02d}{d.day:02d}.en.html"
    )


def build_session() -> requests.Session:
    """Create a requests Session configured with retries and a fixed User-Agent."""
    s = requests.Session()
    s.headers.update(UA)

    retry = Retry(
        total=4,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def get_session() -> requests.Session:
    """Return a thread-local Session (connection reuse + thread safety)."""
    s = getattr(_tls, "session", None)
    if s is None:
        s = build_session()
        _tls.session = s
    return s


def exists(d: date) -> tuple[bool, str, str]:
    """Check whether the statement page for date d exists and returns (ok, date_iso, url)."""
    url = url_for(d)
    try:
        with get_session().get(url, timeout=TIMEOUT, allow_redirects=True, stream=True) as r:
            ok = (r.status_code == 200 and "press_conference/monetary-policy-statement" in r.url)
            return ok, d.isoformat(), url
    except requests.RequestException:
        return False, d.isoformat(), url


def month_end(d: date) -> date:
    """Return the last day of the month containing d."""
    nxt = date(d.year + (d.month // 12), (d.month % 12) + 1, 1)
    return nxt - timedelta(days=1)


def daterange(a: date, b: date):
    """Yield dates from a to b (inclusive)."""
    while a <= b:
        yield a
        a += timedelta(days=1)


def collect_hits(start: date, end: date, max_workers: int) -> tuple[list[dict], int]:
    """Probe URLs in parallel month-by-month and return (hits, checked_count)."""
    hits: list[dict] = []
    checked = 0
    cur = start

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        while cur <= end:
            me = min(month_end(cur), end)
            futures = [ex.submit(exists, d) for d in daterange(cur, me)]

            for fut in as_completed(futures):
                ok, d_iso, url = fut.result()
                checked += 1
                if ok:
                    hits.append({"date": d_iso, "url": url})
                    print("HIT", d_iso, url)

            cur = me + timedelta(days=1)

    return hits, checked


def finalize(hits: list[dict]) -> pd.DataFrame:
    """Deduplicate, sort by date, and return the final URL list dataframe."""
    return (
        pd.DataFrame(hits)
        .drop_duplicates(subset=["url"])
        .sort_values("date")
        .reset_index(drop=True)
    )


def save_csv(df: pd.DataFrame, out_path: Path) -> None:
    """Write the URL list dataframe to CSV."""
    df.to_csv(out_path, index=False, encoding="utf-8")


def main() -> None:
    """Execute the parallel URL probe and write data_raw/ecb_speech_urls.csv."""
    project_root = get_project_root()
    out_path = resolve_output_path(project_root)

    hits, checked = collect_hits(START, END, MAX_WORKERS)
    df = finalize(hits)

    print(f"TOTAL DOCUMENTS (urls): {len(df)}")
    if len(df):
        print(df.head())
        print("Date min:", df["date"].iloc[0], "Date max:", df["date"].iloc[-1])
        print("Example:", df["url"].iloc[0])

    save_csv(df, out_path)
    print(f"Saved: {out_path}")
    print(f"Checked days: {checked}")


if __name__ == "__main__":
    main()