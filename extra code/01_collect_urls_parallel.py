# 01_collect_urls_parallel.py
# Alternative scraping approach (fallback): use this version if the Selenium-based scraper fails.
# Collect ECB monetary-policy statement URLs by probing daily URL patterns in parallel.
# I/O:
#   Inputs: hard-coded date range (START, END) and concurrency params (MAX_WORKERS, TIMEOUT).
#   Outputs: CSV file "data_raw/statement_urls.csv" with columns [date, url].

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

# One requests.Session per thread (connection reuse + thread safety).
_tls = threading.local()


def url_for(d: date) -> str:
    """Build the expected ECB statement URL for date d."""
    yy = d.year % 100
    return (
        f"{BASE_URL}/press/press_conference/monetary-policy-statement/"
        f"{d.year}/html/is{yy:02d}{d.month:02d}{d.day:02d}.en.html"
    )


def get_session() -> requests.Session:
    """Return a thread-local Session configured with retries."""
    s = getattr(_tls, "session", None)
    if s is not None:
        return s

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

    _tls.session = s
    return s


def exists(d: date) -> tuple[bool, str, str]:
    """Check whether the statement page for date d exists."""
    url = url_for(d)
    try:
        with get_session().get(url, timeout=TIMEOUT, allow_redirects=True, stream=True) as r:
            ok = (r.status_code == 200 and "press_conference/monetary-policy-statement" in r.url)
            return ok, d.isoformat(), url
    except requests.RequestException:
        return False, d.isoformat(), url


def month_end(d: date) -> date:
    """Return last day of the month containing d."""
    nxt = date(d.year + (d.month // 12), (d.month % 12) + 1, 1)
    return nxt - timedelta(days=1)


def daterange(a: date, b: date):
    """Yield dates from a to b (inclusive)."""
    while a <= b:
        yield a
        a += timedelta(days=1)


def main() -> None:
    """Run parallel checks and write results to disk."""
    scripts_dir = Path(__file__).resolve().parent   # .../extra_code
    project_root = scripts_dir.parent               # .../ (repo root)

    out_dir = project_root / "data_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "statement_urls.csv"

    hits = []
    checked = 0
    cur = START

    # Submit jobs month by month to limit outstanding futures.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        while cur <= END:
            me = min(month_end(cur), END)
            futures = [ex.submit(exists, d) for d in daterange(cur, me)]

            for fut in as_completed(futures):
                ok, d_iso, url = fut.result()
                checked += 1
                if ok:
                    hits.append({"date": d_iso, "url": url})
                    print("HIT", d_iso, url)

            cur = me + timedelta(days=1)

    df = (
        pd.DataFrame(hits)
        .drop_duplicates(subset=["url"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    print(f"TOTAL DOCUMENTS (urls): {len(df)}")
    print(df.head())

    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path}")

    print(f"\nChecked days: {checked}")
    if len(df):
        print("Date min:", df["date"].iloc[0], "Date max:", df["date"].iloc[-1])
        print("Example:", df["url"].iloc[0])


if __name__ == "__main__":
    main()
