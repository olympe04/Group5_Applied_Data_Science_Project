# 02_scrape_statements.py
# Collect ECB press conference transcripts
# Baseline analysis: prepared monetary policy statements
# Extension: Q&A section identified via standard ECB transition sentence

import os
import re
from datetime import datetime

import requests
import pandas as pd
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0"}


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def clean_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def parse_date_english(text: str):
    """
    Fallback regex for dates like '22 July 2021'
    """
    m = re.search(
        r"\b(\d{1,2})\s+"
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+"
        r"(\d{4})\b",
        text,
    )
    if not m:
        return None
    return datetime.strptime(m.group(0), "%d %B %Y").date()


def extract_date(soup: BeautifulSoup):
    """
    Robust extraction of press conference date.
    Returns ISO string YYYY-MM-DD or None.
    """

    # 1) Meta tags (preferred)
    for prop in [
        "article:published_time",
        "og:published_time",
        "article:modified_time",
        "og:updated_time",
    ]:
        m = soup.find("meta", attrs={"property": prop})
        if m and m.get("content"):
            return m["content"][:10]

    for name in ["date", "pubdate", "publish_date", "DC.date", "DC.Date", "dcterms.date"]:
        m = soup.find("meta", attrs={"name": name})
        if m and m.get("content"):
            return m["content"][:10]

    # 2) <time datetime="...">
    for t in soup.find_all("time"):
        if t.get("datetime"):
            dt = t["datetime"]
            if re.match(r"^\d{4}-\d{2}-\d{2}", dt):
                return dt[:10]

    # 3) Fallback: date near title
    h1 = soup.find("h1")
    if h1:
        container = h1.find_parent() or h1
        d = parse_date_english(container.get_text(" ", strip=True))
        if d:
            return d.isoformat()

    return None


# --------------------------------------------------
# Statement / Q&A extraction
# --------------------------------------------------

def extract_statement_and_qa(soup: BeautifulSoup):
    """
    Split prepared statement (baseline) and Q&A (extension).

    Q&A is identified using the standard ECB transition sentence:
    'We are now ready to take your questions' (and close variants).
    """

    main = soup.find("main") or soup.body
    if not main:
        return "", ""

    blocks = main.find_all(["p", "h2", "h3", "li"])
    full_text = clean_whitespace(
        " ".join(b.get_text(" ", strip=True) for b in blocks)
    )

    if not full_text:
        return "", ""

    text_lower = full_text.lower()

    # ECB-specific Q&A opening sentences
    qa_markers = [
        "we are now ready to take your questions",
        "i will now take your questions",
        "let me now take your questions",
    ]

    cut = -1
    for marker in qa_markers:
        idx = text_lower.find(marker)
        if idx != -1:
            cut = idx
            break

    if cut == -1:
        # No detectable Q&A section
        return full_text, ""

    statement = full_text[:cut].strip()
    qa = full_text[cut:].strip()

    return statement, qa


# --------------------------------------------------
# Scraping logic
# --------------------------------------------------

def scrape_one(url: str):
    r = requests.get(url, timeout=30, headers=HEADERS)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    title = soup.find("h1").get_text(" ", strip=True) if soup.find("h1") else ""
    date_iso = extract_date(soup)

    statement_text, qa_text = extract_statement_and_qa(soup)

    return {
        "date": date_iso,
        "title": title,
        "url": url,
        "statement_text": statement_text,  # baseline
        "qa_text": qa_text,                # extension
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    print("=== START 02_scrape_statements ===")

    urls_path = "data_raw/statement_urls.csv"
    urls = pd.read_csv(urls_path)["url"].dropna().tolist()
    print(f"Loaded {len(urls)} transcript URLs")

    rows = []
    for i, url in enumerate(urls, 1):
        try:
            row = scrape_one(url)
            rows.append(row)

            print(
                f"[{i}/{len(urls)}] OK  "
                f"date={row['date']}  "
                f"statement_len={len(row['statement_text'])}  "
                f"qa_len={len(row['qa_text'])}",
                flush=True,
            )

        except Exception as e:
            print(f"[{i}/{len(urls)}] FAIL {url} -> {e}", flush=True)

    df = pd.DataFrame(rows)

    os.makedirs("data_raw", exist_ok=True)
    out_path = "data_raw/ecb_statements_raw.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} rows to {out_path}")
