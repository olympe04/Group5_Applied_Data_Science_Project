# 2_scraping_statements.py
# Collect ECB press conference transcripts (statement + Q&A) from a list of URLs.
# I/O:
#   Inputs : data_raw/ecb_speech_urls.csv with column [url] (optionally [date]).
#   Outputs: data_raw/ecb_statements_raw.csv with columns
#            [date, title, subtitle, url, method, statement_text, qa_text, error].
# Notes:
#   The script iterates over URLs, downloads each page, extracts metadata + text, splits statement vs Q&A,
#   cleans boilerplate, and writes one row per URL (keeping failures with an error message).

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


CONFIG = {
    "INPUT_CSV": "data_raw/ecb_speech_urls.csv",
    "OUTPUT_CSV": "data_raw/ecb_statements_raw.csv",
    "USER_AGENT": "Mozilla/5.0",
    "TIMEOUT_CONNECT": 10,
    "TIMEOUT_READ": 40,
    "START_INDEX": 0,
    "MAX_URLS": None,
    "VERBOSE_EVERY": 1,
}


def get_project_root() -> Path:
    """Return repository root (project root is parent of replication/)."""
    scripts_dir = Path(__file__).resolve().parent  # .../replication
    return scripts_dir.parent


def resolve_paths(project_root: Path) -> tuple[Path, Path]:
    """Return (input_csv_path, output_csv_path) for the scraper."""
    in_path = project_root / CONFIG["INPUT_CSV"]
    out_path = project_root / CONFIG["OUTPUT_CSV"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return in_path, out_path


def clean_whitespace(s: str) -> str:
    """Normalize whitespace by collapsing runs of spaces and trimming leading/trailing blanks."""
    return re.sub(r"\s+", " ", str(s)).strip()


MONTHS_RE = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
DATE_EN_RE = re.compile(rf"\b\d{{1,2}}\s+{MONTHS_RE}\s+\d{{4}}\b", re.IGNORECASE)


def parse_date_english(text: str):
    """Parse an English date like '13 March 2020' from free text and return a date object (or None)."""
    m = re.search(
        r"\b(\d{1,2})\s+"
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+"
        r"(\d{4})\b",
        text,
    )
    if not m:
        return None
    return datetime.strptime(m.group(0), "%d %B %Y").date()


def extract_date(soup: BeautifulSoup) -> Optional[str]:
    """Extract an ISO date (YYYY-MM-DD) from meta tags, time tags, or header text within the page."""
    for prop in ["article:published_time", "og:published_time", "article:modified_time", "og:updated_time"]:
        m = soup.find("meta", attrs={"property": prop})
        if m and m.get("content"):
            c = m["content"]
            if re.match(r"^\d{4}-\d{2}-\d{2}", c):
                return c[:10]

    for name in ["date", "pubdate", "publish_date", "DC.date", "DC.Date", "dcterms.date"]:
        m = soup.find("meta", attrs={"name": name})
        if m and m.get("content"):
            c = m["content"]
            if re.match(r"^\d{4}-\d{2}-\d{2}", c):
                return c[:10]

    for t in soup.find_all("time"):
        if t.get("datetime"):
            dt = t["datetime"]
            if re.match(r"^\d{4}-\d{2}-\d{2}", dt):
                return dt[:10]

    h1 = soup.find("h1")
    if h1:
        container = h1.find_parent() or h1
        d = parse_date_english(container.get_text(" ", strip=True))
        if d:
            return d.isoformat()

    return None


def extract_title_subtitle(soup: BeautifulSoup) -> Tuple[str, str, Optional[Tag]]:
    """Extract the main title and the first meaningful subtitle-like sibling after the H1 element."""
    title = ""
    subtitle = ""
    subtitle_node: Optional[Tag] = None

    h1 = soup.find("h1")
    if not h1:
        return title, subtitle, None

    title = clean_whitespace(h1.get_text(" ", strip=True))

    for sib in h1.find_all_next(["h2", "p"], limit=15):
        txt = clean_whitespace(sib.get_text(" ", strip=True))
        if not txt:
            continue
        low = txt.lower()
        if "jump to" in low or "download" in low:
            continue
        subtitle = txt
        subtitle_node = sib
        break

    return title, subtitle, subtitle_node


LEADIN_START_RE = re.compile(
    r"^\s*(with\s+the\s+transcript\s+of\s+the\s+questions|transcript\s+of\s+the\s+questions)\b",
    re.IGNORECASE,
)

STATEMENT_ANCHORS = [
    "Ladies and gentlemen",
    "Good afternoon",
    "Good morning",
    "Good evening",
    "Let me begin",
    "Let me start",
    "I should like",
    "I would like",
    "At today’s meeting",
    "At today's meeting",
    "The Governing Council",
    "We decided",
    "We have decided",
    "Introductory statement",
]


def strip_qa_leadin_from_statement(text: str) -> str:
    """Remove leading Q&A-related boilerplate that sometimes precedes the introductory statement."""
    t = (text or "").lstrip()
    if not t or not LEADIN_START_RE.match(t):
        return t

    lower = t.lower()

    for anchor in STATEMENT_ANCHORS:
        idx = lower.find(anchor.lower())
        if 0 < idx < 3000:
            return t[idx:].lstrip(" .;:-–—,")

    dm = DATE_EN_RE.search(t)
    if dm and dm.end() < 1200:
        return t[dm.end():].lstrip(" .;:-–—,")

    block = t[:600]
    if re.search(r"\b(president|vice[-\s]?president|frankfurt|questions|answers)\b", block, re.IGNORECASE):
        return t[min(len(t), 250):].lstrip(" .;:-–—,")

    return t


STATEMENT_HEADER_RE = re.compile(
    r"^\s*(introductory\s+statement|monetary\s+policy\s+statement)\b",
    re.IGNORECASE,
)

LOCATION_DATE_RE = re.compile(
    rf"^\s*[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ .'\-]{{1,80}},\s*\d{{1,2}}\s+{MONTHS_RE}\s+\d{{4}}\b",
    re.IGNORECASE,
)


def strip_statement_heading(text: str) -> str:
    """Strip common statement headings and location/date lines that are not part of the speech body."""
    t = (text or "").lstrip()
    if not t:
        return t

    low = t.lower()
    if ("introductory statement" in low[:250]) or ("monetary policy statement" in low[:250]):
        for anchor in STATEMENT_ANCHORS:
            idx = low.find(anchor.lower())
            if 0 < idx < 3000:
                return t[idx:].lstrip(" .;:-–—,")

    for _ in range(6):
        changed = False

        m1 = STATEMENT_HEADER_RE.match(t)
        if m1:
            t = t[m1.end():].lstrip(" .;:-–—,")
            changed = True

        m2 = LOCATION_DATE_RE.match(t)
        if m2:
            t = t[m2.end():].lstrip(" .;:-–—,")
            changed = True

        if not changed:
            break

    return t


START_BOILERPLATE_RE = re.compile(
    r"^\s*(?:"
    r"jump\s+to\s+the\s+transcript\s+of\s+the\s+questions\s+and\s+answers\.?"
    r"|click\s+here\s+for\s+the\s+transcript\s+of\s+questions\s+and\s+answers\.?"
    r"|click\s+here\s+for\s+the\s+transcript\s+of\s+the\s+questions\s+and\s+answers\.?"
    r"|with\s+a\s+transcript\s+of\s+the\s+questions\s+and\s+answers\.?"
    r")\s*",
    re.IGNORECASE,
)


def strip_start_boilerplate(text: str) -> str:
    """Remove repeated "jump/click for transcript" boilerplate from the start of extracted text."""
    t = (text or "").lstrip()
    if not t:
        return t

    for _ in range(6):
        m = START_BOILERPLATE_RE.match(t)
        if not m:
            break
        t = t[m.end():].lstrip(" .;:-–—,")
    return t


END_STAR_BLOCK_RE = re.compile(r"(?:\s*(?:\*\s*){2,}\s*)+$")

END_TRANSCRIPT_LINK_RE = re.compile(
    r"\s*(?:"
    r"click\s+here\s+for\s+the\s+transcript\s+of\s+(?:the\s+)?questions\s+and\s+answers\.?"
    r"|jump\s+to\s+the\s+transcript\s+of\s+(?:the\s+)?questions\s+and\s+answers\.?"
    r"|click\s+here\s+for\s+the\s+transcript\s+of\s+questions\s+and\s+answers\.?"
    r")\s*$",
    re.IGNORECASE,
)

END_QUESTION_INVITE_RE = re.compile(
    r"\s*(?:"
    r"we\s+are\s+now\s+at\s+your\s+disposal"
    r"(?:\s*,?\s*should\s+you\s+have\s+any\s+questions)?"
    r"(?:\s+for\s+(?:your\s+)?questions)?"
    r"|we\s+are\s+now\s+ready\s+to\s+take\s+(?:your\s+)?questions"
    r"|i\s+will\s+now\s+take\s+your\s+questions"
    r"|let\s+me\s+now\s+take\s+your\s+questions"
    r"|we\s+stand\s+ready\s+to\s+take\s+any\s+questions\s+you\s+might\s+have"
    r"|we\s+stand\s+ready\s+to\s+answer\s+any\s+questions\s+you\s+may\s+have"
    r"|we\s+are\s+at\s+your\s+disposal\s+for\s+any\s+further\s+questions"
    r"|i\s+am\s+now\s+open\s+to\s+questions"
    r")\s*\.?\s*$",
    re.IGNORECASE,
)


def strip_statement_trailing_q_invite(text: str) -> str:
    """Trim trailing separators, transcript links, and question invitations from the statement tail."""
    t = (text or "").rstrip()
    if not t:
        return t

    for _ in range(8):
        changed = False

        m = END_STAR_BLOCK_RE.search(t)
        if m:
            t = t[:m.start()].rstrip()
            changed = True

        m = END_TRANSCRIPT_LINK_RE.search(t)
        if m:
            t = t[:m.start()].rstrip()
            changed = True

        m = END_QUESTION_INVITE_RE.search(t)
        if m:
            t = t[:m.start()].rstrip()
            changed = True

        if not changed:
            break

    return t


EXTRA_LEADIN_PATTERNS = [
    r"with\s+a\s+transcript\s+of\s+the\s+questions\s+and\s+answers\.?",
    r"click\s+here\s+for\s+the\s+transcript\s+of\s+questions\s+and\s+answers\.?",
    r"jump\s+to\s+the\s+transcript\s+of\s+the\s+questions\s+and\s+answers\.?",
    r"welcome\s+address\s+by\s+jean-claude\s+trichet,\s+governor\s+of\s+the\s+banque\s+de\s+france\.?",
]
EXTRA_LEADIN_RE = re.compile(r"^\s*(?:" + "|".join(EXTRA_LEADIN_PATTERNS) + r")\s*", re.IGNORECASE)


def strip_extra_leadin_lines(text: str) -> str:
    """Drop known non-content lead-in lines that occasionally appear at the start of Q&A sections."""
    t = (text or "").lstrip()
    if not t:
        return t

    for _ in range(6):
        m = EXTRA_LEADIN_RE.match(t)
        if not m:
            break
        t = t[m.end():].lstrip(" \t\r\n-–—•:;.,")
    return t


QA_TRANSCRIPT_START_RE = re.compile(
    r"^\s*transcript\s+of\s+the\s+"
    r"(?:questions\s+asked\s+and\s+the\s+answers\s+given\s+by|questions\s+to\s+and\s+answers\s+of)\b",
    re.IGNORECASE,
)

QA_START_MARKER_RE = re.compile(
    r"(?:\bquestion\b\s*(?:\(|:)|\bq\s*[:.\-]|\bon\s+(?:your|the)\s+question\b)",
    re.IGNORECASE,
)

QA_SENTENCE_MARKER_RE = re.compile(
    r"^\s*(?:\bquestion\b\s*(?:\(|:)\s*|\bq\s*[:.\-]\s*)?"
    r"(?:we\s+are\s+now\s+ready\s+to\s+take\s+your\s+questions|"
    r"we\s+are\s+now\s+at\s+your\s+disposal\s+for\s+questions|"
    r"we\s+are\s+now\s+at\s+your\s+disposal\s+for\s+your\s+questions|"
    r"we\s+stand\s+ready\s+to\s+answer\s+any\s+questions\s+you\s+may\s+have|"
    r"we\s+are\s+at\s+your\s+disposal\s+for\s+any\s+further\s+questions|"
    r"i\s+am\s+now\s+open\s+to\s+questions|"
    r"i\s+will\s+now\s+take\s+your\s+questions|"
    r"let\s+me\s+now\s+take\s+your\s+questions)\.?\s*",
    re.IGNORECASE,
)

STAR_BLOCK_RE = re.compile(r"^\s*(?:\*\s*){2,}")


def strip_qa_preamble(text: str) -> str:
    """Remove Q&A preamble markers and separators so the Q&A starts at the first question/answer turn."""
    t = (text or "").lstrip()
    if not t:
        return t

    for _ in range(8):
        changed = False

        m = STAR_BLOCK_RE.match(t)
        if m:
            t = t[m.end():].lstrip(" \t\r\n-–—•:;.,")
            changed = True

        m = QA_SENTENCE_MARKER_RE.match(t)
        if m:
            t = t[m.end():].lstrip(" \t\r\n-–—•:;.,")
            changed = True

        if QA_TRANSCRIPT_START_RE.match(t):
            m2 = QA_START_MARKER_RE.search(t)
            if m2:
                t = t[m2.start():].lstrip(" \t\r\n-–—•:;.,")
            else:
                t = t[400:].lstrip(" \t\r\n-–—•:;.,")
            changed = True

        if not changed:
            break

    return t


QA_HEADER_STRICT_RE = re.compile(
    r"(transcript\s+of\s+the\s+questions|questions\s+and\s+answers)",
    re.IGNORECASE,
)
QA_HEADER_LOOSE_RE = re.compile(
    r"(transcript\s+of\s+the\s+questions|questions\s+and\s+answers|\bq\s*&\s*a\b|\bq&a\b)",
    re.IGNORECASE,
)

QUESTION_RE = re.compile(r"^\s*(question\b|q\s*[:.\-])", re.IGNORECASE)
ON_QUESTION_RE = re.compile(r"^\s*on\s+(your|the)\s+question\b", re.IGNORECASE)
SPEAKER_COLON_RE = re.compile(r"^\s*[A-ZÀ-ÖØ-Ý][^:]{0,60}:\s")

SENTENCE_MARKERS = [
    "we are now ready to take your questions",
    "i will now take your questions",
    "let me now take your questions",
    "we are now at your disposal for questions",
    "we are now at your disposal for your questions",
]

SKIP_ANCESTOR_TAGS = {"header", "nav", "footer", "aside"}


def _is_inside(node: Tag, container: Optional[Tag]) -> bool:
    """Return True if the node is located within the provided container element."""
    if container is None:
        return False
    return any(p is container for p in node.parents)


def _should_skip(node: Tag) -> bool:
    """Filter out nodes that belong to headers, navigation, footers, cookie banners, or table-of-contents blocks."""
    for p in node.parents:
        name = getattr(p, "name", None)
        if name in SKIP_ANCESTOR_TAGS:
            return True

        pid = (p.get("id") or "").lower()
        if pid in {"feedback", "cookieconsent", "toc", "anchorlinks"}:
            return True

        cls = p.get("class") or []
        cls_low = {c.lower() for c in cls if isinstance(c, str)}
        if "address-box" in cls_low or "ecb-cookieconsent" in cls_low:
            return True
        if (
            "ecb-anchorlinks" in cls_low
            or "anchorlinks" in cls_low
            or "anchor-links" in cls_low
            or "ecb-toc" in cls_low
            or "toc" in cls_low
            or "table-of-contents" in cls_low
        ):
            return True
    return False


def _looks_like_qa_after(texts: List[str], start_idx: int, window: int = 30) -> bool:
    """Return True if Q&A-like markers appear shortly after a candidate Q&A header."""
    end = min(len(texts), start_idx + window)
    for j in range(start_idx + 1, end):
        t = texts[j]
        if QUESTION_RE.search(t) or ON_QUESTION_RE.search(t) or SPEAKER_COLON_RE.search(t):
            return True
    return False


def extract_statement_and_qa(soup: BeautifulSoup, subtitle_node: Optional[Tag] = None) -> Tuple[str, str, str]:
    """Split the page into statement and Q&A text using DOM cues and fallback text-pattern heuristics."""
    main = soup.find("main") or soup.body or soup
    if not main:
        return "", "", "none"

    nodes: List[Tag] = []
    for n in main.find_all(["p", "h2", "h3", "h4", "li"]):
        if subtitle_node is not None and n is subtitle_node:
            continue
        if not _should_skip(n):
            nodes.append(n)

    texts = [clean_whitespace(n.get_text(" ", strip=True)) for n in nodes]
    full_text = clean_whitespace(" ".join(texts))
    if not full_text:
        return "", "", "none"

    qa_container = main.find("div", class_=lambda c: c and "appendix" in c.split())
    if qa_container:
        cut_idx = None
        for i, n in enumerate(nodes):
            if _is_inside(n, qa_container):
                cut_idx = i
                break
        if cut_idx is not None:
            statement = clean_whitespace(" ".join(texts[:cut_idx]))
            qa_parts = [texts[i] for i in range(cut_idx, len(nodes)) if _is_inside(nodes[i], qa_container)]
            qa = clean_whitespace(" ".join(qa_parts))
            return statement, qa, "dom_appendix"

    for i, n in enumerate(nodes):
        if n.name not in {"h2", "h3", "h4"}:
            continue

        hdr = texts[i]
        if not hdr:
            continue

        is_candidate = bool(QA_HEADER_STRICT_RE.search(hdr))
        if not is_candidate and QA_HEADER_LOOSE_RE.search(hdr):
            low = hdr.lower()
            if any(x in low for x in ["president of the ecb", "vice president of the ecb", "member of the ecb"]):
                is_candidate = False
            else:
                if re.fullmatch(r"\s*q\s*&?\s*a\s*\b.*", low) and len(hdr) <= 40:
                    is_candidate = True

        if is_candidate and _looks_like_qa_after(texts, i):
            statement = clean_whitespace(" ".join(texts[:i]))
            qa = clean_whitespace(" ".join(texts[i:]))

            if len(statement) < 200 and i < len(nodes) - 1:
                continue

            return statement, qa, "dom_header_confirmed"

    for i, t in enumerate(texts):
        if QUESTION_RE.search(t) or ON_QUESTION_RE.search(t):
            statement = clean_whitespace(" ".join(texts[:i]))
            qa = clean_whitespace(" ".join(texts[i:]))
            return statement, qa, "pattern_question"

    lower = full_text.lower()
    for marker in SENTENCE_MARKERS:
        idx = lower.find(marker)
        if idx != -1:
            return full_text[:idx].strip(), full_text[idx:].strip(), "sentence_marker"

    return full_text, "", "none"


def clean_statement_text(text: str) -> str:
    """Apply cleanup rules to keep only the speech body of the introductory statement."""
    t = strip_qa_leadin_from_statement(text)
    t = strip_statement_heading(t)
    t = strip_start_boilerplate(t)
    t = strip_statement_trailing_q_invite(t)
    return t


def clean_qa_text(text: str) -> str:
    """Apply cleanup rules to remove non-content boilerplate at the start of the Q&A section."""
    t = strip_extra_leadin_lines(text)
    t = strip_qa_preamble(t)
    return t


def build_session() -> requests.Session:
    """Return a requests session configured with retry/backoff."""
    session = requests.Session()
    retry = Retry(
        total=6,
        backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def read_urls_and_dates(csv_path: Path) -> Tuple[List[str], Dict[str, str]]:
    """Return (urls, url_to_date) loaded from the input CSV (date map is optional)."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input introuvable: {csv_path}")

    df = pd.read_csv(csv_path)
    if "url" not in df.columns:
        raise ValueError("Le CSV doit contenir une colonne 'url' (et idéalement 'date').")

    urls = df["url"].dropna().astype(str).str.strip().tolist()

    date_map: Dict[str, str] = {}
    if "date" in df.columns:
        tmp = df[["url", "date"]].dropna()
        tmp["url"] = tmp["url"].astype(str).str.strip()
        tmp["date"] = tmp["date"].astype(str).str.slice(0, 10)
        date_map = dict(zip(tmp["url"], tmp["date"]))

    return urls, date_map


def scrape_one(session: requests.Session, url: str, headers: Dict[str, str], timeout: Tuple[int, int]) -> Dict[str, str]:
    """Download one page and return extracted fields for the output dataset row."""
    r = session.get(url, timeout=timeout, headers=headers)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    title, subtitle, subtitle_node = extract_title_subtitle(soup)
    date_iso = extract_date(soup)

    statement_text, qa_text, method = extract_statement_and_qa(soup, subtitle_node=subtitle_node)
    statement_text = clean_statement_text(statement_text)
    qa_text = clean_qa_text(qa_text)

    return {
        "date": date_iso or "",
        "title": title,
        "subtitle": subtitle,
        "url": url,
        "method": method,
        "statement_text": statement_text,
        "qa_text": qa_text,
        "error": "",
    }


def iter_urls(urls: List[str], start_index: int, max_urls: Optional[int]) -> List[str]:
    """Return the sliced URL list according to start index and optional max count."""
    u = urls[int(start_index or 0):]
    return u if max_urls is None else u[: int(max_urls)]


def scrape_all(
    urls: List[str],
    date_map: Dict[str, str],
    headers: Dict[str, str],
    timeout: Tuple[int, int],
    verbose_every: int,
) -> List[Dict[str, str]]:
    """Scrape all pages, returning one output row per URL and keeping failures with an error field."""
    session = build_session()
    rows: List[Dict[str, str]] = []
    total = len(urls)

    for i, url in enumerate(urls, 1):
        try:
            row = scrape_one(session, url, headers=headers, timeout=timeout)
            if not row["date"]:
                row["date"] = date_map.get(url, "")
            rows.append(row)

            if verbose_every and (i % int(verbose_every) == 0):
                print(
                    f"[{i}/{total}] OK  date={row['date'] or 'NA'}  method={row['method']}  "
                    f"statement_len={len(row['statement_text'])}  qa_len={len(row['qa_text'])}",
                    flush=True,
                )

        except Exception as e:
            err = str(e)
            rows.append(
                {
                    "date": date_map.get(url, ""),
                    "title": "",
                    "subtitle": "",
                    "url": url,
                    "method": "fail",
                    "statement_text": "",
                    "qa_text": "",
                    "error": err,
                }
            )
            print(f"[{i}/{total}] FAIL {url} -> {err}", flush=True)

    return rows


def write_output(rows: List[Dict[str, str]], output_path: Path) -> pd.DataFrame:
    """Write the final dataset to CSV and return it as a dataframe."""
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return df


def print_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Print basic dataset diagnostics after writing output."""
    non_empty_qa = int((df["qa_text"].fillna("").str.len() > 0).sum()) if len(df) else 0
    failed = int((df["method"] == "fail").sum()) if len(df) else 0
    print(f"Saved {len(df)} rows to {output_path}")
    print(f"Rows with non-empty Q&A: {non_empty_qa}/{len(df)}")
    print(f"Failed rows kept in CSV: {failed}")


def main() -> None:
    """Execute the scraping pipeline and save the final transcripts dataset."""
    project_root = get_project_root()
    input_path, output_path = resolve_paths(project_root)

    headers = {"User-Agent": CONFIG["USER_AGENT"]}
    timeout = (CONFIG["TIMEOUT_CONNECT"], CONFIG["TIMEOUT_READ"])

    print(f"Input : {input_path}")
    urls, date_map = read_urls_and_dates(input_path)

    urls = iter_urls(urls, start_index=int(CONFIG["START_INDEX"] or 0), max_urls=CONFIG["MAX_URLS"])
    print(f"Loaded {len(urls)} transcript URLs (start_index={CONFIG['START_INDEX']})")

    rows = scrape_all(
        urls=urls,
        date_map=date_map,
        headers=headers,
        timeout=timeout,
        verbose_every=int(CONFIG["VERBOSE_EVERY"] or 0),
    )

    df = write_output(rows, output_path)
    print_summary(df, output_path)


if __name__ == "__main__":
    main()