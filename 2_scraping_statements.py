# 02_scrape_statements.py
# Collect ECB press conference transcripts (statement + Q&A)
# Output CSV columns: date, title, subtitle, url, method, statement_text, qa_text, error

import os
import re
from datetime import datetime
from typing import List, Optional, Tuple, Dict
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HEADERS = {"User-Agent": "Mozilla/5.0"}

BASE_DIR = Path(__file__).resolve().parent
DATA_RAW = BASE_DIR / "data_raw"

DEFAULT_INPUT_CANDIDATES = [
    DATA_RAW / "statement_urls.csv",
    DATA_RAW / "ecb_transcript_urls.csv",
    DATA_RAW / "ecb_transcripts_urls.csv",
    DATA_RAW / "ecb_speech_urls.csv",
    DATA_RAW / "ecb_speech_urls.txt",
]

DEFAULT_OUTPUT_PATH = DATA_RAW / "ecb_statements_raw.csv"


def clean_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


MONTHS_RE = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
DATE_EN_RE = re.compile(rf"\b\d{{1,2}}\s+{MONTHS_RE}\s+\d{{4}}\b", re.IGNORECASE)


def parse_date_english(text: str):
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
    if container is None:
        return False
    for p in node.parents:
        if p is container:
            return True
    return False


def _should_skip(node: Tag) -> bool:
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
    end = min(len(texts), start_idx + window)
    for j in range(start_idx + 1, end):
        t = texts[j]
        if QUESTION_RE.search(t) or ON_QUESTION_RE.search(t) or SPEAKER_COLON_RE.search(t):
            return True
    return False


def extract_statement_and_qa(soup: BeautifulSoup, subtitle_node: Optional[Tag] = None) -> Tuple[str, str, str]:
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
    t = strip_qa_leadin_from_statement(text)
    t = strip_statement_heading(t)
    t = strip_start_boilerplate(t)
    t = strip_statement_trailing_q_invite(t)
    return t


def clean_qa_text(text: str) -> str:
    t = strip_extra_leadin_lines(text)
    t = strip_qa_preamble(t)
    return t


def build_session() -> requests.Session:
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


def read_urls(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input introuvable: {path}")

    try:
        df = pd.read_csv(path)
        for col in ["url", "URL", "link", "href"]:
            if col in df.columns:
                return df[col].dropna().astype(str).str.strip().tolist()
        return df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    except Exception:
        pass

    urls: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            low = ln.lower()
            if low in {"date,url", "url", "date"}:
                continue
            if ln.startswith(("http://", "https://")):
                urls.append(ln)
                continue
            if "," in ln:
                parts = [p.strip() for p in ln.split(",", 1)]
                if len(parts) == 2 and parts[1].startswith(("http://", "https://")):
                    urls.append(parts[1])
                    continue
            urls.append(ln)
    return urls


def find_default_input() -> str:
    for p in DEFAULT_INPUT_CANDIDATES:
        if Path(p).exists():
            return str(p)
    raise FileNotFoundError(
        "Aucun fichier d'URLs trouvé. Cherché: " + ", ".join(str(p) for p in DEFAULT_INPUT_CANDIDATES)
    )


def scrape_one(session: requests.Session, url: str) -> Dict[str, str]:
    r = session.get(url, timeout=(10, 40), headers=HEADERS)
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


if __name__ == "__main__":
    in_path = find_default_input()
    print(f"Input: {in_path}")

    urls = read_urls(in_path)
    print(f"Loaded {len(urls)} transcript URLs")

    session = build_session()

    rows: List[Dict[str, str]] = []
    for i, url in enumerate(urls, 1):
        try:
            row = scrape_one(session, url)
            rows.append(row)
            print(
                f"[{i}/{len(urls)}] OK  "
                f"date={row['date'] or 'NA'}  "
                f"method={row['method']}  "
                f"title_len={len(row['title'])}  "
                f"subtitle_len={len(row['subtitle'])}  "
                f"statement_len={len(row['statement_text'])}  "
                f"qa_len={len(row['qa_text'])}",
                flush=True,
            )
        except Exception as e:
            err = str(e)
            rows.append(
                {
                    "date": "",
                    "title": "",
                    "subtitle": "",
                    "url": url,
                    "method": "fail",
                    "statement_text": "",
                    "qa_text": "",
                    "error": err,
                }
            )
            print(f"[{i}/{len(urls)}] FAIL {url} -> {err}", flush=True)

    df = pd.DataFrame(rows)

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out_path = str(DEFAULT_OUTPUT_PATH)
    df.to_csv(out_path, index=False)

    non_empty_qa = int((df["qa_text"].fillna("").str.len() > 0).sum()) if len(df) else 0
    failed = int((df["method"] == "fail").sum()) if len(df) else 0

    print(f"Saved {len(df)} rows to {out_path}")
    print(f"Rows with non-empty Q&A: {non_empty_qa}/{len(df)}")
    print(f"Failed rows kept in CSV: {failed}")
