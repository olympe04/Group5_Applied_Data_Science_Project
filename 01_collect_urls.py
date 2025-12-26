import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin

HEADERS = {"User-Agent": "Mozilla/5.0"}

# 1) Index facile (comme avant) : liste les visual statements (par date)
INDEX_URL = "https://www.ecb.europa.eu/press/press_conference/visual-mps/html/index.en.html"
BASE = "https://www.ecb.europa.eu"

def get_visual_urls():
    r = requests.get(INDEX_URL, timeout=30, headers=HEADERS)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    links = []
    for a in soup.select("a[href]"):
        href = a["href"]
        if "/press/press_conference/visual-mps/" in href and href.endswith(".en.html"):
            links.append(urljoin(INDEX_URL, href))

    # dédoublonnage en gardant l'ordre
    return list(dict.fromkeys(links))

def get_transcript_url_from_visual(visual_url: str):
    r = requests.get(visual_url, timeout=30, headers=HEADERS)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    for a in soup.select('a[href*="/press/press_conference/monetary-policy-statement/"]'):
        full = urljoin(BASE, a["href"])

        # ✅ On veut une page "ecb.is..." (réunion spécifique)
        if "ecb.is" in full and full.endswith(".en.html"):
            return full

    return None

if __name__ == "__main__":
    os.makedirs("data_raw", exist_ok=True)

    visual_urls = get_visual_urls()
    print(f"Found {len(visual_urls)} visual-mps URLs")

    transcript_urls = []
    missing = 0

    for i, v in enumerate(visual_urls, 1):
        try:
            t = get_transcript_url_from_visual(v)
            if t:
                transcript_urls.append(t)
            else:
                missing += 1
        except Exception as e:
            missing += 1
            print(f"[{i}/{len(visual_urls)}] FAIL: {v} -> {e}")

    # dédoublonnage
    transcript_urls = sorted(set(transcript_urls))

    # On sauvegarde les deux : utile pour debug / transparence
    pd.DataFrame({"url": visual_urls}).to_csv("data_raw/visual_urls.csv", index=False)
    pd.DataFrame({"url": transcript_urls}).to_csv("data_raw/statement_urls.csv", index=False)

    print(f"Saved {len(transcript_urls)} transcript URLs to data_raw/statement_urls.csv")
    print(f"Missing transcript for {missing} visual pages (not necessarily a problem).")
    if transcript_urls:
        print("Example transcript URL:", transcript_urls[0])
