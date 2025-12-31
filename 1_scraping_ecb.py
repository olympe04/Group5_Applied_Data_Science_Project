from selenium import webdriver
import time
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import pandas as pd

driver = webdriver.Firefox()
try:
    driver.get("https://www.ecb.europa.eu/press/pressconf/html/index.en.html")

    # Scroll down to trigger lazy-loading of older entries
    for _ in range(200):
        driver.execute_script("window.scrollBy(0, 400);")
        time.sleep(0.3)

    # Parse the fully rendered page source
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    print(driver.title)

    # Extract all anchor links from the page
    links = sorted({a["href"] for a in soup.find_all("a", href=True)})
    df = pd.DataFrame(links, columns=["link"])

    # Keep only monetary policy statement pages (English HTML)
    df = df.loc[
        df["link"].str.contains("press_conference/monetary-policy", na=False)
        & df["link"].str.contains(r"\.en\.html$", na=False)
    ].reset_index(drop=True)

    print(f"TOTAL DOCUMENTS : {len(df)}")

finally:
    # Close the browser session
    driver.quit()

# Retrieve publication dates only
session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

dates = []

# Fetch each statement page and extract its published date
for _, row in tqdm(df.iterrows(), total=len(df)):
    url = "https://www.ecb.europa.eu" + row["link"]
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        page = BeautifulSoup(resp.content, "html.parser")

        # Read the Open Graph publication timestamp if available
        meta = page.find("meta", attrs={"property": "article:published_time"})
        dates.append(meta["content"] if meta and meta.get("content") else None)
    except requests.exceptions.RequestException:
        # Record missing date if the request fails
        dates.append(None)

df["date"] = dates

# Build absolute URLs
df["url"] = "https://www.ecb.europa.eu" + df["link"]

# Remove the section index page (not a statement)
df = df[
    df["url"] != "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/html/index.en.html"
].reset_index(drop=True)

# Normalize to YYYY-MM-DD
df["date"] = df["date"].astype(str).str.slice(0, 10)

print(df[["date", "url"]].head())

# Export as CSV (date,url) with header row
import os
os.makedirs("data_raw", exist_ok=True)

os.makedirs("data_raw", exist_ok=True)
df[["date","url"]].to_csv("data_raw/ecb_speech_urls.txt", index=False, header=True, encoding="utf-8")

