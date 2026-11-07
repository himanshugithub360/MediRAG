"""Download openly available medical web content into Data/web/ for store_index.py.

- MedlinePlus health topics (U.S. National Library of Medicine): one official XML file
  with ~1,000 English topic summaries. https://medlineplus.gov/xml.html
- WHO fact sheets (World Health Organization): ~240 pages from
  https://www.who.int/news-room/fact-sheets, fetched politely (1 request/second).
  WHO content is licensed CC BY-NC-SA 3.0 IGO: non-commercial use with attribution.

Usage:
    python fetch_sources.py            # both
    python fetch_sources.py medlineplus
    python fetch_sources.py who
Then rebuild the index:  python store_index.py
"""
import html
import json
import os
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

from src.config import DATA_DIR

WEB_DIR = os.path.join(DATA_DIR, "web")
USER_AGENT = "MediRAG-indexer/1.0 (educational medical chatbot project)"


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def html_to_text(fragment):
    fragment = re.sub(r"(?is)<(script|style|noscript|svg|figure|button)[^>]*>.*?</\1>", " ", fragment)
    fragment = re.sub(r"(?i)<li[^>]*>", "\n- ", fragment)
    fragment = re.sub(r"(?i)<(h[1-6])[^>]*>", "\n\n## ", fragment)
    fragment = re.sub(r"(?i)</?(p|div|br|ul|ol|h[1-6]|section|tr|table)[^>]*>", "\n", fragment)
    text = html.unescape(re.sub(r"<[^>]+>", " ", fragment))
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} documents to {path}")


# ── MedlinePlus ──────────────────────────────────────────────────────────────
def fetch_medlineplus():
    print("Finding the latest MedlinePlus health topics file...")
    index = fetch("https://medlineplus.gov/xml.html")
    links = re.findall(r'href="(https://medlineplus\.gov/xml/mplus_topics_\d{4}-\d{2}-\d{2}\.xml)"', index)
    if not links:
        raise SystemExit("Couldn't find the MedlinePlus XML link on https://medlineplus.gov/xml.html")
    url = sorted(links)[-1]
    print(f"Downloading {url} (~30 MB)...")
    root = ET.fromstring(fetch(url, timeout=300).encode("utf-8"))

    rows = []
    for topic in root.iter("health-topic"):
        if topic.get("language") != "English":
            continue
        summary = topic.findtext("full-summary") or ""
        text = html_to_text(summary)
        if len(text) < 200:
            continue
        also = [a.text for a in topic.findall("also-called") if a.text]
        groups = [g.text for g in topic.findall("group") if g.text]
        title = topic.get("title")
        header = title + (f" (also called: {', '.join(also)})" if also else "")
        rows.append({
            "title": title,
            "url": topic.get("url"),
            "topic": groups[0] if groups else "",
            "text": f"{header}\n\n{text}",
        })
    write_jsonl(os.path.join(WEB_DIR, "medlineplus.jsonl"), rows)


# ── WHO fact sheets ──────────────────────────────────────────────────────────
def fetch_who(delay=1.0):
    print("Listing WHO fact sheets...")
    listing = fetch("https://www.who.int/news-room/fact-sheets")
    # Slugs can contain brackets, e.g. "ambient-(outdoor)-air-quality-and-health"
    slugs = sorted(set(html.unescape(s) for s in
                       re.findall(r'/news-room/fact-sheets/detail/([^"\'#?\s<>]+)', listing)))
    print(f"Found {len(slugs)} fact sheets; downloading (about {len(slugs) * delay / 60:.0f} minutes)...")

    rows = []
    for i, slug in enumerate(slugs, 1):
        url = f"https://www.who.int/news-room/fact-sheets/detail/{slug}"
        try:
            page = fetch(url)
        except Exception as e:
            print(f"  skip {slug}: {e}")
            continue
        finally:
            time.sleep(delay)
        title_match = re.search(r'<meta property="og:title" content="([^"]+)"', page)
        article = re.search(r"(?is)<article[^>]*>(.*?)</article>", page)
        if not article:
            continue
        text = html_to_text(article.group(1))
        if len(text) < 300:
            continue
        title = html.unescape(title_match.group(1)).strip() if title_match else slug.replace("-", " ").title()
        rows.append({"title": title, "url": url, "topic": "", "text": f"{title}\n\n{text}"})
        if i % 25 == 0:
            print(f"  {i}/{len(slugs)}")
    write_jsonl(os.path.join(WEB_DIR, "who_factsheets.jsonl"), rows)


if __name__ == "__main__":
    which = sys.argv[1:] or ["medlineplus", "who"]
    if "medlineplus" in which:
        fetch_medlineplus()
    if "who" in which:
        fetch_who()
