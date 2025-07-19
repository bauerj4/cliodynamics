import sqlite3
import requests
from bs4 import BeautifulSoup
from ..rag.embeddings import embed_text, update_embeddings
from smolagents import tool


DB_PATH = "crisiswatch.db"

def init_db():
    """Initializes SQLite DB and creates tables."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY,
            title TEXT,
            url TEXT UNIQUE,
            date TEXT,
            region TEXT,
            countries TEXT,
            summary TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            report_id INTEGER PRIMARY KEY,
            embedding BLOB,
            FOREIGN KEY(report_id) REFERENCES reports(id)
        )
    """)
    conn.commit()
    conn.close()

@tool
def fetch_crisiswatch_data() -> str:
    """
    Fetches new reports from CrisisWatch and stores in DB.

    Returns
    -------
    str
        Message about the number of new entries added.
    """
    url = "https://www.crisisgroup.org/crisiswatch/database"
    response = requests.get(url)
    if not response.ok:
        return "Failed to fetch CrisisWatch page."

    soup = BeautifulSoup(response.text, "html.parser")
    reports = soup.select(".views-row")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    new_entries = 0

    for report in reports:
        try:
            title = report.select_one("h3").get_text(strip=True)
            link = "https://www.crisisgroup.org" + report.select_one("a")["href"]
            if cur.execute("SELECT 1 FROM reports WHERE url = ?", (link,)).fetchone():
                continue

            date = report.select_one(".crisiswatch-date").get_text(strip=True)
            region = report.select_one(".crisiswatch-region").get_text(strip=True)
            countries = report.select_one(".crisiswatch-country").get_text(strip=True)
            summary = report.select_one(".crisiswatch-summary").get_text(strip=True)

            cur.execute("""
                INSERT INTO reports (title, url, date, region, countries, summary)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (title, link, date, region, countries, summary))
            new_entries += 1
        except Exception:
            continue

    conn.commit()
    conn.close()
    update_embeddings()
    return f"Fetched and stored {new_entries} new entries."
