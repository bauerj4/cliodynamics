import sqlite3
import requests
import re
import sqlite3
import fitz  # PyMuPDF
import requests

from bs4 import BeautifulSoup
from ..rag.embeddings import embed_text, update_embeddings
from smolagents import tool
from datetime import datetime
from typing import List, Optional
from tqdm import tqdm


def init_db(db_path: str = "crisiswatch.db"):
    """Initializes SQLite DB and creates tables."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            title TEXT,
            url TEXT,
            text TEXT,
            region TEXT,
            summary TEXT
        )
    """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            report_id INTEGER PRIMARY KEY,
            embedding BLOB,
            FOREIGN KEY(report_id) REFERENCES reports(id)
        )
    """
    )
    conn.commit()
    conn.close()


@tool
def prepopulate_from_urls(
    urls: List[str], db_path: str = "crisiswatch.db", overwrite: bool = False
) -> str:
    """
    Downloads PDF reports from specified URLs and stores their text in the local CrisisWatch database.

    Description: Fetches and caches CrisisWatch PDF reports from August 2003 to present.

    Args:
        db_path : Path to the SQLite database. Default is "crisiswatch.db".
        urls : list of urls to populate from
        overwrite : (OPTIONAL) If True, existing entries for a report will be overwritten.

    Returns:
        download_description : The description of the operations performed.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            title TEXT,
            url TEXT,
            text TEXT,
            region TEXT,
            summary TEXT
        )
    """
    )

    added = 0
    skipped = 0

    for url in tqdm(urls, desc="Article No."):
        cursor.execute("SELECT id FROM reports WHERE url = ?", (url,))
        if cursor.fetchone() and not overwrite:
            skipped += 1
            continue

        try:
            response = requests.get(url, timeout=15)
            if response.status_code != 200:
                continue

            doc = fitz.open(stream=response.content, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
            doc.close()

            # Try to parse the date from the filename
            match = re.search(r"crisiswatch-(\w+)-(\d{4})", url)
            if match:
                month_str, year = match.groups()
                month_num = datetime.strptime(month_str[:3], "%b").month
                date_str = f"{int(year):04d}-{month_num:02d}-01"
                title = f"CrisisWatch {month_str.capitalize()} {year}"
            else:
                # date_str = datetime.today().strftime("%Y-%m-%d")
                date_str = "2018-01-01"
                title = "CrisisWatch Report"

            cursor.execute(
                "INSERT OR REPLACE INTO reports (date, title, url, text) VALUES (?, ?, ?, ?)",
                (date_str, title, url, text),
            )
            added += 1

        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            continue

        conn.commit()
    conn.close()
    update_embeddings(db_path=db_path)

    return f"Added {added} reports, skipped {skipped}."
