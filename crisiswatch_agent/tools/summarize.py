from sentence_transformers import SentenceTransformer
import sqlite3
from collections import defaultdict
from typing import Optional
from smolagents import tool

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


@tool
def summarize_reports(region: Optional[str] = None) -> str:
    """
    Summarizes recent crisis reports.

    Parameters
    ----------
    region : str, optional

    Returns
    -------
    str
    """
    conn = sqlite3.connect("crisiswatch.db")
    cur = conn.cursor()

    if region:
        cur.execute("SELECT region, summary FROM reports WHERE region LIKE ?", (f"%{region}%",))
    else:
        cur.execute("SELECT region, summary FROM reports")

    rows = cur.fetchall()
    conn.close()

    region_map = defaultdict(list)
    for reg, summ in rows:
        region_map[reg].append(summ)

    output = []
    for reg, summaries in region_map.items():
        combined = " ".join(summaries)[:3000]
        snippet = embedding_model.tokenizer.decode(
            embedding_model.tokenizer.encode(combined)[:256]
        )
        output.append(f"📍 **{reg}**: {snippet}")

    return "\n\n".join(output)
