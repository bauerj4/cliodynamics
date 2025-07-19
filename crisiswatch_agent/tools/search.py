from ..rag.embeddings import embed_text, build_faiss_index
import numpy as np
import sqlite3
from typing import List
from smolagents import tool


@tool
def search_reports_rag(query: str, top_k: int = 5) -> List[str]:
    """
    Semantic RAG search for reports.

    Parameters
    ----------
    query : str
      The description of the documents to retrieve
    top_k : int
      The number of documents to retreive.

    Returns
    -------
    List[str]
    """
    vec = embed_text(query)
    index = build_faiss_index()

    if index.ntotal == 0:
        return ["Index is empty. Run fetch_crisiswatch_data first."]

    scores, ids = index.search(np.expand_dims(vec, axis=0), top_k)
    ids = ids[0]

    conn = sqlite3.connect("crisiswatch.db")
    cur = conn.cursor()
    results = []
    for report_id in ids:
        cur.execute("SELECT url FROM reports WHERE id = ?", (int(report_id),))
        row = cur.fetchone()
        if row:
            results.append(row[0])
    conn.close()
    return results
