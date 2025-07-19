from crisiswatch_agent.rag.embeddings import (
    embed_text,
    build_faiss_index,
    update_embeddings,
)
import numpy as np
import sqlite3
from typing import List
from smolagents import tool


@tool
def search_reports_rag(
    query: str, top_k: int = 5, db_path: str = "crisiswatch.db"
) -> List[str]:
    """
    description: Searches cached reports using a RAG-based approach by embedding the query and retrieving similar documents.

    Args:
        query: A natural language query to match relevant CrisisWatch reports.
        top_k: The number of top-matching reports to return.
        db_path: The path to the database in which to cache results.

    Returns:
        A list of matching report IDs ranked by relevance to the query.
    """
    vec = embed_text(query)
    update_embeddings(db_path=db_path)
    index = build_faiss_index(db_path=db_path)
    import pdb

    pdb.set_trace()
    if index.ntotal == 0:
        return ["Index is empty. Run fetch_crisiswatch_data first."]

    scores, ids = index.search(np.expand_dims(vec, axis=0), top_k)
    ids = ids[0]

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    results = []
    for report_id in ids:
        cur.execute("SELECT url FROM reports WHERE id = ?", (int(report_id),))
        row = cur.fetchone()
        if row:
            results.append(row[0])
    conn.close()
    return results
