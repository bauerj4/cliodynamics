import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> np.ndarray:
    return np.array(
        embedding_model.encode(text, normalize_embeddings=True), dtype="float32"
    )


def update_embeddings(db_path: str = "crisiswatch.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, summary FROM reports
        WHERE id NOT IN (SELECT report_id FROM embeddings)
    """
    )
    rows = cur.fetchall()
    for rid, summary in rows:
        vec = embed_text(summary)
        cur.execute(
            "INSERT INTO embeddings (report_id, embedding) VALUES (?, ?)",
            (rid, vec.tobytes()),
        )
    conn.commit()
    conn.close()


def build_faiss_index(db_path: str = "crisiswatch.db") -> faiss.IndexFlatIP:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT report_id, embedding FROM embeddings")
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return faiss.IndexFlatIP(384)

    ids, vectors = zip(
        *[(rid, np.frombuffer(blob, dtype="float32")) for rid, blob in rows]
    )
    index = faiss.IndexIDMap(faiss.IndexFlatIP(len(vectors[0])))
    index.add_with_ids(np.stack(vectors), np.array(ids))
    return index
