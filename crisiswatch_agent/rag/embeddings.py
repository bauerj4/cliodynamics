import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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
        SELECT id, text FROM reports
        WHERE id NOT IN (SELECT report_id FROM embeddings)
    """
    )
    rows = cur.fetchall()
    for rid, text in tqdm(rows, desc="embedding no."):
        vec = embed_text(text)
        print(vec)
        cur.execute(
            "INSERT INTO embeddings (report_id, embedding) VALUES (?, ?)",
            (rid, vec.tobytes()),
        )
    conn.commit()
    embs = cur.execute("SELECT * FROM embeddings;")
    print(embs.fetchall())
    conn.close()


def build_faiss_index(db_path: str = "crisiswatch.db") -> faiss.IndexFlatIP:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT report_id, embedding FROM embeddings")
    rows = cur.fetchall()
    import pdb

    pdb.set_trace()

    conn.close()

    if not rows:
        return faiss.IndexFlatIP(384)

    ids, vectors = zip(
        *[(rid, np.frombuffer(blob, dtype="float32")) for rid, blob in rows]
    )
    index = faiss.IndexIDMap(faiss.IndexFlatIP(len(vectors[0])))
    index.add_with_ids(np.stack(vectors), np.array(ids))
    return index
