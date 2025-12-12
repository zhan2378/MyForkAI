import numpy as np
import json

def hybrid_retrieve(db, oa, dialog_id, query, k_sem=10, k_fts=10, k_final=5):
    q_emb = oa.embed([query])[0]
    qv = np.asarray(q_emb, dtype=np.float32)
    qn = float(np.linalg.norm(qv) + 1e-12)

    # semantic shortlist (scan dialog memory)
    sem_rows = db.conn.execute(
        "SELECT memory_id, text, embedding, norm, meta_json FROM memory WHERE dialog_id=?",
        (dialog_id,)
    ).fetchall()

    sem_scored = []
    for mid, text, blob, norm, meta_json in sem_rows:
        v = np.frombuffer(blob, dtype=np.float32)
        sim = float(np.dot(qv, v) / (qn * norm))
        sem_scored.append((sim, mid))
    sem_scored.sort(reverse=True, key=lambda x: x[0])
    sem_ids = [mid for _, mid in sem_scored[:k_sem]]

    # keyword shortlist via FTS5 (dialog constrained)
    fts_rows = db.conn.execute(
        """
        SELECT f.rowid
        FROM memory_fts f
        JOIN memory m ON m.memory_id = f.rowid
        WHERE f MATCH ? AND m.dialog_id = ?
        LIMIT ?
        """,
        (query, dialog_id, k_fts)
    ).fetchall()
    fts_ids = [int(r[0]) for r in fts_rows]

    # union + rerank by cosine
    union = list(dict.fromkeys(sem_ids + fts_ids))
    if not union:
        return []

    qmarks = ",".join(["?"] * len(union))
    rows = db.conn.execute(
        f"SELECT memory_id, text, embedding, norm, meta_json FROM memory WHERE memory_id IN ({qmarks})",
        union
    ).fetchall()

    rescored = []
    for mid, text, blob, norm, meta_json in rows:
        v = np.frombuffer(blob, dtype=np.float32)
        sim = float(np.dot(qv, v) / (qn * norm))
        rescored.append((sim, mid, text, json.loads(meta_json)))
    rescored.sort(reverse=True, key=lambda x: x[0])

    return rescored[:k_final]