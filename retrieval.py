# retrieval.py
from __future__ import annotations

import json
import numpy as np
from typing import Any, Dict, List, Tuple, Optional


def _to_text(x: Any) -> str:
    return "" if x is None else str(x)


def _cosine(qv: np.ndarray, qn: float, v: np.ndarray, vn: float) -> float:
    if qn <= 0.0 or vn <= 0.0:
        return -1.0
    return float(np.dot(qv, v) / (qn * vn))


def _safe_json_loads(s: Any) -> Dict[str, Any]:
    if not s:
        return {}
    if isinstance(s, (dict, list)):
        # already parsed (unlikely for sqlite)
        return s if isinstance(s, dict) else {"_": s}
    try:
        return json.loads(s)
    except Exception:
        return {}


def hybrid_retrieve(
    db,
    embedder,
    dialog_id: str,
    query: str,
    k_final: int = 8,
    k_fts: int = 60,
    mode: str = "universal",
    embed_provider: Optional[str] = None,
) -> List[Tuple[float, int, str, Dict[str, Any]]]:
    """
    Retrieval modes:

    1) mode="universal" (RECOMMENDED for mixed LLMs):
       - Use SQLite FTS to shortlist candidate memory rows (text only).
       - Embed query + candidate texts using the *current* embedder.
       - Cosine rerank candidates.
       - Works across all providers/models (no mixed embedding space issues).

    2) mode="stored" (optional fast path):
       - Embed query, compare to stored memory.embedding vectors.
       - Only safe if stored vectors are in the same embedding space.
       - Skips mismatched dimensions.

    embed_provider:
      - None => do not filter (equivalent to "any")
      - "openai"/"gemini"/... => filter memories by meta_json["embed_provider"] == embed_provider
    """

    query = (query or "").strip()
    if not query:
        return []

    mode = (mode or "universal").lower()
    if mode not in ("universal", "stored"):
        mode = "universal"

    # -----------------------------
    # 1) FTS shortlist (global across providers)
    # -----------------------------
    fts_rows = db.conn.execute(
        """
        SELECT f.rowid
        FROM memory_fts AS f
        JOIN memory AS m ON m.memory_id = f.rowid
        WHERE memory_fts MATCH ? AND m.dialog_id = ?
        LIMIT ?
        """,
        (query, dialog_id, int(k_fts)),
    ).fetchall()

    fts_ids = [r[0] for r in fts_rows]

    # If FTS returns nothing, we can optionally fallback to a small recent sample
    # so "universal" can still work even when query terms aren't in text.
    if not fts_ids:
        fallback_rows = db.conn.execute(
            """
            SELECT memory_id
            FROM memory
            WHERE dialog_id=?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (dialog_id, int(min(200, k_fts))),
        ).fetchall()
        fts_ids = [r[0] for r in fallback_rows]

    if not fts_ids:
        return []

    # Load candidate texts + meta
    qmarks = ",".join(["?"] * len(fts_ids))
    cand_rows = db.conn.execute(
        f"""
        SELECT memory_id, text, meta_json, embedding, norm
        FROM memory
        WHERE dialog_id=? AND memory_id IN ({qmarks})
        """,
        [dialog_id, *fts_ids],
    ).fetchall()

    # Optional provider filter (by meta_json)
    candidates: List[Tuple[int, str, Dict[str, Any], Any, Any]] = []
    for mid, text, meta_json, emb_blob, norm in cand_rows:
        meta = _safe_json_loads(meta_json)
        if embed_provider and meta.get("embed_provider") != embed_provider:
            continue
        candidates.append((int(mid), _to_text(text), meta, emb_blob, norm))

    if not candidates:
        return []

    # -----------------------------
    # 2) Universal: embed shortlist on the fly
    # -----------------------------
    if mode == "universal":
        # Embed query
        q_emb = embedder.embed([query])[0]
        qv = np.asarray(q_emb, dtype=np.float32)
        qn = float(np.linalg.norm(qv) + 1e-12)

        # Embed all candidate texts with the SAME embedder
        texts = [c[1] for c in candidates]
        c_embs = embedder.embed(texts)

        rescored: List[Tuple[float, int, str, Dict[str, Any]]] = []
        for (mid, text, meta, _emb_blob, _norm), emb in zip(candidates, c_embs):
            v = np.asarray(emb, dtype=np.float32)
            vn = float(np.linalg.norm(v) + 1e-12)
            sim = _cosine(qv, qn, v, vn)
            rescored.append((sim, mid, text, meta))

        rescored.sort(key=lambda x: x[0], reverse=True)
        return rescored[: int(k_final)]

    # -----------------------------
    # 3) Stored: compare vs stored embeddings (fast path)
    # -----------------------------
    # Embed query
    q_emb = embedder.embed([query])[0]
    qv = np.asarray(q_emb, dtype=np.float32)
    qn = float(np.linalg.norm(qv) + 1e-12)

    rescored2: List[Tuple[float, int, str, Dict[str, Any]]] = []
    for mid, text, meta, emb_blob, norm in candidates:
        if emb_blob is None:
            continue

        # Decode stored embedding (supports JSON TEXT and float32 BLOB)
        emb = None
        try:
            if isinstance(emb_blob, (bytes, bytearray, memoryview)):
                b = bytes(emb_blob)
                try:
                    emb = json.loads(b.decode("utf-8"))
                except Exception:
                    emb = np.frombuffer(b, dtype=np.float32)
            elif isinstance(emb_blob, str):
                emb = json.loads(emb_blob)
            else:
                emb = emb_blob
        except Exception:
            continue

        v = np.asarray(emb, dtype=np.float32)

        # Skip incompatible dimensions
        if v.shape[0] != qv.shape[0]:
            continue

        vn = float(norm) if norm is not None else float(np.linalg.norm(v) + 1e-12)
        sim = _cosine(qv, qn, v, vn)
        rescored2.append((sim, mid, text, meta))

    rescored2.sort(key=lambda x: x[0], reverse=True)
    return rescored2[: int(k_final)]
