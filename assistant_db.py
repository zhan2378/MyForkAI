import sqlite3, json
import numpy as np
from utils import now, new_id, compute_commit

class AssistantDB:
    def __init__(self, path="assistant.sqlite"):
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        with open("init_db.sql", "r") as f:
            self.conn.executescript(f.read())
        self.conn.commit()

    def create_dialog(self, title, system_prompt, params):
        dialog_id = new_id()
        root_id = new_id()
        self.conn.execute(
            "INSERT INTO dialogs VALUES (?,?,?)",
            (dialog_id, title, now())
        )

        commit_hash = compute_commit(None, [{"role":"system","content":system_prompt}], params)
        self.conn.execute(
            "INSERT INTO nodes VALUES (?,?,?,?,?,?,?)",
            (root_id, dialog_id, None, "root", commit_hash, json.dumps(params), now())
        )

        self.conn.execute(
            "INSERT INTO messages VALUES (NULL,?,?,?,?,?)",
            (dialog_id, root_id, "system", system_prompt, now())
        )
        self.conn.commit()
        self.set_head(dialog_id, root_id)
        return dialog_id, root_id

    def fork(self, dialog_id, parent_node, note, params):
        node_id = new_id()
        parent_commit = self.conn.execute(
            "SELECT commit_hash FROM nodes WHERE node_id=?", (parent_node,)
        ).fetchone()[0]

        commit_hash = compute_commit(parent_commit, [], params)
        self.conn.execute(
            "INSERT INTO nodes VALUES (?,?,?,?,?,?,?)",
            (node_id, dialog_id, parent_node, note, commit_hash, json.dumps(params), now())
        )

        rows = self.conn.execute(
            "SELECT role,content,created_at FROM messages WHERE node_id=?",
            (parent_node,)
        ).fetchall()

        for r,c,t in rows:
            self.conn.execute(
                "INSERT INTO messages VALUES (NULL,?,?,?,?,?)",
                (dialog_id, node_id, r, c, t)
            )
        self.conn.commit()
        self.set_head(dialog_id, node_id)
        return node_id

    def append_message(self, dialog_id, node_id, role, content):
        self.conn.execute(
            "INSERT INTO messages VALUES (NULL,?,?,?,?,?)",
            (dialog_id, node_id, role, content, now())
        )
        self.conn.commit()

    def get_messages(self, dialog_id, node_id):
        rows = self.conn.execute(
            "SELECT role,content FROM messages WHERE node_id=? ORDER BY message_id",
            (node_id,)
        ).fetchall()
        return [{"role":r,"content":c} for r,c in rows]

    def add_memory(self, dialog_id, node_id, text, embedding, meta):
        v = np.asarray(embedding, dtype=np.float32)
        self.conn.execute(
            "INSERT INTO memory VALUES (NULL,?,?,?,?,?,?,?)",
            (dialog_id, node_id, text, v.tobytes(),
             float(np.linalg.norm(v)+1e-12), json.dumps(meta), now())
        )
        self.conn.commit()
        # ---- heads ----

    def set_head(self, dialog_id, node_id):
        self.conn.execute(
            """INSERT INTO dialog_heads(dialog_id, head_node_id, updated_at)
               VALUES(?,?,?)
               ON CONFLICT(dialog_id) DO UPDATE SET
                 head_node_id=excluded.head_node_id,
                 updated_at=excluded.updated_at
            """,
            (dialog_id, node_id, now())
        )
        self.conn.commit()

    def get_head(self, dialog_id):
        row = self.conn.execute(
            "SELECT head_node_id FROM dialog_heads WHERE dialog_id=?",
            (dialog_id,)
        ).fetchone()
        return row[0] if row else None

    # ---- listing ----

    def list_dialogs(self, limit=50):
        rows = self.conn.execute(
            "SELECT dialog_id, title, created_at FROM dialogs ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return rows

    def list_nodes(self, dialog_id, limit=100):
        rows = self.conn.execute(
            """SELECT node_id, parent_id, note, commit_hash, created_at
               FROM nodes WHERE dialog_id=?
               ORDER BY created_at DESC LIMIT ?""",
            (dialog_id, limit)
        ).fetchall()
        return rows
        # ---- node helpers ----

    def get_node(self, node_id):
        row = self.conn.execute(
            "SELECT node_id, dialog_id, parent_id, note, commit_hash, created_at FROM nodes WHERE node_id=?",
            (node_id,)
        ).fetchone()
        return row  # (node_id, dialog_id, parent_id, note, commit_hash, created_at) or None

    def get_parent(self, node_id):
        row = self.conn.execute("SELECT parent_id FROM nodes WHERE node_id=?", (node_id,)).fetchone()
        return row[0] if row else None

    def lineage(self, node_id):
        """Return [root ... node_id]."""
        path = []
        cur = node_id
        while cur is not None:
            path.append(cur)
            cur = self.get_parent(cur)
        return list(reversed(path))

    def lca(self, a, b):
        """Lowest common ancestor node_id (within same dialog)."""
        la = self.lineage(a)
        lb = self.lineage(b)
        i = 0
        lca = None
        while i < len(la) and i < len(lb) and la[i] == lb[i]:
            lca = la[i]
            i += 1
        return lca

    def messages_for_node(self, node_id):
        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE node_id=? ORDER BY message_id",
            (node_id,)
        ).fetchall()
        return [{"role": r, "content": c} for r, c in rows]

    def set_head_to_node(self, dialog_id, node_id):
        # sanity: ensure node belongs to dialog
        row = self.conn.execute(
            "SELECT 1 FROM nodes WHERE node_id=? AND dialog_id=?",
            (node_id, dialog_id)
        ).fetchone()
        if not row:
            raise ValueError(f"node_id {node_id} not found in dialog {dialog_id}")
        self.set_head(dialog_id, node_id)