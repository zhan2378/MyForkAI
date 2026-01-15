# frontend_streamlit.py
import os
import time
import streamlit as st
from collections import defaultdict

from assistant_db import AssistantDB
from utils import chunk_text
from retrieval import hybrid_retrieve
from provider_factory import make_provider

st.set_page_config(page_title="RA Dialog Tree", layout="wide")

# ---------------- Sidebar: global config ----------------
st.sidebar.title("RA Frontend")

db_path = st.sidebar.text_input("DB path", value="assistant.sqlite")

provider = st.sidebar.selectbox("Provider", ["openai", "gemini"], index=0)

# Provider-specific defaults
if provider == "openai":
    default_model = "gpt-5.2"
    default_embed = "text-embedding-3-small"
else:
    default_model = "gemini-3-pro-preview"
    default_embed = "gemini-embedding-001"

model = st.sidebar.text_input("Model", value=default_model)
embed_model = st.sidebar.text_input("Embed model", value=default_embed)
default_temp = 0.2 if provider == "openai" else 0.4
temperature = st.slider(
    "Temperature",
    0.0, 1.5,
    value=default_temp,
    step=0.05
)
max_tokens = st.sidebar.number_input("Max tokens", min_value=50, max_value=4000, value=900, step=50)
store_memory = st.sidebar.checkbox("Store assistant reply into memory", value=True)

st.sidebar.divider()

# ---------------- DB load ----------------
db = AssistantDB(db_path)

dialogs = db.list_dialogs(limit=200)
if not dialogs:
    st.warning('No dialogs found in DB. Create one with: `python ra.py new --title "..." --system "..."`')
    st.stop()

dialog_labels = [f"{title}  ({did})" for (did, title, _created_at) in dialogs]
selected_dialog_label = st.sidebar.selectbox("Dialog", dialog_labels, index=0)
dialog_id = selected_dialog_label.split("(")[-1].rstrip(")")

head = db.get_head(dialog_id)

# ---------------- Build tree (nodes) ----------------
rows = db.conn.execute(
    "SELECT node_id, parent_id, note FROM nodes WHERE dialog_id=? ORDER BY created_at ASC",
    (dialog_id,),
).fetchall()

if not rows:
    st.warning("No nodes found for this dialog.")
    st.stop()

children = defaultdict(list)
note_of = {}
root = None

for nid, pid, note in rows:
    children[pid].append(nid)
    note_of[nid] = note
    if pid is None:
        root = nid

if root is None:
    st.error("No root node found (parent_id IS NULL).")
    st.stop()

# Flatten tree for selection
options = []  # (label, node_id)

def walk(nid: str, depth: int = 0):
    label = ("  " * depth) + f"{nid} [{note_of.get(nid, '')}]"
    if nid == head:
        label += " <HEAD>"
    options.append((label, nid))
    for kid in children.get(nid, []):
        walk(kid, depth + 1)

walk(root)

st.sidebar.subheader("Nodes")
selected_node_label = st.sidebar.selectbox(
    "Click a node",
    [lbl for (lbl, _nid) in options],
    index=0,
)
node_id = dict(options)[selected_node_label]

c_head1, c_head2 = st.sidebar.columns(2)
with c_head1:
    if st.button("Checkout (HEAD)"):
        db.set_head(dialog_id, node_id)
        st.sidebar.success(f"HEAD → {node_id}")
        st.rerun()

with c_head2:
    if st.button("Fork here"):
        params = {"provider": provider, "model": model, "temperature": temperature}
        note = f"ui-{time.strftime('%m%d-%H%M%S')}"
        new_node = db.fork(dialog_id, node_id, note, params)
        st.sidebar.success(f"Forked: {new_node} (HEAD moved)")
        st.rerun()

# ---------------- Helper: key checks ----------------
def ensure_provider_key(selected_provider: str):
    if selected_provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY is not set.")
            st.stop()
    else:
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            st.error("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")
            st.stop()

# ---------------- Main layout ----------------
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Tree (ASCII)")

    def print_tree(nid: str, prefix: str = "", is_last: bool = True):
        mark = " <HEAD>" if nid == head else ""
        if prefix == "":
            st.text(f"{nid} [{note_of.get(nid,'')}]"+mark)
        else:
            branch = "└── " if is_last else "├── "
            st.text(prefix + branch + f"{nid} [{note_of.get(nid,'')}]"+mark)

        kids = children.get(nid, [])
        for i, kid in enumerate(kids):
            last = (i == len(kids) - 1)
            ext = "    " if is_last else "│   "
            print_tree(kid, prefix + ext, last)

    print_tree(root)

    st.divider()
    st.subheader("Retrieve (hybrid)")

    q = st.text_input("Query", placeholder="e.g., early stopping uniform sampling")
    if q and q.strip():
        ensure_provider_key(provider)
        provider_client = make_provider(provider, model, embed_model)

        hits = hybrid_retrieve(db, provider_client, dialog_id, q.strip(), k_final=5)
        if not hits:
            st.write("(no hits)")
        else:
            for sim, mid, text, meta in hits:
                st.caption(f"id={mid}  sim={sim:.4f}  meta={meta}")
                st.write(text)

with right:
    st.subheader(f"Chat — node {node_id}")

    msgs = db.get_messages(dialog_id, node_id)
    for m in msgs:
        role = m["role"]
        content = m["content"]
        if role == "system":
            st.info(content)
        elif role == "user":
            st.markdown(f"**You:** {content}")
        else:
            st.markdown(f"**Assistant:** {content}")

    st.divider()

    # Keep a stable key for session state
    input_key = "chat_input"
    if input_key not in st.session_state:
        st.session_state[input_key] = ""

    user_text = st.text_area("Message", height=120, key=input_key, placeholder="Talk to this node...")

    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        if st.button("Send"):
            if not user_text.strip():
                st.warning("Type a message first.")
                st.stop()

            ensure_provider_key(provider)
            provider_client = make_provider(provider, model, embed_model)

            # Store user message
            db.append_message(dialog_id, node_id, "user", user_text.strip())

            # Generate assistant reply
            ans = provider_client.respond(
                db.get_messages(dialog_id, node_id),
                temperature=temperature,
                max_tokens=int(max_tokens),
            )
            db.append_message(dialog_id, node_id, "assistant", ans)

            # Optional memory store (chunks + embeddings)
            if store_memory:
                for ch in chunk_text(ans):
                    emb = provider_client.embed([ch])[0]
                    db.add_memory(
                        dialog_id,
                        node_id,
                        ch,
                        emb,
                        {"type": "assistant_answer", "node": node_id,  "embed_provider": provider, "embed_model": embed_model},
                    )

            # Keep HEAD on this node
            db.set_head(dialog_id, node_id)

            # Clear input
            st.session_state[input_key] = ""
            st.rerun()

    with c2:
        if st.button("Clear input"):
            st.session_state[input_key] = ""
            st.rerun()

    with c3:
        st.caption("Each node is its own branch. Clicking a node switches where you chat.")