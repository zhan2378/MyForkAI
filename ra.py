#!/usr/bin/env python3
import argparse
import os
from assistant_db import AssistantDB
from oa_client import OAClient
from utils import chunk_text
from retrieval import hybrid_retrieve
import time
from collections import defaultdict
from provider_factory import make_provider

def auto_branch_name(prefix="branch"):
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"
def ensure_api_key(provider: str):
    provider = (provider or "openai").lower()
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("ERROR: OPENAI_API_KEY is not set in your environment.")
    elif provider == "gemini":
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise SystemExit("ERROR: GEMINI_API_KEY (or GOOGLE_API_KEY) is not set in your environment.")
    else:
        raise SystemExit(f"ERROR: Unknown provider: {provider}")

def normalize_provider_args(args):
    provider = (getattr(args, "provider", None) or "openai").lower()

    # --- Provider-specific temperature defaults ---
    if hasattr(args, "temperature") and args.temperature is None:
        if provider == "openai":
            args.temperature = 0.2
        elif provider == "gemini":
            args.temperature = 0.4

    # --- existing model / embed-model logic ---
    model = getattr(args, "model", None)
    embed_model = getattr(args, "embed_model", None)

    def is_openai_like(s):
        return isinstance(s, str) and (s.startswith("gpt-") or s.startswith("text-embedding-"))

    def is_gemini_like(s):
        return isinstance(s, str) and s.startswith("gemini-")

    if provider == "gemini":
        if model is None or is_openai_like(model):
            if hasattr(args, "model"):
                args.model = "gemini-3-pro-preview"
        if embed_model is None or is_openai_like(embed_model):
            if hasattr(args, "embed_model"):
                args.embed_model = "gemini-embedding-001"
        if hasattr(args, "provider"):
            args.provider = "gemini"
    else:
        if model is None or is_gemini_like(model):
            if hasattr(args, "model"):
                args.model = "gpt-5.2"
        if embed_model is None or is_gemini_like(embed_model):
            if hasattr(args, "embed_model"):
                args.embed_model = "text-embedding-3-small"
        if hasattr(args, "provider"):
            args.provider = "openai"

    return args

def cmd_new(args):
    db = AssistantDB(args.db)
    params = {"model": args.model, "temperature": args.temperature}
    dialog_id, root_id = db.create_dialog(args.title, args.system, params)
    # head is set inside create_dialog() after your update
    print("dialog_id:", dialog_id)
    print("root_node:", root_id)

def cmd_fork(args):
    db = AssistantDB(args.db)
    params = {"model": args.model, "temperature": args.temperature}

    from_node = args.from_node or db.get_head(args.dialog_id)
    if not from_node:
        raise SystemExit("ERROR: Provide --from-node or set a head for this dialog.")

    note = args.note if args.note is not None else auto_branch_name(args.prefix)

    node_id = db.fork(args.dialog_id, from_node, note, params)
    print("new_node:", node_id)
    print("from_node:", from_node)
    print("note:", note)

def cmd_ask(args):
    ensure_api_key(args.provider)
    db = AssistantDB(args.db)

    node_id = args.node_id or db.get_head(args.dialog_id)
    if not node_id:
        raise SystemExit("ERROR: Provide --node-id or set a head for this dialog.")

    provider = make_provider(args.provider, args.model, args.embed_model)

    # append user question
    db.append_message(args.dialog_id, node_id, "user", args.question)

    # call model
    msgs = db.get_messages(args.dialog_id, node_id)
    ans = provider.respond(msgs, temperature=args.temperature, max_tokens=args.max_tokens)

    # store assistant reply
    db.append_message(args.dialog_id, node_id, "assistant", ans)

    # optionally embed + store to memory
    if not args.no_memory:
        for ch in chunk_text(ans, max_chars=args.chunk_chars, overlap=args.chunk_overlap):
            emb = provider.embed([ch])[0]
            db.add_memory(args.dialog_id, node_id, ch, emb, {"type": "assistant_answer", "node": node_id,"embed_provider": args.provider,"embed_model": args.embed_model})
    # update head
    db.set_head(args.dialog_id, node_id)

    print(ans.strip())

def cmd_retrieve(args):
    ensure_api_key(args.provider)
    db = AssistantDB(args.db)

    provider_client = make_provider(args.provider, args.model, args.embed_model)
    if args.embed_provider == "any":
        ep = None
    elif args.embed_provider == "same":
        ep = args.provider
    else:
        ep = args.embed_provider
    hits = hybrid_retrieve(
        db,
        provider_client,
        args.dialog_id,
        args.query,
        k_final=args.k,
        k_fts=args.k_fts,
        mode="universal",
        embed_provider=ep,   # None means ANY
    )



    if not hits:
        print("(no hits)")
        return

    for sim, mid, text, meta in hits:
        print(f"\n--- id={mid} sim={sim:.4f} meta={meta}")
        print(text.strip())

def cmd_nodes(args):
    db = AssistantDB(args.db)
    head = db.get_head(args.dialog_id)
    rows = db.list_nodes(args.dialog_id, limit=args.limit)
    for node_id, parent_id, note, commit, created_at in rows:
        mark = " <HEAD>" if head == node_id else ""
        print(f"{node_id}  parent={parent_id}  commit={commit}  note={note}{mark}")

def cmd_head(args):
    db = AssistantDB(args.db)
    head = db.get_head(args.dialog_id)
    print(head if head else "(no head set)")
def cmd_checkout(args):
    db = AssistantDB(args.db)
    db.set_head_to_node(args.dialog_id, args.node_id)
    print("HEAD set to:", args.node_id)

def cmd_tree(args):
    db = AssistantDB(args.db)
    head = db.get_head(args.dialog_id)

    # fetch nodes for dialog
    rows = db.conn.execute(
        "SELECT node_id, parent_id, note FROM nodes WHERE dialog_id=? ORDER BY created_at ASC",
        (args.dialog_id,)
    ).fetchall()

    if not rows:
        print("(no nodes)")
        return

    children = defaultdict(list)
    parent_of = {}
    note_of = {}

    root = None
    for nid, pid, note in rows:
        parent_of[nid] = pid
        note_of[nid] = note
        children[pid].append(nid)
        if pid is None:
            root = nid

    def walk(nid, prefix="", is_last=True):
        mark = " <HEAD>" if nid == head else ""
        label = f"{nid} [{note_of.get(nid,'')}]"
        if prefix == "":
            print(label + mark)
        else:
            branch = "└── " if is_last else "├── "
            print(prefix + branch + label + mark)

        kids = children.get(nid, [])
        for i, kid in enumerate(kids):
            last = (i == len(kids) - 1)
            ext = "    " if is_last else "│   "
            walk(kid, prefix + ext, last)

    walk(root)

def cmd_diff(args):
    db = AssistantDB(args.db)

    a = args.a
    b = args.b

    # ensure both in the dialog
    for nid in [a, b]:
        row = db.conn.execute(
            "SELECT 1 FROM nodes WHERE node_id=? AND dialog_id=?",
            (nid, args.dialog_id)
        ).fetchone()
        if not row:
            raise SystemExit(f"ERROR: node {nid} not found in dialog {args.dialog_id}")

    lca = db.lca(a, b)
    if lca is None:
        raise SystemExit("ERROR: No common ancestor found (unexpected).")

    # messages are materialized per node (copied on fork),
    # so diff is just: suffix after LCA message length.
    msgs_lca = db.messages_for_node(lca)
    msgs_a = db.messages_for_node(a)
    msgs_b = db.messages_for_node(b)

    n = len(msgs_lca)
    add_a = msgs_a[n:]
    add_b = msgs_b[n:]

    print("LCA:", lca)
    print("\n=== Only in A (after LCA) ===")
    if not add_a:
        print("(none)")
    else:
        for i, m in enumerate(add_a, 1):
            print(f"\n[A+{i}] {m['role'].upper()}:\n{m['content']}")

    print("\n=== Only in B (after LCA) ===")
    if not add_b:
        print("(none)")
    else:
        for i, m in enumerate(add_b, 1):
            print(f"\n[B+{i}] {m['role'].upper()}:\n{m['content']}")
def main():
    p = argparse.ArgumentParser(prog="ra", description="Branching dialog + retrieval (single SQLite DB).")
    p.add_argument("--db", default="assistant.sqlite", help="SQLite DB path (default: assistant.sqlite)")

    sp = p.add_subparsers(dest="cmd", required=True)

    # new
    p_new = sp.add_parser("new", help="Create a new dialog (root node).")
    p_new.add_argument("--title", required=True)
    p_new.add_argument("--system", required=True, help="System prompt for the dialog.")
    p_new.add_argument("--model", default="gpt-5.2")
    p_new.add_argument("--temperature", type=float, default=0.2)
    p_new.set_defaults(func=cmd_new)

    # fork
    p_fork = sp.add_parser("fork", help="Fork a new node from an existing node (defaults to head).")
    p_fork.add_argument("--dialog-id", required=True)
    p_fork.add_argument("--from-node", dest="from_node", default=None)
    p_fork.add_argument("--note", default=None, help="Branch note; if omitted, auto-named.")
    p_fork.add_argument("--prefix", default="branch", help="Prefix for auto branch naming.")
    p_fork.add_argument("--model", default="gpt-5.2")
    p_fork.add_argument("--temperature", type=float, default=0.2)
    p_fork.set_defaults(func=cmd_fork)

    # ask
    p_ask = sp.add_parser("ask", help="Ask a question on a node (defaults to head) and store answer.")
    p_ask.add_argument("--dialog-id", required=True)
    p_ask.add_argument("--node-id", dest="node_id", default=None)
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--model", default="gpt-5.2")
    p_ask.add_argument("--embed-model", default="text-embedding-3-small")
    p_ask.add_argument("--temperature", type=float, default=None)
    p_ask.add_argument("--max-tokens", type=int, default=900)
    p_ask.add_argument("--no-memory", action="store_true", help="Do not store response chunks into memory.")
    p_ask.add_argument("--chunk-chars", type=int, default=1800)
    p_ask.add_argument("--chunk-overlap", type=int, default=200)
    p_ask.add_argument("--provider", default="openai", choices=["openai","gemini"])
    p_ask.set_defaults(func=cmd_ask)

    # retrieve
    p_ret = sp.add_parser("retrieve", help="Retrieve from memory (hybrid semantic + keyword).")
    p_ret.add_argument("--dialog-id", required=True)
    p_ret.add_argument("--query", required=True)
    p_ret.add_argument("--model", default="gpt-5.2")
    p_ret.add_argument("--embed-model", default="text-embedding-3-small")
    p_ret.add_argument("--k", type=int, default=5)
    p_ret.add_argument("--k-sem", type=int, default=10)
    p_ret.add_argument("--k-fts", type=int, default=10)
    p_ret.add_argument("--provider", default="openai", choices=["openai","gemini"])
    p_ret.add_argument(
        "--embed-provider",
        default="same",
        choices=["same", "any"],
        help="Use same embedding provider as --provider, or search across all stored embeddings."
    )
    p_ret.set_defaults(func=cmd_retrieve)

    # nodes
    p_nodes = sp.add_parser("nodes", help="List recent nodes for a dialog (marks HEAD).")
    p_nodes.add_argument("--dialog-id", required=True)
    p_nodes.add_argument("--limit", type=int, default=50)
    p_nodes.set_defaults(func=cmd_nodes)

    # head
    p_head = sp.add_parser("head", help="Print current head node for a dialog.")
    p_head.add_argument("--dialog-id", required=True)
    p_head.set_defaults(func=cmd_head)

    # checkout
    p_co = sp.add_parser("checkout", help="Set dialog HEAD to an existing node.")
    p_co.add_argument("--dialog-id", required=True)
    p_co.add_argument("--node-id", dest="node_id", required=True)
    p_co.set_defaults(func=cmd_checkout)

    #tree
    p_tree = sp.add_parser("tree", help="ASCII tree of nodes/branches for a dialog.")
    p_tree.add_argument("--dialog-id", required=True)
    p_tree.set_defaults(func=cmd_tree)

    #diff
    p_diff = sp.add_parser("diff", help="Diff two nodes (messages added after LCA).")
    p_diff.add_argument("--dialog-id", required=True)
    p_diff.add_argument("--a", required=True, help="Node A id")
    p_diff.add_argument("--b", required=True, help="Node B id")
    p_diff.set_defaults(func=cmd_diff)

    args = p.parse_args()
    args = normalize_provider_args(args)
    args.func(args)


if __name__ == "__main__":
    main()