from assistant_db import AssistantDB
from oa_client import OAClient
from utils import chunk_text
from retrieval import hybrid_retrieve

params = {"model":"gpt-5.2","temperature":0.2}

db = AssistantDB()
oa = OAClient("gpt-5.2", "text-embedding-3-small")

dialog_id, root = db.create_dialog(
    "Bandit Research",
    "You are my research assistant. Be concise and rigorous.",
    params
)

theory = db.fork(dialog_id, root, "theory", params)

db.append_message(dialog_id, theory, "user",
    "Give a CI-based stopping rule for uniform sampling best-arm ID.")

ans = oa.respond(db.get_messages(dialog_id, theory))
db.append_message(dialog_id, theory, "assistant", ans)

for ch in chunk_text(ans):
    emb = oa.embed([ch])[0]
    db.add_memory(dialog_id, theory, ch, emb, {"branch":"theory"})

hits = hybrid_retrieve(db, oa, dialog_id,
    "early stopping uniform sampling", k=3)

print("Retrieved:")
for _,t,_ in hits:
    print("-", t[:120])