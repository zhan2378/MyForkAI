import time, uuid, hashlib, json

def now():
    return time.time()

def new_id(n=12):
    return uuid.uuid4().hex[:n]

def compute_commit(parent_commit, added_messages, params):
    h = hashlib.sha256()
    h.update((parent_commit or "").encode())
    h.update(json.dumps(added_messages, sort_keys=True).encode())
    h.update(json.dumps(params, sort_keys=True).encode())
    return h.hexdigest()[:12]

def chunk_text(text, max_chars=1800, overlap=200):
    text = text.strip()
    chunks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        chunks.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - overlap)
    return chunks