# Research Assistant (Single DB, Pure Python)

Features:
- Branching dialog manager (Git-like)
- Chat → chunk → embed → store → retrieve
- Single SQLite DB file
- Hybrid retrieval (cosine + FTS5)
- Reproducible commits per branch

Dependencies:
- Python 3.9+
- openai
- numpy

DB file:
- assistant.sqlite