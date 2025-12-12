# 📘 Research Assistant – User Manual (CLI-Only)

This project is a **branching conversational research assistant** built on top of the OpenAI API and a **single SQLite database**.

It provides:
- Git-like **dialog branching**
- Per-branch GPT conversations
- Branch comparison (diffing)
- Long-term memory with retrieval
- Fully reproducible, file-based storage

Everything lives in **one SQLite file**: `assistant.sqlite`.

---

## 1. Core Concepts

| Concept | Meaning |
|------|--------|
| **Dialog** | A research project / conversation tree |
| **Node** | A branch (commit) in the dialog |
| **HEAD** | The currently active node |
| **Fork** | Create a new branch from an existing node |
| **Message** | A system / user / assistant message |
| **Memory** | Embedded assistant outputs for retrieval |

**Mental model:**  
> *Git for reasoning, GPT for execution.*

---

## 2. Installation

### Requirements
- Python ≥ 3.9
- OpenAI API access (separate from ChatGPT Plus)

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 3. OpenAI API Key Setup

Set the API key as an environment variable:

```bash
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"
```

Verify:
```bash
echo $OPENAI_API_KEY
```

⚠️ Do **not** hardcode your key.  
⚠️ Do **not** commit your database.

Add to `.gitignore`:
```gitignore
assistant.sqlite
.env
```

---

## 4. Database

- All state is stored in:
  ```
  assistant.sqlite
  ```
- One DB can hold **multiple dialogs**.

---

## 5. CLI Overview (`ra.py`)

All interaction happens through:
```bash
python ra.py <command> [options]
```

---

## 6. Create a Dialog

```bash
python ra.py new \
  --title "Bandit Notes" \
  --system "You are my research assistant. Be concise and rigorous."
```

---

## 7. Branching & Navigation

```bash
python ra.py fork --dialog-id <ID>
python ra.py checkout --dialog-id <ID> --node-id <NODE>
python ra.py nodes --dialog-id <ID>
python ra.py tree --dialog-id <ID>
```

---

## 8. Asking GPT

```bash
python ra.py ask --dialog-id <ID> --question "Explain UCB regret"
```

---

## 9. Diffing Branches

```bash
python ra.py diff --dialog-id <ID> --a <NODE_A> --b <NODE_B>
```

---

## 10. Retrieval

```bash
python ra.py retrieve --dialog-id <ID> --query "early stopping"
```

---

## 11. Recommended Workflow

1. Create a dialog per project
2. Fork branches for theory / simulation
3. Chat independently on each branch
4. Diff branches to compare reasoning
5. Archive `assistant.sqlite`

---

## 12. One-Line Summary

> **A local, reproducible, Git-like reasoning system powered by GPT.**
