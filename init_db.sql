-- dialogs
CREATE TABLE IF NOT EXISTS dialogs (
  dialog_id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  created_at REAL NOT NULL
);

-- branching nodes
CREATE TABLE IF NOT EXISTS nodes (
  node_id TEXT PRIMARY KEY,
  dialog_id TEXT NOT NULL,
  parent_id TEXT,
  note TEXT NOT NULL,
  commit_hash NOT NULL,
  params_json TEXT NOT NULL,
  created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_nodes_dialog ON nodes(dialog_id);
CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);

-- messages
CREATE TABLE IF NOT EXISTS messages (
  message_id INTEGER PRIMARY KEY AUTOINCREMENT,
  dialog_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_node ON messages(node_id);

-- vector memory
CREATE TABLE IF NOT EXISTS memory (
  memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
  dialog_id TEXT NOT NULL,
  node_id TEXT,
  text TEXT NOT NULL,
  embedding BLOB NOT NULL,
  norm REAL NOT NULL,
  meta_json TEXT NOT NULL,
  created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_dialog ON memory(dialog_id);
CREATE INDEX IF NOT EXISTS idx_memory_node ON memory(node_id);

-- keyword search
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
USING fts5(text, content='memory', content_rowid='memory_id');

-- triggers
CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory BEGIN
  INSERT INTO memory_fts(rowid, text) VALUES (new.memory_id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memory BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, text) VALUES('delete', old.memory_id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE ON memory BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, text) VALUES('delete', old.memory_id, old.text);
  INSERT INTO memory_fts(rowid, text) VALUES (new.memory_id, new.text);
END;
-- track current head node per dialog (so CLI can default to it)
CREATE TABLE IF NOT EXISTS dialog_heads (
  dialog_id TEXT PRIMARY KEY,
  head_node_id TEXT NOT NULL,
  updated_at REAL NOT NULL
);