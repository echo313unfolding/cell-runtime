-- Evidence Graph Schema Reference
-- Source: ~/fgip-engine/fgip.db
-- This is a REFERENCE COPY for the cell-runtime spec.
-- The canonical schema lives in fgip-engine.

-- Core graph tables
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    node_type TEXT,
    metadata TEXT,  -- JSON
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL REFERENCES nodes(id),
    target_id TEXT NOT NULL REFERENCES nodes(id),
    relation TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    metadata TEXT,  -- JSON
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    source_id TEXT,
    confidence REAL DEFAULT 0.5,
    category TEXT,
    metadata TEXT,  -- JSON
    created_at TEXT DEFAULT (datetime('now'))
);

-- FTS5 indices (auto-created by FGIP engine)
-- CREATE VIRTUAL TABLE nodes_fts USING fts5(name, content=nodes, content_rowid=rowid);
-- CREATE VIRTUAL TABLE edges_fts USING fts5(relation, content=edges, content_rowid=rowid);
-- CREATE VIRTUAL TABLE claims_fts USING fts5(text, content=claims, content_rowid=rowid);

-- Sentinel SSM tables (in sentinel.db, separate from FGIP)
-- alerts: detection events
-- verdicts: classification decisions
-- investigations: case tracking
-- hash_cache: content addressing
-- config: runtime configuration
