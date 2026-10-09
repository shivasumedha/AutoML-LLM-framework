"""Small SQLite store for workspace assets and durable assistant conversations."""
import json
import os
import sqlite3
from contextvars import ContextVar
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("AUTOML_DB_PATH", Path(os.getenv("AUTOML_DATA_DIR", _BACKEND_DIR)) / "automl.sqlite3")).resolve()
_session_id = ContextVar("automl_session_id", default=None)


def bind_session(session_id):
    return _session_id.set(session_id)


def reset_session(token):
    _session_id.reset(token)


def current_session():
    return _session_id.get()


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=15)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize():
    with _connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS datasets (
          id INTEGER PRIMARY KEY, name TEXT NOT NULL, content BLOB NOT NULL,
          created_at TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS artifacts (
          kind TEXT PRIMARY KEY, content BLOB NOT NULL, metadata TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chat_sessions (
          id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chat_messages (
          id INTEGER PRIMARY KEY, session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
          role TEXT NOT NULL, content TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS app_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS workspaces (
          session_id TEXT PRIMARY KEY REFERENCES chat_sessions(id) ON DELETE CASCADE,
          dataset_name TEXT, dataset_content BLOB,
          model_content BLOB, model_metadata TEXT NOT NULL DEFAULT '{}',
          updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, id);
        """)
        migrated = db.execute("SELECT value FROM app_meta WHERE key='legacy_workspace_migrated'").fetchone()
        if not migrated:
            session = db.execute("SELECT id FROM chat_sessions ORDER BY updated_at DESC LIMIT 1").fetchone()
            dataset = db.execute("SELECT name,content FROM datasets WHERE active=1 ORDER BY id DESC LIMIT 1").fetchone()
            model = db.execute("SELECT content,metadata FROM artifacts WHERE kind='trained_model'").fetchone()
            if session and (dataset or model):
                db.execute("INSERT OR IGNORE INTO workspaces(session_id,dataset_name,dataset_content,model_content,model_metadata,updated_at) VALUES(?,?,?,?,?,?)", (session["id"], dataset["name"] if dataset else None, dataset["content"] if dataset else None, model["content"] if model else None, model["metadata"] if model else "{}", now()))
            db.execute("INSERT INTO app_meta(key,value) VALUES('legacy_workspace_migrated','1')")


def now():
    return datetime.now(timezone.utc).isoformat()


def save_dataset(name, content):
    with _connect() as db:
        db.execute("UPDATE datasets SET active=0")
        db.execute("INSERT INTO datasets(name,content,created_at,active) VALUES(?,?,?,1)", (name, content, now()))
        session_id = current_session()
        if session_id:
            db.execute("INSERT INTO workspaces(session_id,dataset_name,dataset_content,model_content,model_metadata,updated_at) VALUES(?,?,?,NULL,'{}',?) ON CONFLICT(session_id) DO UPDATE SET dataset_name=excluded.dataset_name,dataset_content=excluded.dataset_content,updated_at=excluded.updated_at", (session_id, name, content, now()))


def latest_dataset():
    with _connect() as db:
        session_id = current_session()
        if session_id:
            row = db.execute("SELECT dataset_name AS name,dataset_content AS content FROM workspaces WHERE session_id=? AND dataset_content IS NOT NULL", (session_id,)).fetchone()
            return (row["name"], bytes(row["content"])) if row else None
        row = db.execute("SELECT name,content FROM datasets WHERE active=1 ORDER BY id DESC LIMIT 1").fetchone()
        return (row["name"], bytes(row["content"])) if row else None


def save_artifact(kind, content, metadata):
    with _connect() as db:
        db.execute("INSERT INTO artifacts(kind,content,metadata,updated_at) VALUES(?,?,?,?) ON CONFLICT(kind) DO UPDATE SET content=excluded.content,metadata=excluded.metadata,updated_at=excluded.updated_at", (kind, content, json.dumps(metadata), now()))
        session_id = current_session()
        if session_id and kind == "trained_model":
            db.execute("INSERT INTO workspaces(session_id,model_content,model_metadata,updated_at) VALUES(?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET model_content=excluded.model_content,model_metadata=excluded.model_metadata,updated_at=excluded.updated_at", (session_id, content, json.dumps(metadata), now()))


def get_artifact(kind):
    with _connect() as db:
        session_id = current_session()
        if session_id and kind == "trained_model":
            row = db.execute("SELECT model_content AS content,model_metadata AS metadata FROM workspaces WHERE session_id=? AND model_content IS NOT NULL", (session_id,)).fetchone()
            return (bytes(row["content"]), json.loads(row["metadata"])) if row else None
        row = db.execute("SELECT content,metadata FROM artifacts WHERE kind=?", (kind,)).fetchone()
        return (bytes(row["content"]), json.loads(row["metadata"])) if row else None


def delete_artifact(kind):
    with _connect() as db:
        db.execute("DELETE FROM artifacts WHERE kind=?", (kind,))
        session_id = current_session()
        if session_id and kind == "trained_model":
            db.execute("UPDATE workspaces SET model_content=NULL,model_metadata='{}',updated_at=? WHERE session_id=?", (now(), session_id))


def create_session(session_id, title="New chat"):
    stamp = now()
    with _connect() as db:
        db.execute("INSERT OR IGNORE INTO chat_sessions(id,title,created_at,updated_at) VALUES(?,?,?,?)", (session_id, title, stamp, stamp))
        db.execute("INSERT OR IGNORE INTO workspaces(session_id,updated_at) VALUES(?,?)", (session_id, stamp))
        row = db.execute("SELECT id,title,created_at,updated_at FROM chat_sessions WHERE id=?", (session_id,)).fetchone()
        return dict(row)


def list_sessions():
    with _connect() as db:
        return [dict(row) for row in db.execute("SELECT id,title,created_at,updated_at FROM chat_sessions ORDER BY updated_at DESC")]


def session_messages(session_id):
    with _connect() as db:
        if not db.execute("SELECT 1 FROM chat_sessions WHERE id=?", (session_id,)).fetchone():
            return None
        return [dict(row) for row in db.execute("SELECT role,content,metadata,created_at FROM chat_messages WHERE session_id=? ORDER BY id", (session_id,))]


def append_message(session_id, role, content, metadata=None):
    stamp = now()
    with _connect() as db:
        exists = db.execute("SELECT title FROM chat_sessions WHERE id=?", (session_id,)).fetchone()
        if not exists:
            db.execute("INSERT INTO chat_sessions(id,title,created_at,updated_at) VALUES(?,?,?,?)", (session_id, "New chat", stamp, stamp))
            exists = {"title": "New chat"}
        db.execute("INSERT INTO chat_messages(session_id,role,content,metadata,created_at) VALUES(?,?,?,?,?)", (session_id, role, content, json.dumps(metadata or {}), stamp))
        title = exists["title"]
        if title == "New chat" and role == "user":
            normalized = " ".join(content.split())
            title = normalized[:48]
            if len(normalized) > 48:
                title = title.rsplit(" ", 1)[0] or title
                title = title.rstrip(" ,.;:-") + "…"
            title = title or "New chat"
        db.execute("UPDATE chat_sessions SET title=?,updated_at=? WHERE id=?", (title, stamp, session_id))


def delete_session(session_id):
    with _connect() as db:
        cursor = db.execute("DELETE FROM chat_sessions WHERE id=?", (session_id,))
        return cursor.rowcount > 0


def rename_session(session_id, title):
    title = " ".join(title.split())[:64]
    if not title:
        return False
    with _connect() as db:
        cursor = db.execute("UPDATE chat_sessions SET title=?,updated_at=? WHERE id=?", (title, now(), session_id))
        return cursor.rowcount > 0
