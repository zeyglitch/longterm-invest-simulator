import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    id            INTEGER PRIMARY KEY,
    isin          TEXT UNIQUE,
    name          TEXT NOT NULL,
    ticker        TEXT,
    fees          REAL DEFAULT 0,
    target_weight REAL
);

CREATE TABLE IF NOT EXISTS transactions (
    id         INTEGER PRIMARY KEY,
    date       TEXT NOT NULL,
    asset_id   INTEGER NOT NULL REFERENCES assets(id),
    kind       TEXT NOT NULL CHECK (kind IN ('BUY', 'SELL')),
    quantity   REAL NOT NULL CHECK (quantity > 0),
    unit_price REAL NOT NULL CHECK (unit_price > 0),
    fees       REAL NOT NULL DEFAULT 0,
    note       TEXT,
    source     TEXT NOT NULL DEFAULT 'manual'
);

CREATE TABLE IF NOT EXISTS cash_movements (
    id     INTEGER PRIMARY KEY,
    date   TEXT NOT NULL,
    kind   TEXT NOT NULL CHECK (kind IN ('DEPOSIT', 'WITHDRAWAL')),
    amount REAL NOT NULL CHECK (amount > 0),
    note   TEXT
);

CREATE TABLE IF NOT EXISTS prices (
    asset_id INTEGER NOT NULL REFERENCES assets(id),
    date     TEXT NOT NULL,
    close    REAL NOT NULL,
    source   TEXT NOT NULL DEFAULT 'yahoo',
    PRIMARY KEY (asset_id, date, source)
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plans (
    id      INTEGER PRIMARY KEY,
    name    TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL
);
"""


def get_connection(path: str | Path) -> sqlite3.Connection:
    """Open the database at `path` (use ":memory:" for tests), creating the folder if needed."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create the tables that do not exist yet. Safe to call at every start."""
    conn.executescript(SCHEMA)
    conn.commit()
