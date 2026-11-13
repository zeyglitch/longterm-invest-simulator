import sqlite3
from datetime import date
from pathlib import Path

from simulator.models import Asset, CashMovement, Transaction

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


def _row_to_asset(row: sqlite3.Row) -> Asset:
    return Asset(
        id=row["id"],
        isin=row["isin"],
        name=row["name"],
        ticker=row["ticker"],
        fees=row["fees"],
        target_weight=row["target_weight"],
    )


def add_asset(conn: sqlite3.Connection, asset: Asset) -> int:
    """Insert an asset and return its new id. Raises sqlite3.IntegrityError if the ISIN exists."""
    cursor = conn.execute(
        "INSERT INTO assets (isin, name, ticker, fees, target_weight) VALUES (?, ?, ?, ?, ?)",
        (asset.isin, asset.name, asset.ticker, asset.fees, asset.target_weight),
    )
    conn.commit()
    return cursor.lastrowid


def update_asset(conn: sqlite3.Connection, asset: Asset) -> None:
    conn.execute(
        "UPDATE assets SET isin = ?, name = ?, ticker = ?, fees = ?, target_weight = ? "
        "WHERE id = ?",
        (asset.isin, asset.name, asset.ticker, asset.fees, asset.target_weight, asset.id),
    )
    conn.commit()


def list_assets(conn: sqlite3.Connection) -> list[Asset]:
    rows = conn.execute("SELECT * FROM assets ORDER BY name").fetchall()
    return [_row_to_asset(row) for row in rows]


def get_asset_by_isin(conn: sqlite3.Connection, isin: str) -> Asset | None:
    row = conn.execute("SELECT * FROM assets WHERE isin = ?", (isin,)).fetchone()
    return _row_to_asset(row) if row else None


def _row_to_transaction(row: sqlite3.Row) -> Transaction:
    return Transaction(
        id=row["id"],
        date=date.fromisoformat(row["date"]),
        asset_id=row["asset_id"],
        kind=row["kind"],
        quantity=row["quantity"],
        unit_price=row["unit_price"],
        fees=row["fees"],
        note=row["note"] or "",
        source=row["source"],
    )


def add_transaction(conn: sqlite3.Connection, tx: Transaction) -> int:
    """Insert a transaction and return its id. A quantity or price <= 0 is refused by the table."""
    cursor = conn.execute(
        "INSERT INTO transactions (date, asset_id, kind, quantity, unit_price, fees, note, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            tx.date.isoformat(),
            tx.asset_id,
            tx.kind,
            tx.quantity,
            tx.unit_price,
            tx.fees,
            tx.note,
            tx.source,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def update_transaction(conn: sqlite3.Connection, tx: Transaction) -> None:
    conn.execute(
        "UPDATE transactions SET date = ?, asset_id = ?, kind = ?, quantity = ?, "
        "unit_price = ?, fees = ?, note = ?, source = ? WHERE id = ?",
        (
            tx.date.isoformat(),
            tx.asset_id,
            tx.kind,
            tx.quantity,
            tx.unit_price,
            tx.fees,
            tx.note,
            tx.source,
            tx.id,
        ),
    )
    conn.commit()


def delete_transaction(conn: sqlite3.Connection, transaction_id: int) -> None:
    conn.execute("DELETE FROM transactions WHERE id = ?", (transaction_id,))
    conn.commit()


def list_transactions(conn: sqlite3.Connection) -> list[Transaction]:
    """All transactions in chronological order (same day: order of entry)."""
    rows = conn.execute("SELECT * FROM transactions ORDER BY date, id").fetchall()
    return [_row_to_transaction(row) for row in rows]


def _row_to_cash_movement(row: sqlite3.Row) -> CashMovement:
    return CashMovement(
        id=row["id"],
        date=date.fromisoformat(row["date"]),
        kind=row["kind"],
        amount=row["amount"],
        note=row["note"] or "",
    )


def add_cash_movement(conn: sqlite3.Connection, movement: CashMovement) -> int:
    cursor = conn.execute(
        "INSERT INTO cash_movements (date, kind, amount, note) VALUES (?, ?, ?, ?)",
        (movement.date.isoformat(), movement.kind, movement.amount, movement.note),
    )
    conn.commit()
    return cursor.lastrowid


def delete_cash_movement(conn: sqlite3.Connection, movement_id: int) -> None:
    conn.execute("DELETE FROM cash_movements WHERE id = ?", (movement_id,))
    conn.commit()


def list_cash_movements(conn: sqlite3.Connection) -> list[CashMovement]:
    rows = conn.execute("SELECT * FROM cash_movements ORDER BY date, id").fetchall()
    return [_row_to_cash_movement(row) for row in rows]
