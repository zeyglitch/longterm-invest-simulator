import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

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
    # streamlit reruns the script in a new thread each time, with the same connection
    conn = sqlite3.connect(path, check_same_thread=False)
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
    """Save the changes of an asset. A new ticker drops its Yahoo prices: they were another's."""
    old = conn.execute("SELECT ticker FROM assets WHERE id = ?", (asset.id,)).fetchone()
    if old is not None and old["ticker"] != asset.ticker:
        # the next refresh downloads the whole history of the new ticker; manual prices stay
        conn.execute("DELETE FROM prices WHERE asset_id = ? AND source = 'yahoo'", (asset.id,))
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


def get_setting(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    """Return the stored text for `key`, or `default` if it was never set."""
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Store `value` as text, replacing any previous value for `key`."""
    conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )
    conn.commit()


def save_prices(conn: sqlite3.Connection, asset_id: int, prices: pd.Series, source: str) -> None:
    """Store a series of closing prices (index = dates). A price already stored is replaced."""
    rows = [
        (asset_id, day.date().isoformat(), float(close), source) for day, close in prices.items()
    ]
    conn.executemany(
        "INSERT INTO prices (asset_id, date, close, source) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(asset_id, date, source) DO UPDATE SET close = excluded.close",
        rows,
    )
    conn.commit()


def get_last_price(
    conn: sqlite3.Connection, asset_id: int, on_date: date | None = None
) -> tuple[date, float] | None:
    """Latest price on or before `on_date` (default: any date) as (date, close), or None.

    On a day that has both a manual and a Yahoo price, the manual one is returned.
    """
    limit = (on_date or date.max).isoformat()
    row = conn.execute(
        "SELECT date, close FROM prices WHERE asset_id = ? AND date <= ? "
        "ORDER BY date DESC, CASE source WHEN 'manual' THEN 0 ELSE 1 END LIMIT 1",
        (asset_id, limit),
    ).fetchone()
    return (date.fromisoformat(row["date"]), row["close"]) if row else None


def get_last_yahoo_date(conn: sqlite3.Connection, asset_id: int) -> date | None:
    """Date of the latest price downloaded from Yahoo, or None if none was ever downloaded."""
    row = conn.execute(
        "SELECT MAX(date) AS date FROM prices WHERE asset_id = ? AND source = 'yahoo'",
        (asset_id,),
    ).fetchone()
    return date.fromisoformat(row["date"]) if row["date"] else None


def get_price_history(conn: sqlite3.Connection, asset_id: int) -> pd.Series:
    """One price per day, oldest first. The manual price wins over the Yahoo one on a same day."""
    rows = conn.execute(
        "SELECT date, close FROM prices WHERE asset_id = ? "
        "ORDER BY date, CASE source WHEN 'manual' THEN 1 ELSE 0 END",
        (asset_id,),
    ).fetchall()
    # rows of a same day come with the manual one last, so it replaces the Yahoo one
    closes = {row["date"]: row["close"] for row in rows}
    return pd.Series(list(closes.values()), index=pd.to_datetime(list(closes)), dtype=float)


def get_price_source(conn: sqlite3.Connection, asset_id: int, day: date) -> str | None:
    """'manual' or 'yahoo': where the price of that day comes from, or None if there is none."""
    row = conn.execute(
        "SELECT source FROM prices WHERE asset_id = ? AND date = ? "
        "ORDER BY CASE source WHEN 'manual' THEN 0 ELSE 1 END LIMIT 1",
        (asset_id, day.isoformat()),
    ).fetchone()
    return row["source"] if row else None


def list_manual_prices(conn: sqlite3.Connection) -> list[tuple[int, date, float]]:
    """Prices typed by hand as (asset id, date, price), most recent first."""
    rows = conn.execute(
        "SELECT asset_id, date, close FROM prices WHERE source = 'manual' ORDER BY date DESC"
    ).fetchall()
    return [(row["asset_id"], date.fromisoformat(row["date"]), row["close"]) for row in rows]


def delete_manual_price(conn: sqlite3.Connection, asset_id: int, day: date) -> None:
    """Remove a hand-typed price: the Yahoo price of that day, if any, applies again."""
    conn.execute(
        "DELETE FROM prices WHERE asset_id = ? AND date = ? AND source = 'manual'",
        (asset_id, day.isoformat()),
    )
    conn.commit()
