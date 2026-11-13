import csv
import sqlite3
from datetime import date

from simulator import config
from simulator.database import (
    add_asset,
    add_cash_movement,
    add_transaction,
    get_asset_by_isin,
    get_connection,
    init_db,
)
from simulator.models import Asset, CashMovement, Transaction

SAMPLE_DIR = config.ROOT_DIR / "sample_data"


def _read_rows(filename: str) -> list[dict]:
    with open(SAMPLE_DIR / filename, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def build_demo_db(path: str = ":memory:") -> sqlite3.Connection:
    """Build the fictional demo portfolio from sample_data/ (in memory unless a path is given)."""
    conn = get_connection(path)
    init_db(conn)

    for row in _read_rows("demo_assets.csv"):
        asset = Asset(
            id=None,
            isin=row["isin"],
            name=row["name"],
            ticker=row["ticker"],
            fees=float(row["fees"]),
            target_weight=float(row["target_weight"]),
        )
        add_asset(conn, asset)

    for row in _read_rows("demo_transactions.csv"):
        tx = Transaction(
            id=None,
            date=date.fromisoformat(row["date"]),
            asset_id=get_asset_by_isin(conn, row["isin"]).id,
            kind=row["kind"],
            quantity=float(row["quantity"]),
            unit_price=float(row["unit_price"]),
            fees=float(row["fees"]),
            note=row["note"],
        )
        add_transaction(conn, tx)

    for row in _read_rows("demo_cash.csv"):
        movement = CashMovement(
            id=None,
            date=date.fromisoformat(row["date"]),
            kind=row["kind"],
            amount=float(row["amount"]),
            note=row["note"],
        )
        add_cash_movement(conn, movement)

    return conn
