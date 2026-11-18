"""Reading trades from CSV files, finding duplicates, and full backups of the database."""

import csv
import io
import json
import sqlite3
from datetime import date, datetime

from simulator.database import (
    get_price_history,
    list_assets,
    list_cash_movements,
    list_plan_names,
    list_transactions,
    load_plan,
    load_tax_settings,
)
from simulator.models import Asset, Transaction

CSV_COLUMNS = ["date", "isin", "kind", "quantity", "unit_price", "fees", "note"]
REQUIRED_COLUMNS = ["date", "isin", "kind", "quantity", "unit_price"]
KINDS = {"BUY": "BUY", "SELL": "SELL", "ACHAT": "BUY", "VENTE": "SELL"}
BACKUP_FORMAT = "ltis-backup"
BACKUP_VERSION = 1
# tables in the order they are filled on a restore: an asset before its trades and prices
BACKUP_TABLES = ["assets", "transactions", "cash_movements", "prices", "settings", "plans"]


def export_transactions(conn: sqlite3.Connection) -> str:
    """All trades as CSV text, oldest first. Assets are identified by their ISIN."""
    isins = {asset.id: asset.isin for asset in list_assets(conn)}
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for tx in list_transactions(conn):
        writer.writerow(
            [
                tx.date.isoformat(),
                isins[tx.asset_id],
                tx.kind,
                tx.quantity,
                tx.unit_price,
                tx.fees,
                tx.note,
            ]
        )
    return output.getvalue()


def _read_text(file) -> str:
    """The text of an uploaded file, a path-like file object or a string."""
    content = file if isinstance(file, (str, bytes)) else file.read()
    if isinstance(content, bytes):
        # utf-8-sig drops the mark that Excel puts at the start of a UTF-8 file
        content = content.decode("utf-8-sig")
    return content.lstrip("﻿")


def _parse_date(text: str) -> date:
    """A date written 2026-01-31 or 31/01/2026."""
    for pattern in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            continue
    raise ValueError(f"unreadable date '{text}' (use 2026-01-31 or 31/01/2026)")


def _parse_number(text: str, name: str) -> float:
    """A number written 1 234,56 or 1234.56.

    Thousands are separated by a space only (also the no-break spaces that Excel writes), and
    the decimal mark is a comma or a dot. So a comma or a dot is always the decimal mark, and a
    number with two of them (1,234.56 or 1.234,56) is refused instead of being guessed.
    """
    cleaned = text.replace(" ", "").replace("\u00a0", "").replace("\u202f", "")
    if cleaned.count(",") + cleaned.count(".") > 1:
        raise ValueError(
            f"{name} '{text}': separate thousands with a space and use one decimal mark, "
            "as in 1 234,56"
        )
    try:
        return float(cleaned.replace(",", "."))
    except ValueError:
        raise ValueError(f"{name} '{text}' is not a number") from None


def read_generic_csv(
    file, assets: list[Asset], today: date | None = None
) -> tuple[list[Transaction], list[str]]:
    """Read trades from a CSV file with the columns of `CSV_COLUMNS` (`fees` and `note` optional).

    The separator can be a comma or a semicolon. Never raises on a bad line: each wrong line
    gives one message with its line number (the header is line 1), and the other lines are still
    read. Returns the trades read, in file order, and the messages.
    """
    today = today or date.today()
    text = _read_text(file)
    first_line = text.split("\n", 1)[0]
    delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    columns = [(name or "").strip().lower() for name in reader.fieldnames or []]
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing:
        return [], [
            "Missing column(s): " + ", ".join(missing) + ". Expected: " + ", ".join(CSV_COLUMNS)
        ]
    reader.fieldnames = columns

    by_isin = {asset.isin: asset for asset in assets}
    transactions, errors = [], []
    for row in reader:
        # the line where the row ends in the file: right even if a note spans two lines
        line = reader.line_num
        values = {key: (value or "").strip() for key, value in row.items() if key}
        if not any(values.values()):
            continue  # empty line
        try:
            transactions.append(_row_to_transaction(values, by_isin, today))
        except ValueError as problem:
            errors.append(f"Line {line}: {problem}.")
    return transactions, errors


def _row_to_transaction(values: dict, by_isin: dict, today: date) -> Transaction:
    day = _parse_date(values["date"])
    if day > today:
        raise ValueError(f"the date {day:%d/%m/%Y} is in the future")
    isin = values["isin"].upper()
    if isin not in by_isin:
        raise ValueError(f"unknown ISIN '{isin}' (create the asset in Settings first)")
    kind = KINDS.get(values["kind"].upper())
    if kind is None:
        raise ValueError(f"type '{values['kind']}' is neither BUY nor SELL")
    quantity = _parse_number(values["quantity"], "quantity")
    unit_price = _parse_number(values["unit_price"], "unit price")
    fees = _parse_number(values.get("fees") or "0", "fees")
    if quantity <= 0:
        raise ValueError("the quantity must be greater than 0")
    if unit_price <= 0:
        raise ValueError("the unit price must be greater than 0")
    if fees < 0:
        raise ValueError("fees cannot be negative")
    return Transaction(
        id=None,
        date=day,
        asset_id=by_isin[isin].id,
        kind=kind,
        quantity=quantity,
        unit_price=unit_price,
        fees=fees,
        note=values.get("note", ""),
        source="import",
    )


def _same_trade_key(tx: Transaction) -> tuple:
    """What makes two trades look like the same one. Rounded: 0.1 + 0.2 must equal 0.3."""
    return (
        tx.date,
        tx.asset_id,
        tx.kind,
        round(tx.quantity, 6),
        round(tx.unit_price, 4),
        round(tx.fees, 2),
    )


def find_duplicates(new: list[Transaction], existing: list[Transaction]) -> list[Transaction]:
    """The trades of `new` that are probably already in `existing`.

    Same date, asset, type, quantity, price and fees (the note does not count). Each existing
    trade matches one new trade only: two identical buys on the same day in the file, with one
    already saved, give one duplicate, not two.
    """
    available: dict[tuple, int] = {}
    for tx in existing:
        key = _same_trade_key(tx)
        available[key] = available.get(key, 0) + 1
    duplicates = []
    for tx in new:
        key = _same_trade_key(tx)
        if available.get(key, 0) > 0:
            available[key] -= 1
            duplicates.append(tx)
    return duplicates


def export_backup(conn: sqlite3.Connection) -> str:
    """Everything in the database as JSON text: assets, trades, deposits, prices, settings, plans.

    The ids are kept, so that a trade still points to its asset after a restore.
    """
    tables = {}
    for table in BACKUP_TABLES:
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()  # names from BACKUP_TABLES only
        tables[table] = [dict(row) for row in rows]
    backup = {
        "format": BACKUP_FORMAT,
        "version": BACKUP_VERSION,
        "created": date.today().isoformat(),
        "tables": tables,
    }
    return json.dumps(backup, ensure_ascii=False, indent=1)


def read_backup(text) -> tuple[dict | None, list[str]]:
    """Check a backup before anything is changed. Returns its tables, or None and the reasons."""
    try:
        backup = json.loads(_read_text(text))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, ["This file is not a backup of this app (it is not readable JSON)."]
    if not isinstance(backup, dict) or backup.get("format") != BACKUP_FORMAT:
        return None, ["This file is not a backup of this app."]
    if backup.get("version") != BACKUP_VERSION:
        return None, [f"Backup version {backup.get('version')} is not supported."]
    tables = backup.get("tables")
    if not isinstance(tables, dict):
        return None, ["The backup has no tables."]
    errors = []
    for table in BACKUP_TABLES:
        rows = tables.get(table)
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            errors.append(f"The table '{table}' is missing or damaged.")
    return (None, errors) if errors else (tables, [])


def _read_everything(conn: sqlite3.Connection) -> None:
    """Read the whole database the way the pages do. Raises if a value cannot be understood."""
    for asset in list_assets(conn):
        get_price_history(conn, asset.id)
    list_transactions(conn)
    list_cash_movements(conn)
    load_tax_settings(conn)
    for name in list_plan_names(conn):
        load_plan(conn, name)


def restore_backup(conn: sqlite3.Connection, tables: dict) -> list[str]:
    """Replace the whole content of the database by the tables of a backup (see `read_backup`).

    All or nothing: if one row is refused by the tables (a negative quantity, a trade without
    its asset...) or cannot be read back by the app (a date that is not a date, a damaged
    plan...), the database is left exactly as it was. Returns the reason, or an empty list.
    """
    columns = {
        table: [row["name"] for row in conn.execute(f"PRAGMA table_info({table})")]
        for table in BACKUP_TABLES
    }
    try:
        # children first: the foreign keys forbid deleting an asset that still has trades
        for table in reversed(BACKUP_TABLES):
            conn.execute(f"DELETE FROM {table}")
        for table in BACKUP_TABLES:
            for row in tables[table]:
                names = [name for name in columns[table] if name in row]
                placeholders = ", ".join("?" for _ in names)
                conn.execute(
                    f"INSERT INTO {table} ({', '.join(names)}) VALUES ({placeholders})",
                    [row[name] for name in names],
                )
        # still inside the same transaction: nothing is saved before this check passes
        _read_everything(conn)
    except (sqlite3.Error, ValueError, TypeError, KeyError) as problem:
        conn.rollback()
        return [f"The backup could not be restored, nothing was changed ({problem})."]
    conn.commit()
    return []


def backup_summary(conn: sqlite3.Connection) -> str:
    """One line saying what the database holds, to show before a backup or a restore."""
    return (
        f"{len(list_assets(conn))} assets, {len(list_transactions(conn))} trades, "
        f"{len(list_cash_movements(conn))} cash movements"
    )
