import csv
import io
import json
import sqlite3
from datetime import date

import pandas as pd
import pytest

from simulator.database import (
    add_asset,
    add_transaction,
    add_transactions,
    get_asset_by_isin,
    get_setting,
    list_assets,
    list_transactions,
    load_plan,
    reset_database,
    save_prices,
    set_setting,
)
from simulator.demo import SAMPLE_DIR, build_demo_db
from simulator.importer import (
    BACKUP_TABLES,
    export_backup,
    export_transactions,
    find_duplicates,
    read_backup,
    read_generic_csv,
    restore_backup,
)
from simulator.models import Asset, Transaction


def test_export_has_a_header_and_one_line_per_trade(conn):
    asset_id = add_asset(conn, Asset(None, "XX0000000001", "Asset A", None))
    add_transaction(
        conn,
        Transaction(None, date(2026, 1, 12), asset_id, "BUY", 2.5, 40.0, 1.0, "second, with comma"),
    )
    add_transaction(conn, Transaction(None, date(2026, 1, 5), asset_id, "BUY", 10, 5.0))

    lines = export_transactions(conn).splitlines()

    assert lines == [
        "date,isin,kind,quantity,unit_price,fees,note",
        "2026-01-05,XX0000000001,BUY,10.0,5.0,0.0,",
        '2026-01-12,XX0000000001,BUY,2.5,40.0,1.0,"second, with comma"',
    ]


def test_export_of_an_empty_database_is_only_the_header(conn):
    assert export_transactions(conn) == "date,isin,kind,quantity,unit_price,fees,note\n"


def test_export_of_the_demo_matches_the_demo_file():
    # same columns as sample_data/demo_transactions.csv, so the values must come back equal
    exported = list(csv.DictReader(io.StringIO(export_transactions(build_demo_db()))))
    with open(SAMPLE_DIR / "demo_transactions.csv", newline="", encoding="utf-8") as file:
        original = list(csv.DictReader(file))

    assert len(exported) == len(original)
    for exported_row, original_row in zip(exported, original, strict=True):
        assert exported_row["date"] == original_row["date"]
        assert exported_row["isin"] == original_row["isin"]
        assert exported_row["kind"] == original_row["kind"]
        assert float(exported_row["quantity"]) == float(original_row["quantity"])
        assert float(exported_row["unit_price"]) == float(original_row["unit_price"])
        assert exported_row["note"] == original_row["note"]


# --- reading a CSV file ---

TODAY = date(2026, 11, 17)


def demo_assets():
    conn = build_demo_db()
    return conn, list_assets(conn)


def test_the_template_is_read_whole():
    # I1
    conn, assets = demo_assets()
    world = get_asset_by_isin(conn, "DEMO00000001")

    with open(SAMPLE_DIR / "import_template.csv", "rb") as file:
        trades, errors = read_generic_csv(file, assets, TODAY)

    assert errors == []
    assert len(trades) == 4
    assert [tx.kind for tx in trades] == ["BUY", "BUY", "BUY", "SELL"]
    assert trades[0].date == date(2026, 1, 5)
    assert trades[2].date == date(2026, 2, 5)  # written 05/02/2026, day first
    assert trades[0].asset_id == world.id
    assert (trades[1].quantity, trades[1].unit_price, trades[1].fees) == (4.5, 52.10, 1.0)
    assert trades[0].note == "monthly buy"
    assert all(tx.source == "import" for tx in trades)


def test_bad_lines_give_one_message_each_and_the_others_are_read():
    # I2
    _, assets = demo_assets()
    text = (
        "date,isin,kind,quantity,unit_price,fees,note\n"
        "2026-01-05,DEMO00000001,BUY,2,100,1,good\n"
        "2026-01-06,DEMO00000001,BUY,-3,100,1,negative\n"
        "31/02/2026,DEMO00000001,BUY,1,100,0,no such day\n"
        "2026-01-07,XX9999999999,BUY,1,100,0,unknown isin\n"
        "2026-01-08,DEMO00000002,SELL,1,50,0,good\n"
    )

    trades, errors = read_generic_csv(text, assets, TODAY)

    assert [tx.note for tx in trades] == ["good", "good"]
    assert len(errors) == 3
    assert errors[0].startswith("Line 3:") and "quantity" in errors[0]
    assert errors[1].startswith("Line 4:") and "date" in errors[1]
    assert errors[2].startswith("Line 5:") and "XX9999999999" in errors[2]


def test_other_mistakes_are_reported_too():
    _, assets = demo_assets()
    text = (
        "date,isin,kind,quantity,unit_price,fees,note\n"
        "2026-01-05,DEMO00000001,GIFT,1,100,0,\n"
        "2026-01-05,DEMO00000001,BUY,one,100,0,\n"
        "2026-01-05,DEMO00000001,BUY,1,0,0,\n"
        "2026-01-05,DEMO00000001,BUY,1,100,-1,\n"
        "2026-12-01,DEMO00000001,BUY,1,100,0,\n"
        ",,,,,,\n"
    )

    trades, errors = read_generic_csv(text, assets, TODAY)

    assert trades == []
    assert len(errors) == 5  # the empty last line is skipped, not an error
    assert "neither BUY nor SELL" in errors[0]
    assert "not a number" in errors[1]
    assert "unit price" in errors[2]
    assert "negative" in errors[3]
    assert "future" in errors[4]


def test_a_file_without_the_needed_columns_is_refused_at_once():
    _, assets = demo_assets()

    trades, errors = read_generic_csv("day,fund,amount\n2026-01-05,X,100\n", assets, TODAY)

    assert trades == []
    assert len(errors) == 1
    assert "date" in errors[0] and "isin" in errors[0]


def test_semicolons_decimal_commas_and_an_excel_mark_are_accepted():
    _, assets = demo_assets()
    text = "﻿Date;ISIN;Kind;Quantity;Unit_price\n05/01/2026;demo00000001;buy;1,5;1 100,25\n"

    trades, errors = read_generic_csv(text.encode("utf-8"), assets, TODAY)

    assert errors == []
    assert (trades[0].quantity, trades[0].unit_price, trades[0].fees) == (1.5, 1100.25, 0.0)
    assert trades[0].kind == "BUY"


# --- duplicates ---


def test_a_file_imported_twice_is_all_duplicates():
    # I3
    conn, assets = demo_assets()
    with open(SAMPLE_DIR / "import_template.csv", "rb") as file:
        trades, _ = read_generic_csv(file, assets, TODAY)
    add_transactions(conn, trades)

    with open(SAMPLE_DIR / "import_template.csv", "rb") as file:
        again, _ = read_generic_csv(file, assets, TODAY)

    assert find_duplicates(again, list_transactions(conn)) == again


def test_each_saved_trade_matches_one_new_trade_only():
    buy = Transaction(None, date(2026, 1, 5), 1, "BUY", 2, 100.0, 1.0)
    other = Transaction(None, date(2026, 1, 5), 1, "BUY", 2, 101.0, 1.0)
    new = [Transaction(None, date(2026, 1, 5), 1, "BUY", 2, 100.0, 1.0, "another note")] * 2

    duplicates = find_duplicates(new + [other], [buy])

    # two identical buys in the file, one already saved: the second is a new trade
    assert duplicates == [new[0]]


def test_rounding_does_not_hide_a_duplicate():
    saved = Transaction(None, date(2026, 1, 5), 1, "BUY", 0.3, 100.0)
    typed = Transaction(None, date(2026, 1, 5), 1, "BUY", 0.1 + 0.2, 100.0)

    assert find_duplicates([typed], [saved]) == [typed]


def test_export_then_import_gives_the_same_trades():
    # I5
    conn, assets = demo_assets()
    before = list_transactions(conn)

    trades, errors = read_generic_csv(export_transactions(conn), assets, TODAY)

    assert errors == []
    fields = ("date", "asset_id", "kind", "quantity", "unit_price", "fees", "note")
    assert [[getattr(tx, f) for f in fields] for tx in trades] == [
        [getattr(tx, f) for f in fields] for tx in before
    ]
    assert find_duplicates(trades, before) == trades


# --- grouped insert ---


def test_a_refused_trade_leaves_the_others_out_too(conn):
    asset_id = add_asset(conn, Asset(None, "XX0000000001", "Asset A", None))
    good = Transaction(None, date(2026, 1, 5), asset_id, "BUY", 1, 10.0)
    bad = Transaction(None, date(2026, 1, 6), asset_id, "BUY", -1, 10.0)

    with pytest.raises(sqlite3.IntegrityError):
        add_transactions(conn, [good, bad])

    assert list_transactions(conn) == []


# --- full backup ---


def test_backup_then_restore_gives_back_the_whole_database():
    source = build_demo_db()
    set_setting(source, "inflation", "0.025")
    add_manual_price(source)
    text = export_backup(source)

    target = build_demo_db()
    reset_database(target)
    tables, errors = read_backup(text)
    assert errors == []
    assert restore_backup(target, tables) == []

    for table in BACKUP_TABLES:
        query = f"SELECT * FROM {table} ORDER BY 1, 2"
        assert [dict(r) for r in target.execute(query)] == [dict(r) for r in source.execute(query)]
    assert load_plan(target, "Example plan") == load_plan(source, "Example plan")
    assert get_setting(target, "inflation") == "0.025"


def add_manual_price(conn):
    prices = pd.Series([123.4], index=pd.to_datetime(["2026-11-14"]))
    save_prices(conn, list_assets(conn)[0].id, prices, "manual")


def test_restore_replaces_what_was_there(conn):
    add_asset(conn, Asset(None, "XX0000000009", "Old asset", None))
    tables, _ = read_backup(export_backup(build_demo_db()))

    assert restore_backup(conn, tables) == []

    assert get_asset_by_isin(conn, "XX0000000009") is None
    assert len(list_assets(conn)) == 3


def test_a_file_that_is_not_a_backup_is_refused():
    assert read_backup("not json")[0] is None
    assert read_backup('{"format": "other"}')[0] is None
    assert "version" in read_backup('{"format": "ltis-backup", "version": 99}')[1][0]
    damaged = json.dumps({"format": "ltis-backup", "version": 1, "tables": {"assets": []}})
    tables, errors = read_backup(damaged)
    assert tables is None
    assert any("transactions" in error for error in errors)


def test_a_backup_with_a_bad_row_leaves_the_database_untouched():
    conn = build_demo_db()
    before = export_backup(conn)
    backup = json.loads(before)
    backup["tables"]["transactions"][0]["quantity"] = -1  # refused by the table
    tables, errors = read_backup(json.dumps(backup))
    assert errors == []

    errors = restore_backup(conn, tables)

    assert len(errors) == 1 and "nothing was changed" in errors[0]
    assert export_backup(conn) == before


@pytest.mark.parametrize(
    ("table", "field", "value"),
    [
        ("transactions", "date", "not a date"),
        ("cash_movements", "date", "2026-13-45"),
        ("plans", "content", "{damaged"),
        ("settings", "value", "not a date"),  # the first setting is the PEA opening date
    ],
)
def test_a_backup_the_app_could_not_read_is_refused(table, field, value):
    conn = build_demo_db()
    before = export_backup(conn)
    backup = json.loads(before)
    backup["tables"][table][0][field] = value
    tables, _ = read_backup(json.dumps(backup))

    assert restore_backup(conn, tables) != []
    assert export_backup(conn) == before


def test_line_numbers_stay_right_after_a_note_on_two_lines():
    _, assets = demo_assets()
    text = (
        "date,isin,kind,quantity,unit_price,fees,note\n"
        '2026-01-05,DEMO00000001,BUY,1,100,0,"first line\nsecond line"\n'
        "2026-01-06,DEMO00000001,BUY,0,100,0,\n"
    )

    trades, errors = read_generic_csv(text, assets, TODAY)

    assert trades[0].note == "first line\nsecond line"
    assert errors == ["Line 4: the quantity must be greater than 0."]


def test_thousands_are_separated_by_a_space_only():
    _, assets = demo_assets()
    text = (
        "date;isin;kind;quantity;unit_price\n"
        "2026-01-05;DEMO00000001;BUY;1,234;1 234,56\n"  # comma: always the decimal mark
        "2026-01-05;DEMO00000001;BUY;2;1\u202f234.5\n"  # narrow no-break space from Excel
        "2026-01-05;DEMO00000001;BUY;1;1.234,56\n"
        "2026-01-05;DEMO00000001;BUY;1;1,234.56\n"
        "2026-01-05;DEMO00000001;BUY;1;1.234.567\n"
    )

    trades, errors = read_generic_csv(text, assets, TODAY)

    assert [(tx.quantity, tx.unit_price) for tx in trades] == [(1.234, 1234.56), (2, 1234.5)]
    assert [error[:7] for error in errors] == ["Line 4:", "Line 5:", "Line 6:"]
    assert all("separate thousands with a space" in error for error in errors)
