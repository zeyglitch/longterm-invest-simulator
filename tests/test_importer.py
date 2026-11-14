import csv
import io
from datetime import date

from simulator.database import add_asset, add_transaction
from simulator.demo import SAMPLE_DIR, build_demo_db
from simulator.importer import export_transactions
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
