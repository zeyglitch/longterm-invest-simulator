import csv

from simulator.database import list_assets, list_cash_movements, list_transactions
from simulator.demo import SAMPLE_DIR, build_demo_db


def count_csv_rows(filename: str) -> int:
    with open(SAMPLE_DIR / filename, newline="", encoding="utf-8") as file:
        return len(list(csv.DictReader(file)))


def test_demo_database_contains_every_csv_row():
    conn = build_demo_db()

    assert len(list_assets(conn)) == count_csv_rows("demo_assets.csv") == 3
    assert len(list_transactions(conn)) == count_csv_rows("demo_transactions.csv") == 20
    assert len(list_cash_movements(conn)) == count_csv_rows("demo_cash.csv")


def test_demo_target_weights_add_up_to_100_percent():
    conn = build_demo_db()

    total = sum(asset.target_weight for asset in list_assets(conn))

    assert round(total, 6) == 1.0
