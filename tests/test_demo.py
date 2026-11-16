import csv
from datetime import date

from simulator.database import (
    get_last_price,
    list_assets,
    list_cash_movements,
    list_plan_names,
    list_transactions,
    load_plan,
    load_tax_settings,
)
from simulator.demo import SAMPLE_DIR, build_demo_db
from simulator.plan import validate_plan


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


def test_demo_has_frozen_prices_for_every_asset():
    conn = build_demo_db()

    for asset in list_assets(conn):
        last = get_last_price(conn, asset.id)
        assert last is not None
        assert last[0] == date(2026, 11, 13)
        assert last[1] > 0


def test_demo_prices_on_trade_days_match_the_trade_prices():
    conn = build_demo_db()

    for tx in list_transactions(conn):
        assert get_last_price(conn, tx.asset_id, tx.date) == (tx.date, tx.unit_price)


def test_demo_has_a_fictional_pea_opening_date():
    conn = build_demo_db()

    assert load_tax_settings(conn).pea_open_date == date(2024, 11, 4)


def test_demo_has_a_valid_example_plan():
    conn = build_demo_db()

    assert list_plan_names(conn) == ["Example plan"]
    assert validate_plan(load_plan(conn, "Example plan")) == []
