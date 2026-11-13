import sqlite3
from datetime import date

import pytest

from simulator.database import (
    add_asset,
    add_cash_movement,
    add_transaction,
    delete_cash_movement,
    delete_transaction,
    get_asset_by_isin,
    get_setting,
    list_assets,
    list_cash_movements,
    list_transactions,
    set_setting,
    update_asset,
    update_transaction,
)
from simulator.models import Asset, CashMovement, Transaction


def make_asset(isin="FR0000000001", name="Test ETF World") -> Asset:
    return Asset(id=None, isin=isin, name=name, ticker="TEST.PA", fees=0.002, target_weight=0.75)


def test_added_asset_can_be_read_back(conn):
    asset_id = add_asset(conn, make_asset())

    assets = list_assets(conn)

    assert len(assets) == 1
    assert assets[0] == Asset(
        id=asset_id,
        isin="FR0000000001",
        name="Test ETF World",
        ticker="TEST.PA",
        fees=0.002,
        target_weight=0.75,
    )


def test_asset_can_be_found_by_isin(conn):
    add_asset(conn, make_asset())

    assert get_asset_by_isin(conn, "FR0000000001").name == "Test ETF World"
    assert get_asset_by_isin(conn, "FR9999999999") is None


def test_duplicate_isin_is_refused(conn):
    add_asset(conn, make_asset())

    with pytest.raises(sqlite3.IntegrityError):
        add_asset(conn, make_asset(name="Same ISIN again"))


def test_updated_asset_is_saved(conn):
    asset_id = add_asset(conn, make_asset())
    asset = get_asset_by_isin(conn, "FR0000000001")
    asset.name = "Renamed ETF"
    asset.target_weight = None

    update_asset(conn, asset)

    saved = list_assets(conn)[0]
    assert saved.id == asset_id
    assert saved.name == "Renamed ETF"
    assert saved.target_weight is None


def make_transaction(asset_id: int, **changes) -> Transaction:
    values = dict(
        id=None,
        date=date(2025, 3, 10),
        asset_id=asset_id,
        kind="BUY",
        quantity=2.5,
        unit_price=100.0,
        fees=1.0,
        note="monthly buy",
        source="manual",
    )
    values.update(changes)
    return Transaction(**values)


def test_added_transaction_can_be_read_back(conn):
    asset_id = add_asset(conn, make_asset())
    tx_id = add_transaction(conn, make_transaction(asset_id))

    transactions = list_transactions(conn)

    assert transactions == [make_transaction(asset_id, id=tx_id)]


def test_transactions_are_listed_in_date_order(conn):
    asset_id = add_asset(conn, make_asset())
    add_transaction(conn, make_transaction(asset_id, date=date(2025, 5, 1)))
    add_transaction(conn, make_transaction(asset_id, date=date(2025, 4, 1)))

    dates = [tx.date for tx in list_transactions(conn)]

    assert dates == [date(2025, 4, 1), date(2025, 5, 1)]


def test_updated_transaction_is_saved(conn):
    asset_id = add_asset(conn, make_asset())
    tx_id = add_transaction(conn, make_transaction(asset_id))

    update_transaction(conn, make_transaction(asset_id, id=tx_id, quantity=4.0, kind="SELL"))

    saved = list_transactions(conn)[0]
    assert saved.quantity == 4.0
    assert saved.kind == "SELL"


def test_deleted_transaction_is_gone(conn):
    asset_id = add_asset(conn, make_asset())
    tx_id = add_transaction(conn, make_transaction(asset_id))

    delete_transaction(conn, tx_id)

    assert list_transactions(conn) == []


@pytest.mark.parametrize("field", ["quantity", "unit_price"])
@pytest.mark.parametrize("value", [0, -1])
def test_zero_or_negative_quantity_and_price_are_refused(conn, field, value):
    asset_id = add_asset(conn, make_asset())

    with pytest.raises(sqlite3.IntegrityError):
        add_transaction(conn, make_transaction(asset_id, **{field: value}))


def test_transaction_on_unknown_asset_is_refused(conn):
    with pytest.raises(sqlite3.IntegrityError):
        add_transaction(conn, make_transaction(asset_id=999))


def test_cash_movement_can_be_added_read_and_deleted(conn):
    movement = CashMovement(id=None, date=date(2025, 3, 1), kind="DEPOSIT", amount=200.0)
    movement_id = add_cash_movement(conn, movement)

    assert list_cash_movements(conn) == [
        CashMovement(id=movement_id, date=date(2025, 3, 1), kind="DEPOSIT", amount=200.0)
    ]

    delete_cash_movement(conn, movement_id)

    assert list_cash_movements(conn) == []


def test_cash_movement_amount_must_be_positive(conn):
    movement = CashMovement(id=None, date=date(2025, 3, 1), kind="DEPOSIT", amount=0)

    with pytest.raises(sqlite3.IntegrityError):
        add_cash_movement(conn, movement)


def test_missing_setting_returns_the_default(conn):
    assert get_setting(conn, "ps_rate", "0.186") == "0.186"
    assert get_setting(conn, "ps_rate") is None


def test_setting_is_saved_and_can_be_overwritten(conn):
    set_setting(conn, "ps_rate", "0.172")
    assert get_setting(conn, "ps_rate", "0.186") == "0.172"

    set_setting(conn, "ps_rate", "0.186")
    assert get_setting(conn, "ps_rate") == "0.186"
