import sqlite3
from datetime import date

import pandas as pd
import pytest

from simulator.database import (
    add_asset,
    add_cash_movement,
    add_transaction,
    delete_cash_movement,
    delete_manual_price,
    delete_transaction,
    get_asset_by_isin,
    get_last_price,
    get_last_yahoo_date,
    get_price_history,
    get_price_source,
    get_setting,
    list_assets,
    list_cash_movements,
    list_manual_prices,
    list_transactions,
    save_prices,
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


def make_series(values: dict[str, float]) -> pd.Series:
    return pd.Series(list(values.values()), index=pd.to_datetime(list(values)))


def test_saved_prices_can_be_read_back(conn):
    asset_id = add_asset(conn, make_asset())

    save_prices(conn, asset_id, make_series({"2026-03-02": 10.0, "2026-03-03": 11.0}), "yahoo")

    history = get_price_history(conn, asset_id)
    assert history.tolist() == [10.0, 11.0]
    assert history.index[0] == pd.Timestamp("2026-03-02")
    assert get_last_price(conn, asset_id) == (date(2026, 3, 3), 11.0)


def test_saving_a_price_again_replaces_it(conn):
    asset_id = add_asset(conn, make_asset())
    save_prices(conn, asset_id, make_series({"2026-03-02": 10.0}), "yahoo")

    save_prices(conn, asset_id, make_series({"2026-03-02": 10.5, "2026-03-03": 11.0}), "yahoo")

    assert get_price_history(conn, asset_id).tolist() == [10.5, 11.0]


def test_asset_without_price_has_no_last_price_and_an_empty_history(conn):
    asset_id = add_asset(conn, make_asset())

    assert get_last_price(conn, asset_id) is None
    assert get_price_history(conn, asset_id).empty


def test_manual_price_wins_over_yahoo_on_the_same_day(conn):
    asset_id = add_asset(conn, make_asset())
    save_prices(conn, asset_id, make_series({"2026-03-02": 10.0, "2026-03-03": 11.0}), "yahoo")
    save_prices(conn, asset_id, make_series({"2026-03-03": 12.34}), "manual")

    assert get_last_price(conn, asset_id) == (date(2026, 3, 3), 12.34)
    assert get_price_history(conn, asset_id).tolist() == [10.0, 12.34]


def test_a_newer_yahoo_price_wins_over_an_older_manual_one(conn):
    asset_id = add_asset(conn, make_asset())
    save_prices(conn, asset_id, make_series({"2026-03-02": 99.0}), "manual")
    save_prices(conn, asset_id, make_series({"2026-03-03": 11.0}), "yahoo")

    assert get_last_price(conn, asset_id) == (date(2026, 3, 3), 11.0)


def test_last_price_on_a_date_never_comes_from_a_later_day(conn):
    asset_id = add_asset(conn, make_asset())
    save_prices(conn, asset_id, make_series({"2026-03-02": 10.0, "2026-03-05": 15.0}), "yahoo")

    assert get_last_price(conn, asset_id, date(2026, 3, 4)) == (date(2026, 3, 2), 10.0)
    assert get_last_price(conn, asset_id, date(2026, 3, 5)) == (date(2026, 3, 5), 15.0)
    assert get_last_price(conn, asset_id, date(2026, 3, 1)) is None


def test_price_source_is_manual_when_a_manual_price_exists_that_day(conn):
    asset_id = add_asset(conn, make_asset())
    save_prices(conn, asset_id, make_series({"2026-03-02": 10.0, "2026-03-03": 11.0}), "yahoo")
    save_prices(conn, asset_id, make_series({"2026-03-03": 12.0}), "manual")

    assert get_price_source(conn, asset_id, date(2026, 3, 2)) == "yahoo"
    assert get_price_source(conn, asset_id, date(2026, 3, 3)) == "manual"
    assert get_price_source(conn, asset_id, date(2026, 3, 4)) is None


def test_deleting_a_manual_price_brings_back_the_yahoo_price(conn):
    asset_id = add_asset(conn, make_asset())
    save_prices(conn, asset_id, make_series({"2026-03-03": 11.0}), "yahoo")
    save_prices(conn, asset_id, make_series({"2026-03-03": 12.0}), "manual")
    assert list_manual_prices(conn) == [(asset_id, date(2026, 3, 3), 12.0)]

    delete_manual_price(conn, asset_id, date(2026, 3, 3))

    assert list_manual_prices(conn) == []
    assert get_last_price(conn, asset_id) == (date(2026, 3, 3), 11.0)


def test_changing_the_ticker_drops_the_yahoo_prices_but_keeps_the_manual_ones(conn):
    asset = make_asset()
    asset.id = add_asset(conn, asset)
    save_prices(conn, asset.id, make_series({"2026-03-02": 10.0, "2026-03-03": 11.0}), "yahoo")
    save_prices(conn, asset.id, make_series({"2026-03-03": 12.0}), "manual")

    asset.ticker = "OTHER.PA"
    update_asset(conn, asset)

    assert get_price_history(conn, asset.id).tolist() == [12.0]
    assert get_last_yahoo_date(conn, asset.id) is None


def test_editing_an_asset_without_changing_its_ticker_keeps_its_prices(conn):
    asset = make_asset()
    asset.id = add_asset(conn, asset)
    save_prices(conn, asset.id, make_series({"2026-03-02": 10.0}), "yahoo")

    asset.name = "Renamed fund"
    update_asset(conn, asset)

    assert get_last_yahoo_date(conn, asset.id) == date(2026, 3, 2)
