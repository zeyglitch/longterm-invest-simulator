import sqlite3

import pytest

from simulator.database import (
    add_asset,
    get_asset_by_isin,
    list_assets,
    update_asset,
)
from simulator.models import Asset


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
