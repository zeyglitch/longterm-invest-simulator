from datetime import date

import pandas as pd
import pytest
from yfinance.exceptions import YFRateLimitError

from simulator import prices
from simulator.database import add_asset, get_last_price, get_price_history, save_prices
from simulator.models import Asset


def yahoo_table(closes: list[float | None], ticker: str = "TEST.PA") -> pd.DataFrame:
    """A table shaped like yf.download: two column levels, one row per day."""
    days = pd.date_range("2026-11-09", periods=len(closes), tz="Europe/Paris")
    columns = pd.MultiIndex.from_product([["Close", "High"], [ticker]])
    return pd.DataFrame({column: closes for column in columns}, index=days, columns=columns)


def fake_download(table):
    return lambda ticker, start: table


def test_history_reads_the_close_column_and_drops_missing_days(monkeypatch):
    monkeypatch.setattr(prices, "_download", fake_download(yahoo_table([10.0, 11.0, None])))

    history = prices.fetch_history("TEST.PA")

    assert history.tolist() == [10.0, 11.0]
    assert history.index[-1] == pd.Timestamp("2026-11-10")  # no time, no timezone


def test_last_price_ignores_a_nan_row_for_the_current_session(monkeypatch):
    monkeypatch.setattr(prices, "_download", fake_download(yahoo_table([10.0, 11.0, None])))

    assert prices.fetch_last_price("TEST.PA") == (date(2026, 11, 10), 11.0)


def test_empty_answer_gives_no_price(monkeypatch):
    monkeypatch.setattr(prices, "_download", fake_download(pd.DataFrame()))

    assert prices.fetch_history("TEST.PA").empty
    assert prices.fetch_last_price("TEST.PA") is None


@pytest.mark.parametrize(
    "error", [YFRateLimitError(), ConnectionError("offline"), ValueError("bad")]
)
def test_network_errors_are_not_raised(monkeypatch, error):
    def failing_download(ticker, start):
        raise error

    monkeypatch.setattr(prices, "_download", failing_download)

    assert prices.fetch_history("TEST.PA").empty
    assert prices.fetch_last_price("TEST.PA") is None


def add_test_asset(conn, isin="XX0000000001", ticker="TEST.PA") -> int:
    return add_asset(conn, Asset(None, isin, f"Fund {isin[-1]}", ticker))


def series(values: dict[str, float]) -> pd.Series:
    return pd.Series(list(values.values()), index=pd.to_datetime(list(values)))


def test_refresh_stores_the_downloaded_prices(conn, monkeypatch):
    asset_id = add_test_asset(conn)
    monkeypatch.setattr(prices, "fetch_history", lambda ticker, start: series({"2026-03-02": 10.0}))

    failures = prices.refresh_prices(conn)

    assert failures == []
    assert get_last_price(conn, asset_id) == (date(2026, 3, 2), 10.0)


def test_refresh_without_answer_keeps_old_prices_and_names_the_asset(conn, monkeypatch):
    asset_id = add_test_asset(conn)
    save_prices(conn, asset_id, series({"2026-03-02": 10.0}), "yahoo")
    monkeypatch.setattr(prices, "fetch_history", lambda ticker, start: pd.Series(dtype=float))

    failures = prices.refresh_prices(conn)

    assert failures == ["Fund 1"]
    assert get_price_history(conn, asset_id).tolist() == [10.0]


def test_refresh_reports_an_asset_without_ticker_and_still_updates_the_others(conn, monkeypatch):
    add_test_asset(conn, "XX0000000001", ticker=None)
    other_id = add_test_asset(conn, "XX0000000002")
    monkeypatch.setattr(prices, "fetch_history", lambda ticker, start: series({"2026-03-02": 5.0}))

    failures = prices.refresh_prices(conn)

    assert failures == ["Fund 1"]
    assert get_last_price(conn, other_id) == (date(2026, 3, 2), 5.0)


def test_refresh_asks_for_the_whole_history_first_then_only_the_recent_days(conn, monkeypatch):
    asset_id = add_test_asset(conn)
    starts = []

    def record_start(ticker, start):
        starts.append(start)
        return series({"2026-03-10": 10.0})

    monkeypatch.setattr(prices, "fetch_history", record_start)

    prices.refresh_prices(conn)
    prices.refresh_prices(conn)

    assert get_last_price(conn, asset_id) == (date(2026, 3, 10), 10.0)
    assert starts == [None, date(2026, 3, 3)]  # seven days before the last stored price


def test_latest_price_date_is_the_most_recent_of_all_assets(conn):
    first = add_test_asset(conn, "XX0000000001")
    second = add_test_asset(conn, "XX0000000002")
    assert prices.latest_price_date(conn) is None

    save_prices(conn, first, series({"2026-03-02": 10.0}), "yahoo")
    save_prices(conn, second, series({"2026-03-05": 20.0}), "yahoo")

    assert prices.latest_price_date(conn) == date(2026, 3, 5)


def test_ticker_search_puts_the_paris_listing_first_and_reads_the_names(monkeypatch):
    quotes = [
        {"symbol": "FAKE.SG", "shortname": "Fake Fund", "exchDisp": "Stuttgart", "exchange": "STU"},
        {
            "symbol": "FAKE.PA",
            "longname": "Fake Fund Long Name",
            "shortname": "Fake Fund",
            "exchDisp": "Paris",
            "exchange": "PAR",
        },
        {"shortname": "Result without symbol"},
    ]
    monkeypatch.setattr(prices, "_search", lambda isin: quotes)

    found = prices.find_ticker("XX0000000001")

    assert [result["ticker"] for result in found] == ["FAKE.PA", "FAKE.SG"]
    assert found[0]["name"] == "Fake Fund Long Name"
    assert found[0]["exchange"] == "Paris"
    assert found[1]["name"] == "Fake Fund"


@pytest.mark.parametrize("error", [YFRateLimitError(), ConnectionError("offline")])
def test_ticker_search_without_answer_is_empty(monkeypatch, error):
    def failing_search(isin):
        raise error

    monkeypatch.setattr(prices, "_search", failing_search)

    assert prices.find_ticker("XX0000000001") == []
