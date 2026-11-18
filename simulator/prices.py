"""Market prices from Yahoo Finance. Network functions never raise: they return None or empty."""

import sqlite3
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFException

from simulator.database import get_last_price, get_last_yahoo_date, list_assets, save_prices

# a few days back is enough to find the last close, even after a long weekend
LAST_PRICE_WINDOW_DAYS = 10
# when updating, go back a few days to pick up prices that Yahoo corrected after publishing
REFRESH_OVERLAP_DAYS = 7


def _download(ticker: str, start: date | None) -> pd.DataFrame:
    """Daily prices adjusted for dividends and splits, as returned by yfinance."""
    if start is None:  # yfinance would otherwise return only the last month
        return yf.download(ticker, period="max", auto_adjust=True, progress=False)
    return yf.download(ticker, start=start, auto_adjust=True, progress=False)


def _last_session_price(ticker: str) -> float | None:
    """Latest price Yahoo shows for a ticker, or None. Never raises.

    Yahoo often leaves the close of the latest session empty in the daily history, for hours
    after the market closed, while its quote already shows the price.
    """
    try:
        price = yf.Ticker(ticker).fast_info["lastPrice"]
    except (YFException, OSError, ValueError, KeyError, TypeError):
        return None
    return float(price) if price and price > 0 else None


def _search(isin: str) -> list[dict]:
    """Raw Yahoo search results for a text such as an ISIN."""
    return yf.Search(isin).quotes


def find_ticker(isin: str) -> list[dict]:
    """Yahoo symbols matching an ISIN: dicts with ticker, name and exchange, Paris first.

    The user must confirm one: an ISIN can be listed on several exchanges. Empty if Yahoo does
    not answer or knows nothing about it.
    """
    try:
        quotes = _search(isin)
    # same rule as the price downloads: a network problem is an empty answer, not a crash
    except (YFException, OSError, ValueError):
        return []
    found = [
        {
            "ticker": quote["symbol"],
            "name": quote.get("longname") or quote.get("shortname", ""),
            "exchange": quote.get("exchDisp", ""),
            "exchange_code": quote.get("exchange", ""),
        }
        for quote in quotes
        if quote.get("symbol")
    ]
    # euro-denominated Paris listings come first: they are the ones eligible for a PEA
    return sorted(found, key=lambda result: result["exchange_code"] != "PAR")


def fetch_history(ticker: str, start: date | None = None) -> pd.Series:
    """Daily closing prices of a ticker, oldest first. Empty if Yahoo does not answer."""
    try:
        table = _download(ticker, start)
    # no network, unknown symbol, rate limit or unreadable answer: the caller keeps its old prices
    except (YFException, OSError, ValueError):
        return pd.Series(dtype=float)
    # an answer without closing prices is treated like no answer
    if table is None or table.empty or "Close" not in table.columns:
        return pd.Series(dtype=float)

    closes = table["Close"]
    if isinstance(closes, pd.DataFrame):  # yfinance puts the ticker as a second column level
        closes = closes.iloc[:, 0]
    # the row of the latest session can have no close yet: take the price of the quote
    # instead; the next refresh goes back a few days and replaces it with the real close
    if pd.isna(closes.iloc[-1]):
        closes = closes.copy()
        closes.iloc[-1] = _last_session_price(ticker)
    closes = closes.dropna()
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    return closes.astype(float).rename(ticker)


def fetch_last_price(ticker: str) -> tuple[date, float] | None:
    """Date and value of the latest close, or None if Yahoo does not answer."""
    start = date.today() - timedelta(days=LAST_PRICE_WINDOW_DAYS)
    history = fetch_history(ticker, start)
    if history.empty:
        return None
    return history.index[-1].date(), float(history.iloc[-1])


def refresh_prices(conn: sqlite3.Connection) -> list[str]:
    """Download the missing prices of every asset into the database.

    Returns the names of the assets that could not be updated (no ticker, or Yahoo did not
    answer). Prices already stored are kept in that case.
    """
    failures = []
    for asset in list_assets(conn):
        if not asset.ticker:
            failures.append(asset.name)
            continue
        # a price typed by hand does not count: the Yahoo history may still be missing
        last = get_last_yahoo_date(conn, asset.id)
        start = None if last is None else last - timedelta(days=REFRESH_OVERLAP_DAYS)
        history = fetch_history(asset.ticker, start)
        if history.empty:
            failures.append(asset.name)
            continue
        save_prices(conn, asset.id, history, "yahoo")
    return failures


def latest_price_date(conn: sqlite3.Connection) -> date | None:
    """Date of the most recent price stored for any asset, or None if there is none."""
    latest = [get_last_price(conn, asset.id) for asset in list_assets(conn)]
    dates = [found[0] for found in latest if found is not None]
    return max(dates) if dates else None


def monthly_returns(prices: pd.Series) -> pd.Series:
    """Log returns from one month end to the next: ln(P_t / P_t-1). Empty without prices."""
    if prices.empty:  # what fetch_history returns when Yahoo does not answer
        return pd.Series(dtype=float)
    month_ends = prices.resample("ME").last().dropna()
    # the month of the last price is not over if a business day is left in it: its return
    # would cover a part of a month only, so it is left out
    last = prices.index[-1]
    if (last + pd.offsets.BDay(1)).month == last.month:
        month_ends = month_ends.iloc[:-1]
    return np.log(month_ends / month_ends.shift(1)).dropna()
