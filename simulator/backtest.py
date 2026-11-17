"""Replay regular deposits on past prices. See 03-finance-rules, section 7."""

from datetime import date

import numpy as np
import pandas as pd


def monthly_prices(prices: pd.Series) -> pd.Series:
    """The first known price of each month, dated on its real day. `prices` are daily."""
    if prices.empty:
        return prices
    months = prices.index.to_period("M")
    return prices[~months.duplicated()]


def backtest_dca(
    prices: pd.Series,
    monthly_amount: float,
    initial_amount: float = 0.0,
    order_fee: float = 0.0,
) -> pd.DataFrame:
    """Buy for `monthly_amount` at each price of `prices` (one per month), month after month.

    `initial_amount` is one more order at the first price. Each order pays `order_fee`, taken
    from the amount: only the rest buys shares. Returns one row per month with `date`, `price`,
    `shares`, `invested` and `value`, the value being seen just after the order of the month.
    """
    shares, invested = 0.0, 0.0
    rows = []
    for number, (day, price) in enumerate(prices.items()):
        orders = [monthly_amount] + ([initial_amount] if number == 0 else [])
        for amount in orders:
            if amount > 0:
                shares += max(amount - order_fee, 0.0) / price
                invested += amount
        rows.append((day, price, shares, invested, shares * price))
    return pd.DataFrame(rows, columns=["date", "price", "shares", "invested", "value"])


def _value_at_rate(flows: np.ndarray, annual_rate: float) -> float:
    """Final value of monthly deposits that all earn `annual_rate`; the last one earns nothing."""
    months_left = np.arange(len(flows) - 1, -1, -1)
    return float(np.sum(flows * (1 + annual_rate) ** (months_left / 12)))


def annualised_return(flows, final_value: float) -> float:
    """Yearly rate at which the same monthly deposits would have given `final_value`.

    It is an internal rate of return, found by bisection: the value grows with the rate, so the
    search keeps the half of the interval that holds the answer. `flows` has one amount per
    month, the last one being deposited on the day of `final_value`.
    """
    flows = np.asarray(flows, dtype=float)
    if flows.sum() <= 0:
        raise ValueError("At least one deposit is needed.")
    low, high = -0.99, 10.0
    for _ in range(100):
        middle = (low + high) / 2
        if _value_at_rate(flows, middle) < final_value:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def drawdowns(prices: pd.Series) -> pd.Series:
    """How far each price is below the highest price seen before it: 0 at a high, -0.25 for
    25 % below."""
    return prices / prices.cummax() - 1


def max_drawdown(prices: pd.Series) -> tuple[float, date, date]:
    """Worst fall of the price from an earlier high: (fall, day of the high, day of the low).

    The fall is negative: -0.25 for 25 %. Measured on the price, not on the value of the
    portfolio, which the deposits would distort.
    """
    falls = drawdowns(prices)
    low_day = falls.idxmin()
    high_day = prices.loc[:low_day].idxmax()
    return float(falls.loc[low_day]), high_day.date(), low_day.date()
