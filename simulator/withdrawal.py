"""Withdrawal phase: living on the capital. See 03-finance-rules, section 8."""

import numpy as np
import pandas as pd

from simulator.montecarlo import normal_returns, run_paths
from simulator.projection import monthly_rate
from simulator.taxes import withdrawal_tax

FIXED = "fixed"
INDEXED = "indexed"
RATE = "rate"


def withdrawal_schedule(
    rule: str, amount_or_rate: float, start_value: float, inflation: float, n_months: int
) -> np.ndarray:
    """The amount to withdraw each month, before tax, for one of the three rules.

    `fixed`: the same monthly amount throughout. `indexed`: a monthly amount raised by the
    inflation once a year. `rate`: a yearly share of the starting capital (0.04 for the 4 %
    rule), paid by twelfths the first year, then raised by the inflation like `indexed`.
    """
    if rule == FIXED:
        return np.full(n_months, float(amount_or_rate))
    if rule == INDEXED:
        first = amount_or_rate
    elif rule == RATE:
        first = amount_or_rate * start_value / 12
    else:
        raise ValueError(f"Unknown rule: {rule}")
    years_elapsed = np.arange(n_months) // 12
    return first * (1 + inflation) ** years_elapsed


def simulate_withdrawals(
    start_value: float, contributions: float, schedule, annual_return: float, tax_rate
) -> pd.DataFrame:
    """The capital month by month while the withdrawals of `schedule` are taken, at a fixed
    return.

    Each month the capital grows first, then the withdrawal leaves; it cannot take more than
    what is left. Its tax follows the proportional rule (03-finance-rules 4.2): the withdrawal
    holds capital and gain in the same proportion as the whole. `tax_rate` is one rate, or one
    rate per month. Returns one row per month (the first is the start) with `month`, `value`,
    `contributions`, `withdrawal`, `tax` and `net`. Once the capital is used up, the rows go on
    with zeros.
    """
    schedule = np.asarray(schedule, dtype=float)
    rates = np.broadcast_to(np.asarray(tax_rate, dtype=float), schedule.shape)
    growth = 1 + monthly_rate(annual_return)
    value = float(start_value)
    rows = [(0, value, contributions, 0.0, 0.0, 0.0)]
    for month, (planned, rate) in enumerate(zip(schedule, rates, strict=True), start=1):
        value *= growth
        withdrawal = min(planned, value)
        tax, capital = withdrawal_tax(withdrawal, value, contributions, rate)
        value -= withdrawal
        contributions -= capital
        rows.append((month, value, contributions, withdrawal, tax, withdrawal - tax))
    columns = ["month", "value", "contributions", "withdrawal", "tax", "net"]
    return pd.DataFrame(rows, columns=columns)


def depletion_month(table: pd.DataFrame) -> int | None:
    """Number of the month in which the capital runs out, or None if it lasts."""
    empty = table[table["value"] <= 0]
    return None if empty.empty else int(empty["month"].iloc[0])


def withdrawal_paths(
    start_value: float, schedule, mu: float, sigma: float, n_paths: int, seed: int
) -> np.ndarray:
    """Random paths of the capital under the same withdrawals, shape (n_paths, n_months + 1).

    Same draws as the Monte Carlo of the saving phase: normal monthly log returns of mean `mu`
    and volatility `sigma` a year. A path that runs out stays at 0.
    """
    schedule = np.asarray(schedule, dtype=float)
    returns = normal_returns(len(schedule), mu, sigma, n_paths, seed)
    return run_paths(start_value, np.zeros(len(schedule)), returns, withdrawals=schedule)[0]


def success_rate(
    start_value: float, schedule, mu: float, sigma: float, n_paths: int, seed: int
) -> float:
    """Share of the random paths that still hold money at the end of the schedule."""
    paths = withdrawal_paths(start_value, schedule, mu, sigma, n_paths, seed)
    return float(np.mean(paths[:, -1] > 0))
