"""Random trajectories of the portfolio. See 03-finance-rules, section 5.

All the paths are computed together with NumPy: one row per path, one column per month.
"""

import numpy as np
import pandas as pd

from simulator.prices import monthly_returns

LEVELS = (10, 25, 50, 75, 90)


def normal_returns(n_months: int, mu: float, sigma: float, n_paths: int, seed: int) -> np.ndarray:
    """Monthly log returns drawn from a normal law, shape (n_paths, n_months).

    `mu` is the mean yearly return and `sigma` the yearly volatility. The drift removes
    sigma² / 2 so that the average of the paths follows the projection at the rate `mu`.
    """
    rng = np.random.default_rng(seed)
    drift = (np.log(1 + mu) - sigma**2 / 2) / 12
    return rng.normal(drift, sigma / np.sqrt(12), size=(n_paths, n_months))


def bootstrap_returns(n_months: int, past_returns, n_paths: int, seed: int) -> np.ndarray:
    """Monthly log returns drawn at random, with replacement, among those of the past."""
    rng = np.random.default_rng(seed)
    return rng.choice(np.asarray(past_returns, dtype=float), size=(n_paths, n_months))


def run_paths(
    start_value: float,
    flows,
    log_returns: np.ndarray,
    start_contributions: float = 0.0,
    withdrawals=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the returns and the flows month by month, for all the paths at once.

    `flows` is one net flow per month, or one row of flows per path. When `withdrawals` is
    given, `flows` holds the deposits and `withdrawals` the amounts taken out (both positive),
    so that a month can have both. Returns the values, shape (n_paths, n_months + 1), and the
    net deposits left in each path at the end. Same rules as the deterministic projection:
    growth first, then the deposit, then the withdrawal, which is capped at the value and
    removes capital and gain in proportion.
    """
    n_paths, n_months = log_returns.shape
    flows = np.broadcast_to(np.asarray(flows, dtype=float), (n_paths, n_months))
    if withdrawals is None:
        deposits, withdrawals = np.maximum(flows, 0.0), np.maximum(-flows, 0.0)
    else:
        deposits = flows
        withdrawals = np.broadcast_to(np.asarray(withdrawals, dtype=float), (n_paths, n_months))
    growth = np.exp(log_returns)
    paths = np.empty((n_paths, n_months + 1))
    paths[:, 0] = start_value
    value = np.full(n_paths, float(start_value))
    contributions = np.full(n_paths, float(start_contributions))
    for month in range(n_months):
        deposit = deposits[:, month]
        value = value * growth[:, month] + deposit
        contributions = contributions + deposit
        withdrawal = np.minimum(withdrawals[:, month], value)
        # share of the value that is gain; 0 on an empty or losing path
        gain_share = np.divide(
            np.maximum(value - contributions, 0.0), value, out=np.zeros(n_paths), where=value > 0
        )
        contributions = contributions - withdrawal * (1 - gain_share)
        value = value - withdrawal
        paths[:, month + 1] = value
    return paths, contributions


def simulate_normal(
    start_value: float, flows, mu: float, sigma: float, n_paths: int, seed: int
) -> np.ndarray:
    """Paths with normal monthly log returns, shape (n_paths, n_months + 1)."""
    n_months = np.asarray(flows).shape[-1]
    returns = normal_returns(n_months, mu, sigma, n_paths, seed)
    return run_paths(start_value, flows, returns)[0]


def simulate_bootstrap(
    start_value: float, flows, past_returns, n_paths: int, seed: int
) -> np.ndarray:
    """Paths whose monthly returns are drawn among past ones, shape (n_paths, n_months + 1)."""
    n_months = np.asarray(flows).shape[-1]
    returns = bootstrap_returns(n_months, past_returns, n_paths, seed)
    return run_paths(start_value, flows, returns)[0]


def random_unemployment(
    flows, n_paths: int, annual_prob: float, duration_months: int, seed: int
) -> np.ndarray:
    """One row of flows per path, with deposits set to 0 during random unemployment periods.

    Each year of each path has the probability `annual_prob` of starting a period, in a month
    drawn at random in that year. Planned withdrawals are never changed.
    """
    flows = np.asarray(flows, dtype=float)
    n_months = len(flows)
    n_years = -(-n_months // 12)
    rng = np.random.default_rng(seed)
    happens = rng.random((n_paths, n_years)) < annual_prob
    first_month = rng.integers(0, 12, size=(n_paths, n_years)) + 12 * np.arange(n_years)

    months = np.arange(n_months)
    unemployed = np.zeros((n_paths, n_months), dtype=bool)
    for year in range(n_years):
        start = first_month[:, year, None]
        in_period = (months >= start) & (months < start + duration_months)
        unemployed |= happens[:, year, None] & in_period
    return np.where(unemployed & (flows > 0), 0.0, flows)


def percentiles(paths: np.ndarray, levels: tuple = LEVELS) -> pd.DataFrame:
    """Percentiles of the value at each month: one row per month, one column per level."""
    return pd.DataFrame(np.percentile(paths, levels, axis=0).T, columns=list(levels))


def probability_above(final_values, target: float) -> float:
    """Share of the paths whose final value reaches `target`."""
    return float(np.mean(np.asarray(final_values) >= target))


def net_final_values(
    final_values: np.ndarray, contributions: np.ndarray, rate: float
) -> np.ndarray:
    """What is left of each final value once the gain is taxed at `rate` (no tax on a loss)."""
    return final_values - np.maximum(final_values - contributions, 0.0) * rate


def estimate_mu_sigma(prices: pd.Series) -> tuple[float, float]:
    """Mean yearly return and yearly volatility seen in a history of prices.

    Uses the log returns between month ends. Needs at least three months of prices.
    """
    returns = monthly_returns(prices)
    if len(returns) < 2:
        raise ValueError("At least three months of prices are needed.")
    sigma = float(returns.std() * np.sqrt(12))
    mu = float(np.exp(returns.mean() * 12 + sigma**2 / 2) - 1)
    return mu, sigma
