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


def _flow_arrays(flows, withdrawals, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Deposits and withdrawals of each month of each path, both positive or zero."""
    flows = np.broadcast_to(np.asarray(flows, dtype=float), shape)
    if withdrawals is None:
        return np.maximum(flows, 0.0), np.maximum(-flows, 0.0)
    return flows, np.broadcast_to(np.asarray(withdrawals, dtype=float), shape)


def _withdraw(value: np.ndarray, contributions: np.ndarray, amount: np.ndarray):
    """Take `amount` out of each path, at most its value. Returns the new value, the new
    contributions and what was really taken. Capital and gain leave in proportion."""
    taken = np.minimum(amount, value)
    # share of the value that is gain; 0 on an empty or losing path
    gain_share = np.divide(
        np.maximum(value - contributions, 0.0), value, out=np.zeros(len(value)), where=value > 0
    )
    return value - taken, contributions - taken * (1 - gain_share), taken


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
    deposits, withdrawals = _flow_arrays(flows, withdrawals, log_returns.shape)
    growth = np.exp(log_returns)
    paths = np.empty((n_paths, n_months + 1))
    paths[:, 0] = start_value
    value = np.full(n_paths, float(start_value))
    contributions = np.full(n_paths, float(start_contributions))
    for month in range(n_months):
        value = value * growth[:, month] + deposits[:, month]
        contributions = contributions + deposits[:, month]
        value, contributions, _ = _withdraw(value, contributions, withdrawals[:, month])
        paths[:, month + 1] = value
    return paths, contributions


def run_paths_with_ceiling(
    start_value: float,
    flows,
    log_returns: np.ndarray,
    ceiling: float,
    already_deposited: float,
    start_contributions: float = 0.0,
    withdrawals=None,
    cto_extra_return: float = 0.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Like `run_paths`, with a PEA whose deposits are capped and a CTO that takes the surplus.

    Same rules as `projection.project_with_ceiling`: the start is in the PEA, a withdrawal gives
    no capacity back and empties the CTO before the PEA. The CTO gets the same random returns as
    the PEA, plus `cto_extra_return` of log return each month (0: same returns). Returns the
    total values, shape (n_paths, n_months + 1), and one row per path with the final
    `value_pea`, `value_cto`, `contributions_pea` and `contributions_cto`.
    """
    n_paths, n_months = log_returns.shape
    deposits, withdrawals = _flow_arrays(flows, withdrawals, log_returns.shape)
    growth = np.exp(log_returns)
    cto_growth = np.exp(cto_extra_return)
    paths = np.empty((n_paths, n_months + 1))
    paths[:, 0] = start_value
    pea = np.full(n_paths, float(start_value))
    pea_paid = np.full(n_paths, float(start_contributions))
    cto, cto_paid = np.zeros(n_paths), np.zeros(n_paths)
    deposited = np.full(n_paths, float(already_deposited))
    for month in range(n_months):
        to_pea = np.minimum(deposits[:, month], np.maximum(ceiling - deposited, 0.0))
        to_cto = deposits[:, month] - to_pea
        deposited = deposited + to_pea
        pea = pea * growth[:, month] + to_pea
        cto = cto * growth[:, month] * cto_growth + to_cto
        pea_paid, cto_paid = pea_paid + to_pea, cto_paid + to_cto
        cto, cto_paid, from_cto = _withdraw(cto, cto_paid, withdrawals[:, month])
        pea, pea_paid, _ = _withdraw(pea, pea_paid, withdrawals[:, month] - from_cto)
        paths[:, month + 1] = pea + cto
    final = pd.DataFrame(
        {
            "value_pea": pea,
            "value_cto": cto,
            "contributions_pea": pea_paid,
            "contributions_cto": cto_paid,
        }
    )
    return paths, final


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


def worst_falls(log_returns: np.ndarray) -> np.ndarray:
    """Largest fall of each path from an earlier high, as a share: 0.35 for a fall of 35 %.

    Measured on the returns alone, like the price of one share of a fund: on the value of the
    portfolio, the monthly deposits would hide part of each fall.
    """
    n_paths = log_returns.shape[0]
    level = np.hstack([np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)])
    high = np.maximum.accumulate(level, axis=1)
    return 1 - np.exp((level - high).min(axis=1))


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
