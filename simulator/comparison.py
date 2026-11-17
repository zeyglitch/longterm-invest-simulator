"""Envelopes and strategies side by side. See 03-finance-rules, section 6."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from simulator.models import TaxSettings
from simulator.montecarlo import normal_returns, run_paths, worst_falls
from simulator.projection import (
    add_net_value,
    net_return,
    project,
    project_livret,
    project_with_ceiling,
)
from simulator.taxes import CTO, LIVRET, PEA

COLUMNS = ["date", "value", "contributions", "tax", "net_value", "not_invested"]


@dataclass
class Strategy:
    """One way to invest. `annual_return` is gross (for a Livret A: its rate, and `volatility`
    and `fees` are not used). `flows` are the monthly flows from next month on;
    `initial_deposit` is put in today, on top of the starting value."""

    name: str
    envelope: str
    annual_return: float
    volatility: float
    fees: float
    flows: pd.Series | pd.DataFrame
    initial_deposit: float = 0.0


def lump_sum_flows(amount: float, months: pd.DatetimeIndex) -> tuple[float, pd.Series]:
    """The whole `amount` invested today: (initial deposit, monthly flows, all zero)."""
    return amount, pd.Series(0.0, index=months)


def dca_flows(amount: float, n_months: int, months: pd.DatetimeIndex) -> tuple[float, pd.Series]:
    """`amount` spread in equal parts over the first `n_months` months: (0, monthly flows)."""
    flows = pd.Series(0.0, index=months)
    flows.iloc[:n_months] = amount / n_months
    return 0.0, flows


def _project_in(
    envelope: str,
    start_value: float,
    start_contributions: float,
    flows: pd.Series | pd.DataFrame,
    annual_return: float,
    settings: TaxSettings,
    already_deposited: float,
    livret_ceiling: float,
) -> pd.DataFrame:
    """Projection of the same money in one envelope, with its tax and its ceiling.

    PEA: deposits above the ceiling go to a CTO with the same return. Livret A: nothing above
    its ceiling is accepted, and there is no tax. `not_invested` is what the ceiling refused.
    """
    if envelope == LIVRET:
        table = project_livret(
            min(start_value, livret_ceiling), flows, annual_return, livret_ceiling
        )
        table.loc[0, "not_invested"] = max(start_value - livret_ceiling, 0.0)
        table["tax"] = 0.0
        table["net_value"] = table["value"]
        return table[COLUMNS]
    if envelope == PEA:
        table = project_with_ceiling(
            start_value,
            start_contributions,
            flows,
            annual_return,
            settings.pea_ceiling,
            already_deposited,
        )
    elif envelope == CTO:
        table = project(start_value, start_contributions, flows, annual_return)
    else:
        raise ValueError(f"Unknown envelope: {envelope}")
    table = add_net_value(table, envelope, settings.pea_open_date, settings)
    table["not_invested"] = 0.0
    return table[COLUMNS]


def compare_envelopes(
    start_value: float,
    start_contributions: float,
    flows: pd.Series | pd.DataFrame,
    annual_return: float,
    settings: TaxSettings,
    livret_rate: float,
    livret_ceiling: float,
    already_deposited: float | None = None,
) -> pd.DataFrame:
    """The same start and the same flows in a PEA, a CTO and a Livret A, month by month.

    PEA and CTO earn `annual_return` (already net of fees), the Livret A its own rate: the gap
    between PEA and CTO comes from the tax only. One row per month and per envelope, with the
    columns `name`, `date`, `value`, `contributions`, `tax`, `net_value`, `not_invested`.
    `already_deposited` counts towards the PEA ceiling (default: the starting contributions).
    """
    if already_deposited is None:
        already_deposited = start_contributions
    tables = []
    for envelope, rate in ((PEA, annual_return), (CTO, annual_return), (LIVRET, livret_rate)):
        table = _project_in(
            envelope,
            start_value,
            start_contributions,
            flows,
            rate,
            settings,
            already_deposited,
            livret_ceiling,
        )
        table.insert(0, "name", envelope)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def compare_strategies(
    strategies: list[Strategy],
    start_value: float,
    start_contributions: float,
    settings: TaxSettings,
    livret_ceiling: float,
    already_deposited: float | None = None,
) -> pd.DataFrame:
    """Each strategy projected at its own fixed return, from the same starting point.

    Same columns as `compare_envelopes`, `name` being the name of the strategy. The initial
    deposit of a strategy joins the starting value: it is invested today.
    """
    if already_deposited is None:
        already_deposited = start_contributions
    tables = []
    for strategy in strategies:
        rate = strategy.annual_return
        if strategy.envelope != LIVRET:
            rate = net_return(rate, strategy.fees)
        table = _project_in(
            strategy.envelope,
            start_value + strategy.initial_deposit,
            start_contributions + strategy.initial_deposit,
            strategy.flows,
            rate,
            settings,
            already_deposited + strategy.initial_deposit,
            livret_ceiling,
        )
        table.insert(0, "name", strategy.name)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def simulate_strategy(
    strategy: Strategy, start_value: float, start_contributions: float, n_paths: int, seed: int
) -> pd.DataFrame:
    """Random paths of one strategy (not a Livret A): one row per path, with its final `value`
    before tax, its `contributions` and its `worst_fall` along the way.

    Normal monthly returns around the strategy's return after fees. Two strategies run with the
    same seed get the same draws: only their return, volatility and flows differ. The PEA
    ceiling is not applied here.
    """
    flows = strategy.flows
    if isinstance(flows, pd.DataFrame):
        deposits, withdrawals = flows["deposit"].to_numpy(), flows["withdrawal"].to_numpy()
    else:
        deposits, withdrawals = np.maximum(flows, 0.0), np.maximum(-flows, 0.0)
    mean = net_return(strategy.annual_return, strategy.fees)
    returns = normal_returns(len(flows), mean, strategy.volatility, n_paths, seed)
    paths, contributions = run_paths(
        start_value + strategy.initial_deposit,
        deposits,
        returns,
        start_contributions + strategy.initial_deposit,
        withdrawals,
    )
    return pd.DataFrame(
        {
            "value": paths[:, -1],
            "contributions": contributions,
            "worst_fall": worst_falls(returns),
        }
    )
