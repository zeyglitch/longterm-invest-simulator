"""Deterministic projection, month by month. Rates are decimals; see 03-finance-rules, section 2."""

from datetime import date

import numpy as np
import pandas as pd

from simulator.models import TaxSettings
from simulator.taxes import PEA, pea_five_year_date, withdrawal_tax


def net_return(gross: float, fees: float) -> float:
    """Yearly return once the fund's yearly fees are taken."""
    return (1 + gross) * (1 - fees) - 1


def monthly_rate(annual: float) -> float:
    """Monthly rate that compounds to `annual` over 12 months. Not annual / 12."""
    return (1 + annual) ** (1 / 12) - 1


def _apply_flow(value: float, contributions: float, flow: float) -> tuple[float, float, float]:
    """Add a deposit or take a withdrawal. Returns the new value, contributions, and real flow.

    A withdrawal cannot take more than the value. It removes capital and gain in the same
    proportion as the whole pocket, so only its capital part lowers the contributions.
    """
    if flow >= 0:
        return value + flow, contributions + flow, flow
    withdrawal = min(-flow, value)
    _, capital = withdrawal_tax(withdrawal, value, contributions, 0.0)
    return value - withdrawal, contributions - capital, -withdrawal


def _start_date(flows: pd.Series | pd.DataFrame) -> pd.Timestamp:
    """Date of the starting row: the month end just before the first flow."""
    if flows.empty:
        return pd.NaT
    return flows.index[0] - pd.offsets.MonthEnd(1)


def _monthly_parts(flows: pd.Series | pd.DataFrame):
    """Yield (date, deposit, withdrawal) for each month, both amounts positive or zero.

    `flows` is either one net flow per month (positive: deposit, negative: withdrawal), or a
    table with the columns `deposit` and `withdrawal` as built by `plan.build_flow_parts`. The
    table matters when a month has both: the two must not be netted before the rules apply.
    """
    if isinstance(flows, pd.DataFrame):
        yield from zip(flows.index, flows["deposit"], flows["withdrawal"], strict=True)
    else:
        for day, flow in flows.items():
            yield day, max(flow, 0.0), max(-flow, 0.0)


def project(
    start_value: float,
    start_contributions: float,
    flows: pd.Series | pd.DataFrame,
    annual_return: float,
) -> pd.DataFrame:
    """Value of the portfolio month by month, as `date`, `value`, `contributions`, `gains`, `flow`.

    `flows` has one row per month (index: month ends): a net amount, or a deposit and a
    withdrawal (see `_monthly_parts`). Each month the value grows first, then the deposit
    arrives, then the withdrawal leaves. The first row is the start, before any flow. `flow` is
    what really moved, deposit minus withdrawal: a withdrawal is capped at the value.
    """
    rate = monthly_rate(annual_return)
    value, contributions = start_value, start_contributions
    rows = [(_start_date(flows), value, contributions, 0.0)]
    for day, deposit, withdrawal in _monthly_parts(flows):
        value *= 1 + rate
        value, contributions, _ = _apply_flow(value, contributions, deposit)
        value, contributions, taken = _apply_flow(value, contributions, -withdrawal)
        rows.append((day, value, contributions, deposit + taken))
    table = pd.DataFrame(rows, columns=["date", "value", "contributions", "flow"])
    table["gains"] = table["value"] - table["contributions"]
    return table[["date", "value", "contributions", "gains", "flow"]]


def project_with_ceiling(
    start_value: float,
    start_contributions: float,
    flows: pd.Series | pd.DataFrame,
    annual_return: float,
    ceiling: float,
    already_deposited: float,
) -> pd.DataFrame:
    """Like `project`, with a PEA whose deposits are capped and a CTO that takes the surplus.

    `start_value` and `start_contributions` are those of the PEA; the CTO starts empty.
    `already_deposited` is the total of deposits ever made on the PEA: a withdrawal does not give
    capacity back. Withdrawals are taken from the CTO first, then from the PEA. Adds the columns
    `value_pea`, `value_cto`, `contributions_pea`, `contributions_cto`.
    """
    rate = monthly_rate(annual_return)
    pea, pea_paid = start_value, start_contributions
    cto, cto_paid = 0.0, 0.0
    deposited = already_deposited
    rows = [(_start_date(flows), pea, cto, pea_paid, cto_paid, 0.0)]
    for day, deposit, withdrawal in _monthly_parts(flows):
        pea *= 1 + rate
        cto *= 1 + rate
        to_pea = min(deposit, max(0.0, ceiling - deposited))
        deposited += to_pea
        pea, pea_paid, _ = _apply_flow(pea, pea_paid, to_pea)
        cto, cto_paid, _ = _apply_flow(cto, cto_paid, deposit - to_pea)
        cto, cto_paid, from_cto = _apply_flow(cto, cto_paid, -withdrawal)
        pea, pea_paid, from_pea = _apply_flow(pea, pea_paid, -withdrawal - from_cto)
        rows.append((day, pea, cto, pea_paid, cto_paid, deposit + from_cto + from_pea))
    columns = ["date", "value_pea", "value_cto", "contributions_pea", "contributions_cto", "flow"]
    table = pd.DataFrame(rows, columns=columns)
    table["value"] = table["value_pea"] + table["value_cto"]
    table["contributions"] = table["contributions_pea"] + table["contributions_cto"]
    table["gains"] = table["value"] - table["contributions"]
    return table[["date", "value", "contributions", "gains", "flow", *columns[1:5]]]


def project_livret(
    start_value: float, flows: pd.Series | pd.DataFrame, rate: float, ceiling: float
) -> pd.DataFrame:
    """Balance of a Livret A month by month, as `date`, `value`, `contributions`, `gains`,
    `flow`, `not_invested`.

    Interest is added monthly (an approximation: the real one is computed by fortnight and added
    once a year). A deposit is accepted up to the ceiling; the rest is `not_invested`. Interest
    alone can push the balance above the ceiling: no deposit is accepted then.
    """
    monthly = monthly_rate(rate)
    value, contributions = start_value, start_value
    rows = [(_start_date(flows), value, contributions, 0.0, 0.0)]
    for day, deposit, withdrawal in _monthly_parts(flows):
        value *= 1 + monthly
        accepted = max(0.0, min(deposit, ceiling - value))
        value, contributions, _ = _apply_flow(value, contributions, accepted)
        value, contributions, taken = _apply_flow(value, contributions, -withdrawal)
        rows.append((day, value, contributions, accepted + taken, deposit - accepted))
    columns = ["date", "value", "contributions", "flow", "not_invested"]
    table = pd.DataFrame(rows, columns=columns)
    table["gains"] = table["value"] - table["contributions"]
    return table[["date", "value", "contributions", "gains", "flow", "not_invested"]]


def to_real(values: pd.Series, inflation: float) -> pd.Series:
    """Values in today's euros. The position in the series is the number of months elapsed."""
    months = np.arange(len(values))
    return values / (1 + inflation) ** (months / 12)


def fees_cost(
    start_value: float,
    start_contributions: float,
    flows: pd.Series | pd.DataFrame,
    gross: float,
    fees: float,
) -> float:
    """Euros lost to the fund's fees at the end: final value without fees minus with fees."""
    without_fees = project(start_value, start_contributions, flows, gross)
    with_fees = project(start_value, start_contributions, flows, net_return(gross, fees))
    return without_fees["value"].iloc[-1] - with_fees["value"].iloc[-1]


def date_target_reached(projection: pd.DataFrame, target: float) -> date | None:
    """Date of the first month whose value reaches `target`, or None if it never does."""
    reached = projection[projection["value"] >= target]
    if reached.empty or pd.isna(reached["date"].iloc[0]):
        return None
    return reached["date"].iloc[0].date()


def add_net_value(
    projection: pd.DataFrame, envelope: str, open_date: date | None, settings: TaxSettings
) -> pd.DataFrame:
    """Add `tax` and `net_value`: the tax due, and what would be left, if everything were
    withdrawn in that month.

    PEA: social charges only from its fifth birthday, the flat tax before (and when the opening
    date is unknown). What sits in a CTO, including the surplus above the PEA ceiling, always
    bears the flat tax. A pocket in loss owes nothing.
    """
    table = projection.copy()
    flat_tax = settings.ir_rate + settings.ps_rate
    pea_rate = flat_tax
    if envelope == PEA and open_date is not None:
        five_years = pd.Timestamp(pea_five_year_date(open_date))
        pea_rate = np.where(table["date"] >= five_years, settings.ps_rate, flat_tax)

    if "value_pea" in table:
        pea_gain = (table["value_pea"] - table["contributions_pea"]).clip(lower=0)
        cto_gain = (table["value_cto"] - table["contributions_cto"]).clip(lower=0)
    else:
        pea_gain = table["gains"].clip(lower=0)
        cto_gain = 0.0
    table["tax"] = pea_gain * pea_rate + cto_gain * flat_tax
    table["net_value"] = table["value"] - table["tax"]
    return table


def yearly_table(projection: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar year: the last month of each year, with a `year` column first."""
    dated = projection.dropna(subset=["date"])
    last_of_year = dated.groupby(dated["date"].dt.year).tail(1).copy()
    last_of_year.insert(0, "year", last_of_year["date"].dt.year)
    return last_of_year.reset_index(drop=True)
