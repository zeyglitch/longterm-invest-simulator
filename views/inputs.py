"""Inputs shared by the simulation pages: which plan, and where the simulation starts from."""

import streamlit as st

from simulator.database import (
    get_last_price,
    list_assets,
    list_cash_movements,
    list_plan_names,
    list_transactions,
    load_plan,
)
from simulator.models import Plan
from simulator.portfolio import cash_balance, compute_positions, gross_deposits, total_deposits
from views.common import format_eur, get_conn, get_plan


def choose_plan(key: str) -> Plan:
    """The plan of the Plan page, or one of the saved plans. `key` is the name of the page."""
    current = get_plan()
    # each text is unique: Streamlit finds the chosen option by its text
    choices = {f"{current.name} (as shown in the Plan page)": current}
    for name in list_plan_names(get_conn()):
        choices[f"{name} (saved)"] = None
    label = st.selectbox("Plan", list(choices), key=f"{key}_plan")
    if choices[label] is not None:
        return current
    return load_plan(get_conn(), label.removesuffix(" (saved)"))


def render_portfolio_start() -> tuple[float, float, float]:
    """Value, net deposits and total deposits of the tracked portfolio, with a line saying so.

    The value includes the cash left on the account: the deposits count it too, and leaving it
    out would look like a loss.
    """
    conn = get_conn()
    assets = list_assets(conn)
    prices = {asset.id: get_last_price(conn, asset.id) for asset in assets}
    prices = {asset_id: price for asset_id, price in prices.items() if price is not None}
    transactions = list_transactions(conn)
    positions = compute_positions(assets, transactions, prices)
    cash_movements = list_cash_movements(conn)
    cash = max(cash_balance(transactions, cash_movements), 0.0)
    value = sum(p.value for p in positions if p.value is not None) + cash
    deposits = total_deposits(cash_movements)

    with_cash = f" (including {format_eur(cash)} of uninvested cash)" if cash >= 0.005 else ""
    st.caption(f"Value {format_eur(value)}{with_cash}, deposits {format_eur(deposits)}.")
    unpriced = [p.asset.name for p in positions if p.value is None]
    if unpriced:
        st.warning(
            "No price for " + ", ".join(unpriced) + ": left out of the starting value, although "
            "the deposits that paid for it are counted. Add a price in Settings, or use Custom."
        )
    return value, deposits, gross_deposits(cash_movements)


def render_start(key: str) -> tuple[float, float, float]:
    """Where the simulation starts from: (value, net deposits, deposits counted in the ceiling)."""
    source = st.segmented_control(
        "Starting point",
        ["Current portfolio", "Custom"],
        default="Current portfolio",
        required=True,
        key=f"{key}_start",
    )
    if source == "Current portfolio":
        return render_portfolio_start()
    left, right = st.columns(2)
    value = left.number_input(
        "Starting value (€)", min_value=0.0, step=500.0, key=f"{key}_custom_value"
    )
    deposits = right.number_input(
        "Deposits already made (€)",
        min_value=0.0,
        step=500.0,
        key=f"{key}_custom_deposits",
        help="Used for the tax (the gain is the value minus the deposits) and the PEA ceiling.",
    )
    return value, deposits, deposits
