from datetime import date, timedelta

import pandas as pd
import streamlit as st

from simulator import charts
from simulator.database import (
    get_last_price,
    get_price_history,
    list_assets,
    list_cash_movements,
    list_transactions,
)
from simulator.models import Asset, Position
from simulator.portfolio import compute_positions, portfolio_history, total_deposits
from simulator.prices import latest_price_date
from views.common import format_eur, format_pct, format_quantity, get_conn

PERIOD_DAYS = {"3 months": 91, "1 year": 365}


def format_gap(weight: float | None, target: float | None) -> str:
    """Distance between the real and the target weight, in percentage points."""
    if weight is None or target is None:
        return "—"
    return f"{(weight - target) * 100:+.1f} pts"


def render_empty_state(has_assets: bool) -> None:
    st.info(
        "Welcome. Your portfolio is empty: enter your first purchase to see where you stand.",
        icon=":material/waving_hand:",
    )
    if not has_assets:
        st.page_link("views/settings.py", label="1. Add your assets", icon=":material/settings:")
    st.page_link(
        "views/transactions.py", label="Enter a transaction", icon=":material/receipt_long:"
    )


def render_indicators(positions: list[Position], deposits: float) -> None:
    priced = [p for p in positions if p.value is not None]
    invested = sum(p.invested for p in positions)
    value = sum(p.value for p in priced)
    # the gain only compares what has a price with what it cost
    cost_of_priced = sum(p.invested for p in priced)
    gain = value - cost_of_priced
    gain_pct = gain / cost_of_priced if cost_of_priced > 0 else None

    columns = st.columns(4)
    columns[0].metric(
        "Invested", format_eur(invested), help="Cost of the positions you hold, fees included."
    )
    columns[1].metric("Current value", format_eur(value))
    columns[2].metric(
        "Unrealised gain",
        format_eur(gain),
        delta=format_pct(gain_pct) if gain_pct is not None else None,
        help="Current value minus cost, on the positions that are still open.",
    )
    columns[3].metric(
        "Total deposits",
        format_eur(deposits),
        help="Money put on the account: deposits minus withdrawals.",
    )

    unpriced = [p.asset.name for p in positions if p.value is None]
    if unpriced:
        st.warning(
            "No price for " + ", ".join(unpriced) + ": left out of the value and the gain. "
            "Refresh prices, or enter one by hand in Settings."
        )
        st.page_link("views/settings.py", label="Go to Settings", icon=":material/settings:")


def positions_table(positions: list[Position]) -> pd.DataFrame:
    """One row per open position, then a total row. Everything is formatted as text."""
    rows = [
        {
            "Asset": p.asset.name,
            "ISIN": p.asset.isin,
            "Quantity": format_quantity(p.quantity),
            "Avg cost": format_eur(p.avg_cost),
            "Last price": format_eur(p.last_price),
            "Price date": f"{p.price_date:%d/%m/%Y}" if p.price_date else "—",
            "Invested": format_eur(p.invested),
            "Value": format_eur(p.value),
            "Gain": format_eur(p.gain),
            "Gain %": format_pct(p.gain_pct),
            "Weight": format_pct(p.weight),
            "Target": format_pct(p.asset.target_weight),
            "Gap": format_gap(p.weight, p.asset.target_weight),
        }
        for p in positions
    ]
    priced = [p for p in positions if p.value is not None]
    value = sum(p.value for p in priced)
    cost_of_priced = sum(p.invested for p in priced)
    gain = value - cost_of_priced
    rows.append(
        {
            "Asset": "Total",
            "Invested": format_eur(sum(p.invested for p in positions)),
            "Value": format_eur(value),
            "Gain": format_eur(gain),
            "Gain %": format_pct(gain / cost_of_priced if cost_of_priced > 0 else None),
            "Weight": format_pct(1.0 if value > 0 else None),
        }
    )
    return pd.DataFrame(rows).fillna("")


def allocation_table(assets: list[Asset], positions: list[Position]) -> pd.DataFrame:
    """Real and target weight of each asset, for the allocation chart."""
    weights = {p.asset.id: p.weight for p in positions}
    return pd.DataFrame(
        {
            "name": [asset.name for asset in assets],
            "weight": [weights.get(asset.id) for asset in assets],
            "target_weight": [asset.target_weight for asset in assets],
        }
    )


def render_allocation(assets: list[Asset], positions: list[Position]) -> None:
    st.subheader("Allocation")
    table = allocation_table(assets, positions)
    if table["weight"].fillna(0).sum() == 0:
        st.caption("No priced position yet: the allocation will appear once prices are loaded.")
        return
    st.plotly_chart(charts.allocation_chart(table), width="stretch")
    if table["target_weight"].isna().all():
        st.caption("Set target weights in Settings to compare them with your real weights.")


def render_history(assets: list[Asset], transactions, end: date) -> None:
    st.subheader("History")
    period = st.segmented_control(
        "Period", ["3 months", "1 year", "All"], default="All", key="history_period"
    )
    prices = {asset.id: get_price_history(get_conn(), asset.id) for asset in assets}
    history = portfolio_history(transactions, prices, end=end)
    if period in PERIOD_DAYS:
        since = pd.Timestamp(end - timedelta(days=PERIOD_DAYS[period]))
        history = history[history["date"] >= since]
    if len(history) < 2:
        st.caption("Not enough history to draw a curve yet.")
        return
    st.plotly_chart(charts.history_chart(history), width="stretch")
    st.caption(
        "One point per week. Value = shares held × last known price. "
        "Before a price exists, the price you paid is used."
    )


st.title("Portfolio")

conn = get_conn()
assets = list_assets(conn)
transactions = list_transactions(conn)
if not transactions:
    render_empty_state(has_assets=bool(assets))
    st.stop()

last_prices = {}
for asset in assets:
    last = get_last_price(conn, asset.id)
    if last is not None:
        last_prices[asset.id] = last
positions = compute_positions(assets, transactions, last_prices)

render_indicators(positions, total_deposits(list_cash_movements(conn)))
if positions:
    st.dataframe(positions_table(positions), hide_index=True)
else:
    st.info("No open position: every share you bought has been sold.")
render_allocation(assets, positions)

# the grid stops at the latest price (or trade), so the frozen demo does not drift with the date
history_end = max(latest_price_date(conn) or date.min, transactions[-1].date)
render_history(assets, transactions, history_end)
