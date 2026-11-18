from datetime import date, timedelta

import pandas as pd
import streamlit as st

from simulator import charts
from simulator.database import (
    get_include_fees,
    get_last_price,
    get_price_history,
    list_assets,
    list_cash_movements,
    list_transactions,
    load_tax_settings,
)
from simulator.models import Asset, CashMovement, Position, TaxSettings
from simulator.portfolio import (
    STALE_PRICE_DAYS,
    compute_positions,
    gross_deposits,
    pea_capacity,
    portfolio_history,
    realised_gains,
    stale_prices,
    total_deposits,
)
from simulator.prices import latest_price_date
from simulator.taxes import PEA, net_value, pea_five_year_date, tax_rate
from views.common import format_eur, format_pct, format_quantity, get_conn
from views.positions import render_positions

PERIOD_DAYS = {"3 months": 91, "1 year": 365}
OVERVIEW = "Overview"
DETAIL = "Positions and orders"


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


def render_indicators(positions: list[Position], deposits: float, include_fees: bool) -> None:
    priced = [p for p in positions if p.value is not None]
    invested = sum(p.invested for p in positions)
    value = sum(p.value for p in priced)
    # the gain only compares what has a price with what it cost
    cost_of_priced = sum(p.invested for p in priced)
    gain = value - cost_of_priced
    gain_pct = gain / cost_of_priced if cost_of_priced > 0 else None

    columns = st.columns(4)
    fees_note = "order fees included" if include_fees else "order fees left out"
    # cards of the same height, even though only one of them has a delta
    columns[0].metric(
        "Invested",
        format_eur(invested),
        help=f"Cost of the positions you hold, {fees_note} (see Settings › Account).",
        border=True,
        height="stretch",
    )
    columns[1].metric("Current value", format_eur(value), border=True, height="stretch")
    columns[2].metric(
        "Unrealised gain",
        format_eur(gain),
        delta=format_pct(gain_pct) if gain_pct is not None else None,
        help="Current value minus cost, on the positions that are still open.",
        border=True,
        height="stretch",
    )
    columns[3].metric(
        "Total deposits",
        format_eur(deposits),
        help="Money put on the account: deposits minus withdrawals.",
        border=True,
        height="stretch",
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


def render_history(assets: list[Asset], transactions, end: date, include_fees: bool) -> None:
    st.subheader("History")
    period = st.segmented_control(
        "Period", ["3 months", "1 year", "All"], default="All", key="history_period"
    )
    prices = {asset.id: get_price_history(get_conn(), asset.id) for asset in assets}
    history = portfolio_history(transactions, prices, end=end, include_fees=include_fees)
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


def render_freshness(positions: list[Position], today: date) -> None:
    """Warn about prices that are too old, or that the last refresh could not update."""
    # the demo has frozen prices on purpose: they would always look old
    if st.session_state["demo"]:
        return
    stale = stale_prices(positions, today)
    failed = st.session_state.get("price_failures_real", [])
    if stale:
        st.warning(
            f"Price older than {STALE_PRICE_DAYS} days for: " + ", ".join(stale) + ". "
            "Refresh prices, or enter one by hand in Settings."
        )
    if failed:
        st.warning("The last refresh found no new price for: " + ", ".join(failed) + ".")
    if stale or failed:
        st.page_link("views/settings.py", label="Go to Settings", icon=":material/settings:")


def render_pea_block(
    positions: list[Position],
    cash_movements: list[CashMovement],
    settings: TaxSettings,
    today: date,
) -> None:
    st.subheader("PEA")
    ceiling = settings.pea_ceiling
    deposited = gross_deposits(cash_movements)
    capacity = pea_capacity(cash_movements, ceiling)
    left, middle, right = st.columns(3)

    left.metric(
        "Remaining deposit capacity",
        format_eur(max(capacity, 0.0)),
        help="Ceiling minus all deposits ever made: a withdrawal does not give capacity back.",
    )
    share = deposited / ceiling
    left.progress(
        min(1.0, share),
        text=f"{format_eur(deposited)} of {format_eur(ceiling)} ({format_pct(share)})",
    )
    if capacity < 0:
        left.warning(f"The ceiling is exceeded by {format_eur(-capacity)}.")

    open_date = settings.pea_open_date
    if open_date is None:
        middle.metric("5-year date", "—")
        middle.caption("Enter the opening date in Settings › Account.")
    else:
        five_years = pea_five_year_date(open_date)
        days_left = (five_years - today).days
        middle.metric(
            "5-year date",
            f"{five_years:%d/%m/%Y}",
            delta=f"{days_left} days left" if days_left > 0 else "reached",
            delta_color="off",
            help="After 5 years, gains bear social charges only.",
        )

    value = sum(p.value for p in positions if p.value is not None)
    rate = tax_rate(PEA, open_date, today, settings)
    net = net_value(value, total_deposits(cash_movements), rate)
    right.metric(
        "Net value if withdrawn today",
        format_eur(net),
        help=f"Value minus the tax on the gain, at {format_pct(rate, 1)}. Estimate only.",
    )
    right.caption(f"Tax rate applied: {format_pct(rate, 1)}")

    if open_date is None or today < pea_five_year_date(open_date):
        st.warning(
            "Before 5 years, a withdrawal closes the PEA and the gain bears the flat tax "
            f"({format_pct(rate, 1)}). This tool only applies that rate: it does not simulate "
            "the closing or its exceptions."
        )
    st.caption(
        "Estimate: tax = (value − net deposits) × rate, if positive. Uninvested cash is ignored."
    )
    if any(p.value is None for p in positions):
        st.caption("A position has no price: it is left out, so the net value is understated.")


def render_realised_gains(assets: list[Asset], transactions, include_fees: bool) -> None:
    gains = realised_gains(assets, transactions, include_fees)
    if gains.empty:
        return
    st.subheader("Realised gains")
    st.metric("Total realised gain", format_eur(gains["gain"].sum()))
    gains = gains.sort_values("date", ascending=False, kind="stable")
    table = pd.DataFrame(
        {
            "Date": [f"{day:%d/%m/%Y}" for day in gains["date"]],
            "Asset": gains["asset"],
            "Quantity": [format_quantity(q) for q in gains["quantity"]],
            "Sale price": [format_eur(p) for p in gains["sale_price"]],
            "Avg cost": [format_eur(c) for c in gains["avg_cost"]],
            "Gain": [format_eur(g) for g in gains["gain"]],
        }
    )
    st.dataframe(table, hide_index=True)


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
include_fees = get_include_fees(conn)
positions = compute_positions(assets, transactions, last_prices, include_fees)
cash_movements = list_cash_movements(conn)
today = date.today()

view = st.segmented_control(
    "View",
    [OVERVIEW, DETAIL],
    default=OVERVIEW,
    required=True,
    key="portfolio_view",
    label_visibility="collapsed",
)
render_freshness(positions, today)
if view == DETAIL:
    render_positions(assets, transactions, positions, include_fees, today)
    st.stop()

render_indicators(positions, total_deposits(cash_movements), include_fees)
if positions:
    st.dataframe(positions_table(positions), hide_index=True)
else:
    st.info("No open position: every share you bought has been sold.")
render_allocation(assets, positions)
render_pea_block(positions, cash_movements, load_tax_settings(conn), today)
render_realised_gains(assets, transactions, include_fees)

# the grid stops at the latest price (or trade), so the frozen demo does not drift with the date
history_end = max(latest_price_date(conn) or date.min, transactions[-1].date)
render_history(assets, transactions, history_end, include_fees)
