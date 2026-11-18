"""Detailed view of the Portfolio page: positions, last orders, performance, one fund."""

from datetime import date

import pandas as pd
import streamlit as st

from simulator import charts
from simulator.database import get_price_history
from simulator.models import Asset, Position, Transaction
from simulator.portfolio import (
    PERIODS,
    period_start,
    portfolio_history,
    price_change,
    price_range,
)
from views.common import asset_label, format_eur, format_pct, format_quantity, get_conn

TREND_WEEKS = 13
LAST_ORDERS = 8


def signed_colour(value) -> str:
    """CSS for a table cell: green for a gain, red for a loss."""
    if not isinstance(value, (int, float)) or pd.isna(value) or value == 0:
        return ""
    return "color: #34D399" if value > 0 else "color: #F87171"


def delta_style(value: float) -> dict:
    """Colour and arrow of a metric delta: green up, red down, gray when nothing moved."""
    if value > 0:
        return {"delta_color": "green", "delta_arrow": "up"}
    if value < 0:
        return {"delta_color": "red", "delta_arrow": "down"}
    return {"delta_color": "gray", "delta_arrow": "off"}


def signed_eur(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return ("+" if value > 0 else "") + format_eur(value)


def signed_pct(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return ("+" if value > 0 else "") + format_pct(value, 2)


def render_kpis(positions: list[Position], changes: dict[int, dict | None]) -> None:
    priced = [p for p in positions if p.value is not None]
    value = sum(p.value for p in priced)
    cost = sum(p.invested for p in priced)
    # what the last price move earned or lost, on the shares held now
    moved = [(p, changes.get(p.asset.id)) for p in priced]
    day_change = sum(p.quantity * change["change"] for p, change in moved if change)
    before = value - day_change
    left, middle, right = st.columns(3)
    left.metric(
        "Value",
        format_eur(value),
        delta=f"{signed_eur(day_change)} ({signed_pct(day_change / before if before else 0)})",
        **delta_style(day_change),
        border=True,
        help="Change since the previous known price of each fund, on the shares held today.",
    )
    gain = value - cost
    middle.metric(
        "Unrealised gain",
        signed_eur(gain),
        delta=signed_pct(gain / cost if cost else 0),
        **delta_style(gain),
        border=True,
    )
    right.metric(
        "Invested",
        format_eur(cost),
        delta=f"{len(priced)} position(s)",
        delta_color="off",
        delta_arrow="off",
        border=True,
        help="Cost of the shares held, as on the Overview.",
    )


def positions_frame(
    positions: list[Position], changes: dict, prices: dict[int, pd.Series]
) -> pd.DataFrame:
    rows = []
    for p in positions:
        change = changes.get(p.asset.id)
        weekly = prices[p.asset.id].resample("W").last().dropna().tail(TREND_WEEKS)
        rows.append(
            {
                "Asset": p.asset.name,
                "Trend": weekly.tolist(),
                "Quantity": p.quantity,
                "Last price": p.last_price,
                "Change %": change["change_pct"] if change else None,
                "Change €": change["change"] if change else None,
                "Value": p.value,
                "Avg cost": p.avg_cost,
                "Gain €": p.gain,
                "Gain %": p.gain_pct,
                "Weight": p.weight,
            }
        )
    return pd.DataFrame(rows)


def render_positions_table(positions, changes, prices) -> None:
    table = positions_frame(positions, changes, prices)
    styled = (
        table.style.format(format_quantity, subset=["Quantity"])
        .format(format_eur, subset=["Last price", "Value", "Avg cost"], na_rep="—")
        .format(signed_eur, subset=["Change €", "Gain €"])
        .format(signed_pct, subset=["Change %", "Gain %"])
        .format(
            lambda w: format_pct(w) if w is not None and not pd.isna(w) else "—", subset=["Weight"]
        )
        .map(signed_colour, subset=["Change %", "Change €", "Gain €", "Gain %"])
    )
    st.dataframe(
        styled,
        hide_index=True,
        column_config={
            "Trend": st.column_config.LineChartColumn(
                f"{TREND_WEEKS} weeks", help="Price at the end of each week.", width="small"
            ),
            "Change %": st.column_config.Column(help="Since the previous known price."),
            "Change €": st.column_config.Column(help="Per share, since the previous price."),
        },
    )


def render_last_orders(transactions: list[Transaction], assets: list[Asset]) -> None:
    names = {asset.id: asset.name for asset in assets}
    latest = sorted(transactions, key=lambda tx: (tx.date, tx.id or 0), reverse=True)
    rows = [
        {
            "Date": f"{tx.date:%d/%m/%Y}",
            "Asset": names[tx.asset_id],
            "Type": "Buy" if tx.kind == "BUY" else "Sell",
            "Quantity": format_quantity(tx.quantity),
            "Total": format_eur(
                tx.quantity * tx.unit_price + (tx.fees if tx.kind == "BUY" else -tx.fees)
            ),
        }
        for tx in latest[:LAST_ORDERS]
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.page_link("views/transactions.py", label="All transactions", icon=":material/receipt_long:")


def render_performance(transactions, prices, include_fees: bool, today: date) -> None:
    period = st.segmented_control(
        "Period",
        PERIODS,
        default="1Y",
        required=True,
        key="perf_period",
        label_visibility="collapsed",
    )
    start = period_start(today, period)
    # one point a day on short periods, one a week otherwise
    freq = "B" if period in ("1W", "1M") else "W"
    history = portfolio_history(
        transactions, prices, freq=freq, include_fees=include_fees, end=today
    )
    if start is not None:
        history = history[history["date"] >= pd.Timestamp(start)]
    if len(history) < 2:
        st.caption("Not enough history for this period yet.")
        return
    history = history.assign(gain=history["value"] - history["invested"])
    moved = history["gain"].iloc[-1] - history["gain"].iloc[0]
    first, last = history["gain"].iloc[0], history["gain"].iloc[-1]
    since = history["date"].iloc[0]
    st.metric(
        f"Change since {since:%d/%m/%Y}",
        signed_eur(moved),
        **delta_style(moved),
        delta=f"unrealised gain: {signed_eur(first)} then {signed_eur(last)}",
        help="What the market added or took away over the period: the unrealised gain (value "
        "minus what the shares cost) at the end, minus the one at the start. Buying or selling "
        "does not count. The curve shows that unrealised gain day by day.",
    )
    st.plotly_chart(charts.gain_chart(history), width="stretch")


def render_instrument(positions, transactions, prices, today: date) -> None:
    held = {p.asset.id: p for p in positions}
    position = st.selectbox(
        "Fund",
        list(held.values()),
        format_func=lambda p: asset_label(p.asset),
        key="detail_asset",
        label_visibility="collapsed",
    )
    asset = position.asset
    history = prices[asset.id]
    change = price_change(history)
    left, right = st.columns(2)
    if change:
        left.metric(
            "Last price",
            format_eur(change["last"]),
            delta=f"{signed_eur(change['change'])} ({signed_pct(change['change_pct'])})",
            **delta_style(change["change"]),
            help=f"On {change['last_date']:%d/%m/%Y}, against {change['previous_date']:%d/%m/%Y}.",
        )
    else:
        left.metric("Last price", format_eur(position.last_price))
    right.metric(
        "Your position",
        format_eur(position.value),
        delta=signed_pct(position.gain_pct),
        **delta_style((position.gain_pct or 0)),
    )

    yearly = price_range(history, today)
    last = change["last"] if change else position.last_price
    if yearly and last is not None:
        low, high = yearly
        where = (last - low) / (high - low) if high > low else 1.0
        st.progress(
            min(max(where, 0.0), 1.0),
            text=f"1 year: low {format_eur(low)} · high {format_eur(high)}",
        )

    period = st.segmented_control(
        "Chart period",
        PERIODS,
        default="1Y",
        required=True,
        key="detail_period",
        label_visibility="collapsed",
    )
    start = period_start(today, period)
    shown = history if start is None else history[history.index >= pd.Timestamp(start)]
    own = [tx for tx in transactions if tx.asset_id == asset.id and tx.kind == "BUY"]
    buys = pd.DataFrame(
        {"date": [pd.Timestamp(tx.date) for tx in own], "price": [tx.unit_price for tx in own]}
    )
    if start is not None and not buys.empty:
        buys = buys[buys["date"] >= pd.Timestamp(start)]
    if len(shown) >= 2:
        st.plotly_chart(charts.price_chart(shown, buys), width="stretch")
    else:
        st.caption("Not enough prices for this period.")

    first_buy = min((tx.date for tx in own), default=None)
    stats = {
        "Shares held": format_quantity(position.quantity),
        "Average cost": format_eur(position.avg_cost),
        "Invested": format_eur(position.invested),
        "Unrealised gain": signed_eur(position.gain),
        "Weight / target": f"{format_pct(position.weight)} / {format_pct(asset.target_weight)}",
        "Yearly fees": format_pct(asset.fees, 2),
        "First buy": f"{first_buy:%d/%m/%Y}" if first_buy else "—",
        "ISIN": asset.isin,
    }
    columns = st.columns(2)
    for number, (label, text) in enumerate(stats.items()):
        columns[number % 2].markdown(f":gray[{label}]  \n**{text}**")


def render_positions(
    assets: list[Asset],
    transactions: list[Transaction],
    positions: list[Position],
    include_fees: bool,
    today: date,
) -> None:
    """The detailed view: one card per question, as on a trading screen."""
    if not positions:
        st.info("No open position: every share you bought has been sold.")
        return
    prices = {asset.id: get_price_history(get_conn(), asset.id) for asset in assets}
    changes = {p.asset.id: price_change(prices[p.asset.id]) for p in positions}

    unpriced = [p.asset.name for p in positions if p.value is None]
    if unpriced:
        st.warning(
            "No price for " + ", ".join(unpriced) + ": left out of the value and the gain. "
            "Refresh prices, or enter one by hand in Settings."
        )
    render_kpis(positions, changes)

    with st.container(border=True):
        st.subheader("Your positions")
        render_positions_table(positions, changes, prices)
        st.caption(
            "Change: since the previous known price of each fund (the day before with daily "
            "prices). Gain: on the shares held, against what they cost."
        )

    left, right = st.columns([3, 2])
    with left.container(border=True, height="stretch"):
        st.subheader("Your performance")
        render_performance(transactions, prices, include_fees, today)
    with right.container(border=True, height="stretch"):
        st.subheader("Instrument")
        render_instrument(positions, transactions, prices, today)

    with st.container(border=True):
        st.subheader("Your last orders")
        render_last_orders(transactions, assets)
