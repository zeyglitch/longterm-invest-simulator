from datetime import date

import pandas as pd
import streamlit as st

from simulator import charts, config
from simulator.backtest import (
    annualised_return,
    backtest_dca,
    drawdowns,
    max_drawdown,
    monthly_prices,
)
from simulator.database import get_price_history, list_assets
from views.common import cached_history, data_mode, format_eur, format_pct, get_conn

DATE_FORMAT = "DD/MM/YYYY"
OTHER = "Another ticker"
DEFAULT_YEARS = 10


def load_prices(ticker: str, assets: list) -> tuple[pd.Series, str]:
    """Daily prices of a ticker and where they come from.

    Yahoo first. If it does not answer and the ticker is the one of a known asset, the prices
    already stored for it are used instead (the demo assets only have those).
    """
    history = cached_history(ticker)
    if not history.empty:
        return history, "Yahoo Finance"
    for asset in assets:
        if asset.ticker == ticker:
            return get_price_history(get_conn(), asset.id), "the prices stored in the app"
    return history, ""


def summary(table: pd.DataFrame, flows: list[float]) -> dict:
    last = table.iloc[-1]
    return {
        "value": last["value"],
        "invested": last["invested"],
        "gain": last["value"] - last["invested"],
        "rate": annualised_return(flows, last["value"]),
    }


st.title("Backtest")
st.caption("What if you had invested every month since a past date? Replayed on real prices.")

conn = get_conn()
today = date.today()
assets = [asset for asset in list_assets(conn) if asset.ticker]
result_key = f"backtest_{data_mode()}"

with st.container(border=True):
    left, right = st.columns(2)
    choices = [asset.ticker for asset in assets] + [OTHER]
    names = {asset.ticker: f"{asset.name} ({asset.ticker})" for asset in assets}
    choice = left.selectbox(
        "Fund", choices, format_func=lambda ticker: names.get(ticker, ticker), key="bt_choice"
    )
    typed = right.text_input(
        "Yahoo ticker",
        value=config.DEFAULT_BENCHMARK,
        key="bt_ticker",
        disabled=choice != OTHER,
        help="For example CW8.PA. A fund with a long history gives a longer backtest.",
    )
    ticker = typed.strip().upper() if choice == OTHER else choice

    left, right = st.columns(2)
    start = left.date_input(
        "Start",
        value=date(today.year - DEFAULT_YEARS, today.month, 1),
        min_value=date(1990, 1, 1),
        max_value=today,
        format=DATE_FORMAT,
        key="bt_start",
    )
    end = right.date_input(
        "End",
        value=today,
        min_value=date(1990, 1, 1),
        max_value=today,
        format=DATE_FORMAT,
        key="bt_end",
    )
    left, middle, right = st.columns(3)
    initial = left.number_input(
        "Initial deposit (€)", min_value=0.0, value=0.0, step=500.0, key="bt_initial"
    )
    monthly = middle.number_input(
        "Monthly deposit (€)", min_value=0.0, value=100.0, step=50.0, key="bt_monthly"
    )
    fee = right.number_input(
        "Fee per order (€)",
        min_value=0.0,
        value=0.0,
        step=1.0,
        key="bt_fee",
        help="Taken from each deposit: only the rest buys shares.",
    )
    left, right = st.columns(2, vertical_alignment="center")
    run = left.button("Run backtest", type="primary", icon=":material/history:")
    compare_lump = right.toggle(
        "Compare with a lump sum",
        key="bt_lump",
        help="The same total amount, all invested on the first day.",
    )

if run:
    if not ticker:
        st.error("Type a Yahoo ticker first.")
        st.stop()
    with st.spinner("Loading the price history..."):
        history, source = load_prices(ticker, assets)
    if history.empty:
        st.error(f"No price history for {ticker}. Check the ticker and your connection.")
        st.stop()
    st.session_state[result_key] = {"ticker": ticker, "history": history, "source": source}

result = st.session_state.get(result_key)
if result is None:
    st.info("Choose a fund and your deposits, then click Run backtest.")
    st.stop()
if result["ticker"] != ticker:
    st.warning(
        f"The fund changed: the results below are those of {result['ticker']}. "
        "Click Run backtest to update them."
    )
if start >= end:
    st.error("The start must be before the end.")
    st.stop()
if initial + monthly <= 0:
    st.error("Enter an initial deposit or a monthly deposit.")
    st.stop()

history = result["history"]
daily = history[(history.index >= pd.Timestamp(start)) & (history.index <= pd.Timestamp(end))]
prices = monthly_prices(daily)
if len(prices) < 2:
    st.warning(
        f"Prices of {result['ticker']} run from {history.index[0]:%d/%m/%Y} to "
        f"{history.index[-1]:%d/%m/%Y}: less than two months fall in the period chosen."
    )
    st.stop()
if daily.index[0] > pd.Timestamp(start) + pd.Timedelta(days=31):
    st.info(
        f"No price for {result['ticker']} before {daily.index[0]:%d/%m/%Y}: "
        "the backtest starts on that day."
    )

table = backtest_dca(prices, monthly, initial, fee)
flows = [monthly] * len(prices)
flows[0] += initial
dca = summary(table, flows)
fall, high_day, low_day = max_drawdown(daily)

st.subheader(f"{result['ticker']}, {prices.index[0]:%m/%Y} to {prices.index[-1]:%m/%Y}")
columns = st.columns(4)
columns[0].metric("Final value", format_eur(dca["value"]))
columns[1].metric(
    "Invested", format_eur(dca["invested"]), help="Sum of the deposits, order fees included."
)
columns[2].metric("Gain", format_eur(dca["gain"]), delta=format_pct(dca["gain"] / dca["invested"]))
columns[3].metric(
    "Annualised return",
    format_pct(dca["rate"], 2),
    help="The fixed yearly rate at which the same deposits would have given the same final value.",
)

curves = pd.DataFrame({"date": table["date"], "Monthly deposits": table["value"]})
lump = None
if compare_lump:
    total = dca["invested"]
    lump_table = backtest_dca(prices, 0.0, total, fee)
    lump = summary(lump_table, [total] + [0.0] * (len(prices) - 1))
    curves["Lump sum"] = lump_table["value"]
curves["Contributions"] = table["invested"]
st.plotly_chart(charts.comparison_chart(curves), width="stretch")
st.caption(
    f"One order a month, at the first price of the month, from {result['source']}. "
    "`Contributions` is what the monthly deposits had put in by then."
)

if lump is not None:
    st.subheader("Monthly deposits against a lump sum")
    rows = [
        {
            "Strategy": name,
            "Invested": format_eur(numbers["invested"]),
            "Final value": format_eur(numbers["value"]),
            "Gain": format_eur(numbers["gain"]),
            "Annualised return": format_pct(numbers["rate"], 2),
        }
        for name, numbers in (("Monthly deposits", dca), ("Lump sum", lump))
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(
        f"Lump sum: the same {format_eur(dca['invested'])} all invested on "
        f"{prices.index[0]:%d/%m/%Y}. It was invested longer, so it wins whenever the market "
        "rose over the period; that says nothing about the next one. Having the whole sum on "
        "the first day is also rarely possible when you save from a salary."
    )

st.subheader("Falls from the previous high")
left, right = st.columns([1, 3])
left.metric("Worst fall", format_pct(fall))
left.caption(f"From the high of {high_day:%d/%m/%Y} to the low of {low_day:%d/%m/%Y}.")
right.plotly_chart(charts.drawdown_chart(drawdowns(daily)), width="stretch")
st.caption(
    "Measured on the price of one share, day by day, not on the value of your account: your "
    "deposits would hide part of each fall."
)

with st.expander("What do these numbers mean?"):
    example = 10_000
    st.markdown(
        f"""
- **Final value, {format_eur(dca["value"])}**: what your shares would be worth on
  {prices.index[-1]:%d/%m/%Y}, before tax, had you really made these deposits.
- **Invested, {format_eur(dca["invested"])}**: the money you would have paid in,
  {len(prices)} deposits in all.
- **Gain, {format_eur(dca["gain"])}**: the difference between the two. It is
  {format_pct(dca["gain"] / dca["invested"])} of what you paid in, but spread over the whole
  period, and the last deposits had little time to grow.
- **Annualised return, {format_pct(dca["rate"], 2)}**: the yearly interest rate of a savings
  account that would have turned the same deposits into the same final value. It is the number
  to compare with the return you type in Projection and Monte Carlo.
- **Worst fall, {format_pct(fall)}**: between {high_day:%d/%m/%Y} and {low_day:%d/%m/%Y}, the
  fund lost that much. {format_eur(example)} held at the high were worth
  {format_eur(example * (1 + fall))} at the low.
"""
    )

st.caption(
    "Past performance does not predict future performance. Yahoo prices of accumulating ETFs "
    "already include their dividends and their fees. Tax is not counted here."
)
