from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from simulator import charts, config
from simulator.database import get_float_setting, load_tax_settings
from simulator.montecarlo import percentiles
from simulator.taxes import CTO, PEA, tax_rate
from simulator.withdrawal import (
    FIXED,
    INDEXED,
    RATE,
    depletion_month,
    simulate_withdrawals,
    withdrawal_paths,
    withdrawal_schedule,
)
from views.common import data_mode, format_eur, format_pct, get_conn

DATE_FORMAT = "DD/MM/YYYY"
FROM_PROJECTION = "End of a Projection scenario"
CUSTOM = "Custom"
RULES = {
    FIXED: "Fixed monthly amount",
    INDEXED: "Monthly amount indexed on inflation",
    RATE: "Yearly share of the starting capital",
}
SAMPLE_PATHS = 20


def render_start(today: date) -> tuple[float, float, date]:
    """Capital, deposits left in it, and the day the withdrawals start."""
    projected = st.session_state.get(f"projection_result_{data_mode()}")
    sources = [FROM_PROJECTION, CUSTOM] if projected else [CUSTOM]
    source = st.segmented_control(
        "Starting capital", sources, default=sources[0], required=True, key="wd_source"
    )
    if not projected:
        st.caption("Open the Projection page first to start from the end of one of its scenarios.")
    if source == FROM_PROJECTION:
        name = st.selectbox("Scenario", list(projected), key="wd_scenario")
        end = projected[name]
        st.caption(
            f"{format_eur(end['value'])} on {end['date']:%d/%m/%Y}, of which "
            f"{format_eur(end['contributions'])} of deposits. PEA and CTO of the projection are "
            "taken together, as one account in the envelope chosen below."
        )
        return end["value"], end["contributions"], end["date"]
    left, middle, right = st.columns(3)
    capital = left.number_input(
        "Capital (€)", min_value=0.0, value=300_000.0, step=10_000.0, key="wd_capital"
    )
    deposits = middle.number_input(
        "Deposits in it (€)",
        min_value=0.0,
        value=150_000.0,
        step=10_000.0,
        key="wd_deposits",
        help="What you paid in and never withdrew. The rest of the capital is gain, and only "
        "the gain is taxed.",
    )
    start = right.date_input("Withdrawals start", value=today, format=DATE_FORMAT, key="wd_start")
    return capital, deposits, start


st.title("Withdrawal")
st.caption("Living on the capital: how long it lasts, and what is left after tax.")

conn = get_conn()
today = date.today()
settings = load_tax_settings(conn)

with st.container(border=True):
    capital, deposits, start = render_start(today)

    left, middle, right = st.columns(3)
    envelope = left.segmented_control(
        "Envelope", [PEA, CTO], default=PEA, required=True, key="wd_envelope"
    )
    mean = (
        middle.number_input(
            "Yearly return, after fees (%)",
            min_value=-20.0,
            max_value=30.0,
            value=config.DEFAULT_SCENARIOS["Central"] * 100,
            step=0.5,
            format="%.2f",
            key="wd_return",
        )
        / 100
    )
    volatility = (
        right.number_input(
            "Yearly volatility (%)",
            min_value=0.0,
            max_value=80.0,
            value=config.DEFAULT_VOLATILITY * 100,
            step=1.0,
            format="%.2f",
            key="wd_volatility",
            help="Only used by the Monte Carlo run below.",
        )
        / 100
    )

    rule = st.segmented_control(
        "Withdrawal rule",
        list(RULES),
        format_func=RULES.get,
        default=RATE,
        required=True,
        key="wd_rule",
    )
    left, middle, right = st.columns(3)
    # the two fields are always drawn: Streamlit forgets the value of a field it stops drawing
    amount = left.number_input(
        "Monthly amount, before tax (€)",
        min_value=0.0,
        value=1000.0,
        step=100.0,
        key="wd_amount",
        disabled=rule == RATE,
    )
    share = (
        left.number_input(
            "Yearly share of the starting capital (%)",
            min_value=0.0,
            max_value=100.0,
            value=4.0,
            step=0.5,
            format="%.2f",
            key="wd_share",
            disabled=rule != RATE,
            help="The first year you withdraw this share of the capital, by twelfths; then the "
            "amount follows inflation. 4 % is a common rule of thumb, not a guarantee.",
        )
        / 100
    )
    inflation = (
        middle.number_input(
            "Inflation (%)",
            min_value=0.0,
            max_value=20.0,
            value=get_float_setting(conn, "inflation", config.DEFAULT_INFLATION) * 100,
            step=0.1,
            format="%.2f",
            key="wd_inflation",
            disabled=rule == FIXED,
        )
        / 100
    )
    years = right.number_input(
        "Duration (years)", min_value=1, max_value=60, value=30, step=1, key="wd_years"
    )

if capital <= 0:
    st.info("Enter a capital above 0.")
    st.stop()

n_months = int(years) * 12
schedule = withdrawal_schedule(
    rule, share if rule == RATE else amount, capital, inflation, n_months
)
if schedule[0] <= 0:
    st.info("Enter a withdrawal above 0.")
    st.stop()
# the day the withdrawals start, then the month ends that follow it
first_day = pd.Timestamp(start)
days = pd.date_range(first_day + pd.Timedelta(days=1), periods=n_months, freq="ME")
days = days.insert(0, first_day)
rates = [tax_rate(envelope, settings.pea_open_date, day.date(), settings) for day in days[1:]]
table = simulate_withdrawals(capital, deposits, schedule, mean, rates)
table.insert(0, "date", days)
empty_month = depletion_month(table)

st.subheader("With a fixed return")
columns = st.columns(4)
if empty_month is None:
    columns[0].metric(
        "Capital lasts", f"{int(years)} years", delta="the whole period", delta_color="off"
    )
    columns[1].metric("Left at the end", format_eur(table["value"].iloc[-1]))
else:
    lasted = f"{empty_month // 12} years {empty_month % 12} months"
    columns[0].metric("Capital runs out", f"{days[empty_month]:%m/%Y}", delta=f"after {lasted}")
    columns[1].metric("Left at the end", format_eur(0.0))
columns[2].metric(
    "Total withdrawn, before tax",
    format_eur(table["withdrawal"].sum()),
)
columns[3].metric(
    "Total withdrawn, after tax",
    format_eur(table["net"].sum()),
    delta=f"-{format_eur(table['tax'].sum())} of tax",
    delta_color="off",
)
first = table.iloc[1]
st.caption(
    f"First month: {format_eur(first['withdrawal'])} withdrawn, {format_eur(first['tax'])} of "
    f"tax ({format_pct(rates[0], 1)} on the gain part), {format_eur(first['net'])} for you. "
    f"Last planned month: {format_eur(schedule[-1])} before tax."
)
curves = pd.DataFrame({"date": table["date"], "Remaining capital": table["value"]})
st.plotly_chart(charts.comparison_chart(curves), width="stretch")
if empty_month is not None:
    st.warning(
        f"At {format_pct(mean, 2)} a year, the capital is used up in {days[empty_month]:%B %Y}: "
        "the withdrawals after that date are not paid."
    )
if envelope == PEA and (settings.pea_open_date is None or rates[0] > settings.ps_rate):
    st.warning(
        "The PEA is less than 5 years old when the withdrawals start (or its opening date is "
        "unknown): a withdrawal would close it. This page only applies the flat tax until the "
        "fifth birthday; it does not simulate the closing."
    )

with st.expander("How is each withdrawal taxed?"):
    gain_share = max(capital - deposits, 0.0) / capital
    st.markdown(
        f"""
- Your capital of **{format_eur(capital)}** holds **{format_eur(deposits)}** of your own
  deposits; the rest, **{format_pct(gain_share)}** of it, is gain.
- Each withdrawal is treated as a slice of the whole: it holds the same share of gain. Only
  that gain part is taxed, never your deposits coming back.
- First month: {format_eur(first["withdrawal"])} withdrawn x {format_pct(gain_share)} of gain
  x {format_pct(rates[0], 1)} of tax = **{format_eur(first["tax"])}**. You receive
  **{format_eur(first["net"])}**.
- The share of gain moves over time: it rises while the capital keeps growing, so later
  withdrawals bear a little more tax.
- Over the whole period: **{format_eur(table["withdrawal"].sum())}** withdrawn,
  **{format_eur(table["tax"].sum())}** of tax, **{format_eur(table["net"].sum())}** received.
"""
    )

st.subheader("With random returns (Monte Carlo)")
left, middle, right = st.columns(3, vertical_alignment="bottom")
n_paths = left.number_input(
    "Number of paths",
    min_value=100,
    max_value=20_000,
    value=config.DEFAULT_N_PATHS,
    step=1000,
    key="wd_paths",
)
seed = middle.number_input(
    "Random seed", min_value=0, value=config.DEFAULT_SEED, step=1, key="wd_seed"
)
inputs = (capital, tuple(schedule), mean, volatility, int(n_paths), int(seed))
result_key = f"withdrawal_mc_{data_mode()}"
if right.button("Run Monte Carlo", type="primary", icon=":material/casino:"):
    paths = withdrawal_paths(capital, schedule, mean, volatility, int(n_paths), int(seed))
    months_held = (paths > 0).sum(axis=1) - 1
    st.session_state[result_key] = {
        "inputs": inputs,
        "bands": percentiles(paths),
        "samples": pd.DataFrame(paths[:SAMPLE_PATHS].T),
        "final": paths[:, -1],
        "months_held": months_held,
    }
result = st.session_state.get(result_key)
if result is None:
    st.caption(
        "A fixed return never happens: a few bad years at the start can empty a capital that "
        "the average return would have kept alive. Run Monte Carlo to see how often it lasts."
    )
    st.stop()
if result["inputs"] != inputs:
    st.warning("The settings changed since this run: click Run Monte Carlo to update it.")
    st.stop()

final, months_held = result["final"], result["months_held"]
success = float(np.mean(final > 0))
failed = months_held[final <= 0]
columns = st.columns(3)
columns[0].metric(
    "Chance the capital lasts",
    format_pct(success),
    help=f"Share of the paths that still hold money after {int(years)} years.",
)
columns[1].metric("Median left at the end", format_eur(float(np.median(final))))
if len(failed):
    worst = int(np.percentile(failed, 10))
    columns[2].metric(
        "When it fails, it often lasts",
        f"{int(np.median(failed)) // 12} years",
        help="Median duration of the paths that run out. One failing path out of ten runs out "
        f"before {worst // 12} years.",
    )
bands = result["bands"].copy()
bands.insert(0, "date", days)
st.plotly_chart(charts.fan_chart(bands, result["samples"]), width="stretch")
st.caption(
    "Remaining capital before tax: median, 25-75 % band and 10-90 % band; "
    f"{len(result['samples'].columns)} single paths in grey. A path that reaches 0 stays there."
)
with st.expander("What do these numbers mean?"):
    n_drawn = f"{len(final):,}".replace(",", " ")
    low, high = (format_eur(v) for v in np.percentile(final, (10, 90)))
    st.markdown(
        f"""
- The run drew {n_drawn} possible futures with the same withdrawals. The return of each month
  is random, around {format_pct(mean, 2)} a year.
- **Chance the capital lasts, {format_pct(success)}**: in that share of the futures, there is
  still money after {int(years)} years. In the other {format_pct(1 - success)}, the capital
  runs out before the end and the withdrawals stop.
- **Median left at the end, {format_eur(float(np.median(final)))}**: half of the futures end
  with less, half with more. In 1 future out of 10 less than **{low}** is left; in 1 out of 10,
  more than **{high}**.
- Why it differs from the fixed return above: the order of the good and bad years matters.
  A fall early on forces you to sell more shares for the same withdrawal, and those shares
  are no longer there for the recovery.
- A lower withdrawal, or one that you cut in bad years, raises the chance that the capital
  lasts. This page does not simulate such an adjustment.
"""
    )
st.caption(
    "Each month is drawn independently, with a constant volatility: no lasting crisis is "
    "modelled. Tax and inflation rules are those of today. Not a forecast."
)
