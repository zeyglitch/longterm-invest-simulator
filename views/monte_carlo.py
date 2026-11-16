from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from simulator import charts, config
from simulator.database import get_float_setting, list_assets, load_tax_settings
from simulator.montecarlo import (
    bootstrap_returns,
    estimate_mu_sigma,
    net_final_values,
    normal_returns,
    percentiles,
    probability_above,
    random_unemployment,
    run_paths,
)
from simulator.plan import build_flow_parts, plan_to_json, validate_plan
from simulator.portfolio import weighted_fees
from simulator.prices import fetch_history, monthly_returns
from simulator.projection import net_return, project
from simulator.taxes import CTO, PEA, tax_rate
from views.common import data_mode, format_eur, format_pct, get_conn
from views.inputs import choose_plan, render_start

NORMAL = "Normal returns"
BOOTSTRAP = "Historical bootstrap"
SAMPLE_PATHS = 20
MIN_HISTORY_MONTHS = 60
HORIZON_YEARS = (10, 20, 30)


def benchmark_history(ticker: str) -> pd.Series:
    """Prices of the reference index, downloaded once per session. Empty if Yahoo fails."""
    key = f"benchmark_{ticker}"
    if key not in st.session_state:
        history = fetch_history(ticker)
        if history.empty:
            return history  # not kept: the next click tries again
        st.session_state[key] = history
    return st.session_state[key]


def describe_history(history: pd.Series) -> str:
    months = len(monthly_returns(history))
    return f"{history.index[0]:%m/%Y} to {history.index[-1]:%m/%Y} ({months} months)"


def estimate_from_history() -> None:
    """Called by `Estimate from history`: fill the mean return and the volatility."""
    ticker = st.session_state["mc_ticker"].strip().upper()
    history = benchmark_history(ticker)
    if len(monthly_returns(history)) < 2:
        st.session_state["mc_estimate"] = ("warning", f"No usable price history for {ticker}.")
        return
    mu, sigma = estimate_mu_sigma(history)
    st.session_state["mc_mu"] = round(mu * 100, 2)
    st.session_state["mc_sigma"] = round(sigma * 100, 2)
    st.session_state["mc_fees"] = 0.0
    st.session_state["mc_estimate"] = (
        "success",
        f"{ticker}, {describe_history(history)}: mean return {format_pct(mu, 2)}, volatility "
        f"{format_pct(sigma, 2)}. Fees are set to 0: a fund's prices already include its fees.",
    )


st.title("Monte Carlo")
st.caption(
    "Nobody knows future returns. This page draws thousands of possible futures at random and "
    "shows how spread out the results are."
)

conn = get_conn()
today = date.today()
settings = load_tax_settings(conn)
st.session_state.setdefault("mc_mu", config.DEFAULT_SCENARIOS["Central"] * 100)
st.session_state.setdefault("mc_sigma", config.DEFAULT_VOLATILITY * 100)
# same default as the Projection page, unrounded: with no volatility the two pages agree
st.session_state.setdefault("mc_fees", weighted_fees(list_assets(conn)) * 100)
st.session_state.setdefault("mc_ticker", config.DEFAULT_BENCHMARK)

with st.container(border=True):
    left, right = st.columns(2)
    with left:
        plan = choose_plan("mc")
    with right:
        start_value, start_deposits, _ = render_start("mc")

    method = st.segmented_control(
        "Method",
        [NORMAL, BOOTSTRAP],
        default=NORMAL,
        required=True,
        key="mc_method",
        help="Normal returns: you give a mean return and a volatility. Historical bootstrap: "
        "each simulated month gets the return of a past month of the reference index, drawn "
        "at random.",
    )
    # always drawn, even when not used: Streamlit forgets the value of a field it stops drawing
    normal = method == NORMAL
    left, middle, right = st.columns(3)
    mu = left.number_input(
        "Mean yearly return (%)",
        min_value=-20.0,
        max_value=30.0,
        step=0.5,
        format="%.2f",
        key="mc_mu",
        disabled=not normal,
    )
    sigma = middle.number_input(
        "Yearly volatility (%)",
        min_value=0.0,
        max_value=80.0,
        step=1.0,
        format="%.2f",
        key="mc_sigma",
        disabled=not normal,
        help="How much yearly returns move around their mean. About 15 % for world equities.",
    )
    fees = right.number_input(
        "Yearly fees (%)",
        min_value=0.0,
        max_value=5.0,
        step=0.05,
        format="%.3f",
        key="mc_fees",
        disabled=not normal,
    )
    left, middle, right = st.columns(3, vertical_alignment="bottom")
    ticker = left.text_input(
        "Reference index (Yahoo ticker)",
        key="mc_ticker",
        help="A fund with a long price history. Recent funds have too little.",
    )
    if method == NORMAL:
        middle.button("Estimate from history", on_click=estimate_from_history)
        message = st.session_state.get("mc_estimate")
        if message:
            (st.success if message[0] == "success" else st.warning)(message[1])
    else:
        st.caption(
            "The three fields above are not used here: the returns come from the past prices "
            "of the reference index, which already include its fees."
        )

    left, middle, right = st.columns(3)
    n_paths = left.number_input(
        "Number of paths",
        min_value=100,
        max_value=20_000,
        value=config.DEFAULT_N_PATHS,
        step=1000,
        key="mc_paths",
    )
    seed = middle.number_input(
        "Random seed",
        min_value=0,
        value=config.DEFAULT_SEED,
        step=1,
        key="mc_seed",
        help="The same seed with the same settings gives the same result again.",
    )
    target = right.number_input(
        "Target amount (€, optional)", min_value=0.0, value=None, step=10000.0, key="mc_target"
    )

    left, middle, right = st.columns(3)
    unemployment = left.toggle(
        "Random unemployment",
        key="mc_unemployment",
        help="Each year of each path can start a period without deposits.",
    )
    job_prob = middle.number_input(
        "Yearly probability (%)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=1.0,
        key="mc_job_prob",
        disabled=not unemployment,
    )
    job_months = right.number_input(
        "Duration (months)",
        min_value=1,
        value=6,
        step=1,
        key="mc_job_months",
        disabled=not unemployment,
    )

    left, middle, right = st.columns(3)
    envelope = left.segmented_control(
        "Envelope", [PEA, CTO], default=PEA, required=True, key="mc_envelope"
    )
    after_tax = middle.toggle("After tax (final values)", key="mc_after_tax")
    index_deposits = middle.toggle(
        "Index contributions on inflation",
        key="mc_index",
        help="The regular deposits of the plan grow with inflation, once a year.",
    )
    adjust = right.toggle("Adjust for inflation", key="mc_adjust")
    inflation = (
        right.number_input(
            "Inflation (%)",
            min_value=0.0,
            max_value=20.0,
            value=get_float_setting(conn, "inflation", config.DEFAULT_INFLATION) * 100,
            step=0.1,
            format="%.2f",
            key="mc_inflation",
        )
        / 100
    )
    run = st.button("Run simulation", type="primary", icon=":material/casino:")

errors = validate_plan(plan)
if errors:
    st.error("This plan cannot be simulated:\n\n" + "\n\n".join(errors))
    st.page_link("views/plan.py", label="Fix it in Plan", icon=":material/edit:")
    st.stop()
# deposits and withdrawals apart: unemployment stops the deposits, never a planned withdrawal
flows = build_flow_parts(plan, today, inflation, index_on_inflation=index_deposits)
if flows.empty:
    st.info("This plan ends before next month: choose a later end date in Plan.")
    st.stop()

# what the stored result depends on: if it changes, the result on screen is out of date
inputs = (
    plan_to_json(plan),
    start_value,
    start_deposits,
    method,
    ticker.strip().upper() if method == BOOTSTRAP else (mu, sigma, fees),
    int(n_paths),
    int(seed),
    (job_prob, int(job_months)) if unemployment else None,
    inflation if index_deposits else None,
)
result_key = f"mc_result_{data_mode()}"

if run:
    n_months = len(flows)
    if method == NORMAL:
        mean_return = net_return(mu / 100, fees / 100)
        returns = normal_returns(n_months, mean_return, sigma / 100, int(n_paths), int(seed))
        model = (
            f"normal returns, mean {format_pct(mean_return, 2)} after fees, "
            f"volatility {sigma:.2f} %"
        )
    else:
        with st.spinner("Downloading the price history..."):
            history = benchmark_history(ticker.strip().upper())
        past = monthly_returns(history)
        if len(past) < 2:
            st.error(f"No usable price history for {ticker}. Check the ticker and your connection.")
            st.stop()
        mean_return = estimate_mu_sigma(history)[0]
        returns = bootstrap_returns(n_months, past, int(n_paths), int(seed))
        model = (
            f"monthly returns of {ticker.strip().upper()} drawn at random, "
            f"{describe_history(history)}"
        )
        if len(past) < MIN_HISTORY_MONTHS:
            model += " (less than 5 years: a short and fragile sample)"
    deposits = flows["deposit"].to_numpy()
    if unemployment:
        # another seed than the returns, so that the two draws are independent
        deposits = random_unemployment(
            deposits, int(n_paths), job_prob / 100, int(job_months), int(seed) + 1
        )
    paths, contributions = run_paths(
        start_value, deposits, returns, start_deposits, flows["withdrawal"].to_numpy()
    )
    planned = project(start_value, start_deposits, flows, mean_return)
    bands = percentiles(paths)
    bands.insert(0, "date", planned["date"])
    bands["contributions"] = planned["contributions"]
    st.session_state[result_key] = {
        "inputs": inputs,
        "bands": bands,
        "samples": pd.DataFrame(paths[:SAMPLE_PATHS].T),
        "final": paths[:, -1],
        "contributions": contributions,
        "model": model,
    }

result = st.session_state.get(result_key)
if result is None:
    st.info("Choose your settings, then click Run simulation.")
    st.stop()
if result["inputs"] != inputs:
    st.warning("The settings changed since this simulation: click Run simulation to update it.")

bands, samples = result["bands"], result["samples"]
n_months = len(bands) - 1
# euros of today: each month is divided by the inflation accumulated until then
deflator = (1 + inflation) ** (np.arange(n_months + 1) / 12) if adjust else np.ones(n_months + 1)
shown_bands = bands.copy()
value_columns = [column for column in bands.columns if column != "date"]
shown_bands[value_columns] = bands[value_columns].div(deflator, axis=0)
shown_samples = samples.div(deflator, axis=0)

st.subheader("Spread of the paths over time")
st.caption(
    "Before tax, "
    + ("in today's euros" if adjust else "in future euros")
    + f". {len(result['final']):,} "
    f"paths; {len(samples.columns)} of them are drawn in grey.".replace(",", " ")
)
st.plotly_chart(charts.fan_chart(shown_bands, shown_samples), width="stretch")

st.subheader("Percentiles")
horizons = [(f"{years} years", years * 12) for years in HORIZON_YEARS if years * 12 < n_months]
horizons.append((f"End ({bands['date'].iloc[-1]:%m/%Y})", n_months))
table = pd.DataFrame(
    [
        {
            "Horizon": label,
            "10 %": format_eur(shown_bands[10].iloc[month]),
            "25 %": format_eur(shown_bands[25].iloc[month]),
            "Median": format_eur(shown_bands[50].iloc[month]),
            "75 %": format_eur(shown_bands[75].iloc[month]),
            "90 %": format_eur(shown_bands[90].iloc[month]),
            "Contributions (plan)": format_eur(shown_bands["contributions"].iloc[month]),
        }
        for label, month in horizons
    ]
)
st.dataframe(table, hide_index=True)
st.caption("10 %: one path out of ten ends below this value. Median: half of the paths end below.")

# final values, with tax and inflation if asked
final, contributions = result["final"], result["contributions"]
shown_final = final
note = "before tax"
if after_tax:
    horizon = bands["date"].iloc[-1].date()
    rate = tax_rate(envelope, settings.pea_open_date, horizon, settings)
    shown_final = net_final_values(final, contributions, rate)
    note = f"after tax at {format_pct(rate, 1)} on the gain of each path"
shown_final = shown_final / deflator[-1]
note += ", in today's euros" if adjust else ", in future euros"

st.subheader("Final values")
st.caption(f"Shown {note}.")
left, middle, right = st.columns(3)
left.metric("Median final value", format_eur(float(np.median(shown_final))))
if target:
    middle.metric(
        "Chance to reach the target",
        format_pct(probability_above(shown_final, target)),
        help=f"Share of the paths that end at {format_eur(target)} or more, as shown here.",
    )
right.metric(
    "Chance to end below your deposits",
    format_pct(float(np.mean(final < contributions))),
    help="Share of the paths whose final value, before tax, is below what was put in.",
)
st.plotly_chart(charts.histogram_chart(pd.Series(shown_final), target), width="stretch")
st.caption("The highest 1 % of the paths are left out of the chart, to keep the rest readable.")

st.subheader("What this model does and does not do")
st.markdown(
    f"""
- **Model used**: {result["model"]}.
- Each month is drawn independently of the others: no crash followed by a rebound, no good or
  bad market periods lasting several years. Real markets have both.
- Volatility is constant, and the mean return is your assumption, not a forecast.
- The tax uses today's rules and one rate for the whole final value; the PEA ceiling and the
  switch to a CTO are not modelled on this page (see Projection).
- A path never goes below zero: a planned withdrawal is limited to what is left.
"""
)
