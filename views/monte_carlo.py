from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from simulator import charts, config
from simulator.database import get_float_setting, list_assets, load_tax_settings
from simulator.montecarlo import (
    LEVELS,
    bootstrap_returns,
    estimate_mu_sigma,
    net_final_values,
    normal_returns,
    percentiles,
    probability_above,
    random_unemployment,
    run_paths,
    run_paths_with_ceiling,
    worst_falls,
)
from simulator.plan import build_flow_parts, plan_to_json, validate_plan
from simulator.portfolio import weighted_fees
from simulator.prices import monthly_returns
from simulator.projection import net_return, project
from simulator.taxes import CTO, PEA, tax_rate
from views.common import cached_history, data_mode, format_eur, format_pct, get_conn
from views.inputs import choose_plan, render_start

NORMAL = "Normal returns"
BOOTSTRAP = "Historical bootstrap"
SAMPLE_PATHS = 20
MIN_HISTORY_MONTHS = 60
HORIZON_YEARS = (10, 20, 30)


def describe_history(history: pd.Series) -> str:
    months = len(monthly_returns(history))
    return f"{history.index[0]:%m/%Y} to {history.index[-1]:%m/%Y} ({months} months)"


def estimate_from_history() -> None:
    """Called by `Estimate from history`: fill the mean return and the volatility."""
    ticker = st.session_state["mc_ticker"].strip().upper()
    history = cached_history(ticker)
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
st.session_state.setdefault("mc_cto_mu", config.DEFAULT_CTO_RETURN * 100)
st.session_state.setdefault("mc_sigma", config.DEFAULT_VOLATILITY * 100)
# same default as the Projection page, unrounded: with no volatility the two pages agree
st.session_state.setdefault("mc_fees", weighted_fees(list_assets(conn)) * 100)
st.session_state.setdefault("mc_ticker", config.DEFAULT_BENCHMARK)

with st.container(border=True):
    left, right = st.columns(2)
    with left:
        plan = choose_plan("mc")
    with right:
        start_value, start_deposits, counted_in_ceiling = render_start("mc")

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
    cto_mu = left.number_input(
        "CTO mean yearly return (%)",
        min_value=-20.0,
        max_value=30.0,
        step=0.5,
        format="%.2f",
        key="mc_cto_mu",
        disabled=envelope == CTO or not normal,
        help="Deposits above the PEA ceiling go to a CTO. It gets the same random ups and downs "
        "as the PEA, around this mean return, with the same fees.",
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
    (settings.pea_ceiling, counted_in_ceiling, cto_mu if normal else None)
    if envelope == PEA
    else None,
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
            history = cached_history(ticker.strip().upper())
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
    withdrawals = flows["withdrawal"].to_numpy()
    if envelope == PEA:
        cto_extra = 0.0
        if normal:
            # the CTO gets the same draws as the PEA, moved to its own mean return
            cto_mean = net_return(cto_mu / 100, fees / 100)
            cto_extra = (np.log(1 + cto_mean) - np.log(1 + mean_return)) / 12
            model += f"; CTO above the PEA ceiling: mean {format_pct(cto_mean, 2)} after fees"
        else:
            model += "; CTO above the PEA ceiling: same returns as the PEA"
        paths, pockets = run_paths_with_ceiling(
            start_value,
            deposits,
            returns,
            settings.pea_ceiling,
            counted_in_ceiling,
            start_deposits,
            withdrawals,
            cto_extra,
        )
    else:
        # the whole portfolio is a CTO: one pocket, the PEA one stays empty
        paths, paid = run_paths(start_value, deposits, returns, start_deposits, withdrawals)
        pockets = pd.DataFrame(
            {
                "value_pea": 0.0,
                "value_cto": paths[:, -1],
                "contributions_pea": 0.0,
                "contributions_cto": paid,
            }
        )
    planned = project(start_value, start_deposits, flows, mean_return)
    bands = percentiles(paths)
    bands.insert(0, "date", planned["date"])
    bands["contributions"] = planned["contributions"]
    st.session_state[result_key] = {
        "inputs": inputs,
        "bands": bands,
        "samples": pd.DataFrame(paths[:SAMPLE_PATHS].T),
        "pockets": pockets,
        "falls": worst_falls(returns),
        "model": model,
    }

result = st.session_state.get(result_key)
if result is None or "falls" not in result:
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

n_drawn = f"{len(result['pockets']):,}".replace(",", " ")
st.subheader("Spread of the paths over time")
st.caption(
    "Before tax, "
    + ("in today's euros" if adjust else "in future euros")
    + f". {n_drawn} paths; {len(samples.columns)} of them are drawn in grey."
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
with st.expander("What do these percentages mean?"):
    end = {level: format_eur(shown_bands[level].iloc[-1]) for level in LEVELS}
    st.markdown(
        f"""
The simulation drew {n_drawn} possible futures. Sort them by their value on a given date, from
the worst to the best: each column is a position in that ranking. With the numbers of the last
line (the end of your plan, {bands["date"].iloc[-1]:%m/%Y}), before tax:

- **10 %**: in 1 future out of 10, you end with less than **{end[10]}**. This is the unlucky
  case; 9 times out of 10 you do better.
- **25 %**: in 1 future out of 4, you end with less than **{end[25]}**.
- **Median**: half of the futures end below **{end[50]}** and half above. It is the middle
  result, not a promise.
- **75 %**: in 3 futures out of 4, you end with less than **{end[75]}**; 1 time out of 4 you
  do better.
- **90 %**: only 1 future out of 10 ends above **{end[90]}**. This is the lucky case.
- **Contributions (plan)**: what you will have put in by then,
  **{format_eur(shown_bands["contributions"].iloc[-1])}**, with no gain and no loss. Compare
  it with the other columns to see what the market added or took away.

In short: half of the futures end between **{end[25]}** and **{end[75]}**, and 8 out of 10
between **{end[10]}** and **{end[90]}**. The other lines read the same way, at their own date.
"""
    )

# final values of each path, PEA and CTO apart, with the tax of each
pockets = result["pockets"]
horizon = bands["date"].iloc[-1].date()
rates = {
    PEA: tax_rate(PEA, settings.pea_open_date, horizon, settings),
    CTO: tax_rate(CTO, None, horizon, settings),
}
ends = pd.DataFrame()
for account, rate in rates.items():
    pocket = account.lower()
    value, paid = pockets[f"value_{pocket}"], pockets[f"contributions_{pocket}"]
    ends[f"paid_{pocket}"] = paid
    ends[f"value_{pocket}"] = value
    ends[f"net_{pocket}"] = net_final_values(value, paid, rate)
ends = ends / deflator[-1]
final = (pockets["value_pea"] + pockets["value_cto"]).to_numpy()
contributions = (pockets["contributions_pea"] + pockets["contributions_cto"]).to_numpy()
before_tax = (ends["value_pea"] + ends["value_cto"]).to_numpy()
net_of_tax = (ends["net_pea"] + ends["net_cto"]).to_numpy()

shown_final = net_of_tax if after_tax else before_tax
note = "after tax on the gain of each path" if after_tax else "before tax"
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

split = (pockets["value_pea"] > 0).any()
st.subheader("PEA and CTO apart" if split else "Before and after tax")
euros = "today's euros" if adjust else "future euros"
st.caption(f"Average of all the paths at the end, in {euros}.")
average = ends.mean()
rows = []
for account in (PEA, CTO) if split else (CTO,):
    pocket = account.lower()
    value, paid, net = (
        average[f"value_{pocket}"],
        average[f"paid_{pocket}"],
        average[f"net_{pocket}"],
    )
    rows.append(
        {
            "Account": account,
            "Deposits": format_eur(paid),
            "Gains": format_eur(value - paid),
            "Before tax": format_eur(value),
            "Tax rate": format_pct(rates[account], 1),
            "Tax": format_eur(value - net),
            "After tax": format_eur(net),
        }
    )
if split:
    rows.append(
        {
            "Account": "Total",
            "Deposits": format_eur(average["paid_pea"] + average["paid_cto"]),
            "Gains": format_eur(before_tax.mean() - average["paid_pea"] - average["paid_cto"]),
            "Before tax": format_eur(before_tax.mean()),
            "Tax rate": "",
            "Tax": format_eur(before_tax.mean() - net_of_tax.mean()),
            "After tax": format_eur(net_of_tax.mean()),
        }
    )
st.dataframe(pd.DataFrame(rows), hide_index=True)
st.caption(
    "An average adds up (PEA + CTO = total, before tax - tax = after tax), but a few very good "
    "paths pull it above the median. Tax: on the gain of each path and each account, if "
    "everything were withdrawn at the end; a path in loss owes nothing."
    + (
        f" The PEA takes deposits up to its ceiling of {format_eur(settings.pea_ceiling)}; "
        "the rest goes to the CTO."
        if split
        else ""
    )
)

st.caption("Spread of the total at the end. Each column is ranked on its own.")
spread = pd.DataFrame(
    {
        "Level": ["10 %", "25 %", "Median", "75 %", "90 %"],
        "Before tax": [format_eur(v) for v in np.percentile(before_tax, LEVELS)],
        "Tax": [format_eur(v) for v in np.percentile(before_tax - net_of_tax, LEVELS)],
        "After tax": [format_eur(v) for v in np.percentile(net_of_tax, LEVELS)],
    }
)
st.dataframe(spread, hide_index=True)
with st.expander("How to read these two tables?"):
    low, middle_net, high = (format_eur(v) for v in np.percentile(net_of_tax, (10, 50, 90)))
    st.markdown(
        f"""
- **Before tax** is what the accounts are worth at the end. **Tax** is what you would owe if
  you withdrew everything on that day: only the gains are taxed, never your own deposits.
  **After tax** is what would really reach your bank account.
- **First table**: the average of the {n_drawn} futures. On average you end with
  **{format_eur(before_tax.mean())}**, owe **{format_eur(before_tax.mean() - net_of_tax.mean())}**
  of tax and keep **{format_eur(net_of_tax.mean())}**.
- **Second table**: the same ranking as the percentiles above, at the end of the plan. After
  tax, in 1 future out of 10 you keep less than **{low}**; half of the futures leave you less
  than **{middle_net}** and half more; only 1 out of 10 leaves you more than **{high}**.
- The average is higher than the median because a few very lucky futures pull it up: the
  median is the better picture of a typical result.
"""
    )

st.subheader("Worst fall along the way")
falls = result["falls"]
mild, typical, severe = np.percentile(falls, (10, 50, 90))
left, middle, right = st.columns(3)
left.metric(
    "Typical worst fall",
    f"-{format_pct(typical)}",
    help="Median: half of the paths go through a fall at least this deep.",
)
middle.metric(
    "1 path out of 10 falls by more than",
    f"-{format_pct(severe)}",
)
right.metric(
    "Paths that fall by 30 % or more",
    format_pct(float(np.mean(falls >= 0.30))),
)
st.caption(
    "Largest drop from an earlier high, at any time before the end, measured on the market "
    "itself (the price of one share), before tax. The final values above do not show it: a "
    "path can end well after a deep fall."
)
with st.expander("What does this mean for me?"):
    example = 100_000
    st.markdown(
        f"""
The tables above only look at the end. This one looks at the road: at some point, each future
goes through its worst moment, when the market is far below a high it had reached before.

- **Typical worst fall, -{format_pct(typical)}**: in half of the futures, the market loses at
  least that much from a high before recovering. On {format_eur(example)} invested at the high,
  you would see about **{format_eur(example * (1 - typical))}** on your screen at the low.
- **1 path out of 10, more than -{format_pct(severe)}**: in the roughest futures,
  {format_eur(example)} at the high become less than **{format_eur(example * (1 - severe))}**.
- **Even the calm futures fall**: 9 out of 10 go through a drop of at least
  **-{format_pct(mild)}**.
- These falls are not final losses: the paths go on afterwards, and most end well above your
  deposits. The loss becomes real only if you sell at the low.
- The question this number asks: would you keep investing every month while your account
  shows such a fall? The final value only counts if you stay invested until the end.
- Your account can show a smaller fall than the market, because your monthly deposits keep
  adding money during the drop.
"""
    )

st.subheader("What this model does and does not do")
st.markdown(
    f"""
- **Model used**: {result["model"]}.
- Each month is drawn independently of the others: no crash followed by a rebound, no good or
  bad market periods lasting several years. Real markets have both.
- Volatility is constant, and the mean return is your assumption, not a forecast.
- The tax uses today's rules and rates, as if everything were withdrawn at the end. With the
  PEA envelope, deposits above the PEA ceiling go to a CTO, taxed at the flat tax.
- The CTO moves with the PEA (same random draws), only its mean return differs.
- The worst fall is seen at the end of each month only: a drop and a rebound inside a month
  are not counted.
- A path never goes below zero: a planned withdrawal is limited to what is left.
"""
)
