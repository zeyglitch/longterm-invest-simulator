from datetime import date

import pandas as pd
import streamlit as st

from simulator import charts, config
from simulator.database import (
    get_float_setting,
    list_assets,
    load_tax_settings,
)
from simulator.models import TaxSettings
from simulator.plan import build_flow_parts, validate_plan
from simulator.portfolio import weighted_fees
from simulator.projection import (
    add_net_value,
    date_target_reached,
    fees_cost,
    net_return,
    project,
    project_livret,
    project_with_ceiling,
    to_real,
    yearly_table,
)
from simulator.taxes import CTO, PEA, pea_five_year_date, tax_rate
from views.common import data_mode, format_eur, format_number, format_pct, get_conn
from views.inputs import choose_plan, render_start

RESERVED_NAMES = ("date", "contributions", "livret a")


def render_scenarios() -> dict[str, float]:
    """Three scenarios: a name and a yearly gross return for each."""
    scenarios = {}
    columns = st.columns(3)
    for number, (name, rate) in enumerate(config.DEFAULT_SCENARIOS.items()):
        label = columns[number].text_input(
            f"Scenario {number + 1}", value=name, key=f"sc_name_{number}"
        )
        gross = columns[number].number_input(
            "Yearly gross return (%)",
            min_value=-20.0,
            max_value=30.0,
            value=rate * 100,
            step=0.5,
            format="%.2f",
            key=f"sc_rate_{number}",
        )
        # each name becomes a column of the chart: it must be unique, and not one of ours
        name = label.strip() or f"Scenario {number + 1}"
        if name in scenarios or name.lower() in RESERVED_NAMES:
            name = f"{name} ({number + 1})"
        scenarios[name] = round(gross / 100, 6)
    return scenarios


def displayed(table: pd.DataFrame, column: str, inflation: float | None) -> pd.Series:
    """A column of a projection, in today's euros if inflation is given."""
    return to_real(table[column], inflation) if inflation is not None else table[column]


st.title("Projection")
st.caption("A fixed yearly return, applied month by month to your portfolio and your plan.")

conn = get_conn()
today = date.today()
settings: TaxSettings = load_tax_settings(conn)

with st.container(border=True):
    left, right = st.columns(2)
    with left:
        plan = choose_plan("projection")
    with right:
        start_value, start_deposits, counted_in_ceiling = render_start("projection")
    scenarios = render_scenarios()

    left, middle, right = st.columns(3)
    fees = (
        left.number_input(
            "Yearly fees (%)",
            min_value=0.0,
            max_value=5.0,
            value=weighted_fees(list_assets(conn)) * 100,
            step=0.05,
            format="%.3f",
            key="projection_fees",
            help="Ongoing charges of the funds. Default: your assets' fees, weighted by their "
            "target weights.",
        )
        / 100
    )
    envelope = middle.segmented_control(
        "Envelope", [PEA, CTO], default=PEA, required=True, key="projection_envelope"
    )
    target = right.number_input(
        "Target amount (€, optional)",
        min_value=0.0,
        value=None,
        step=10000.0,
        key="projection_target",
    )

    cto_return = (
        left.number_input(
            "CTO yearly gross return (%)",
            min_value=-20.0,
            max_value=30.0,
            value=config.DEFAULT_CTO_RETURN * 100,
            step=0.5,
            format="%.2f",
            key="projection_cto_return",
            disabled=envelope == CTO,
            help="Deposits above the PEA ceiling go to a CTO, which has its own return. This "
            "one is used in the second scenario; the two others keep the same gap with it as "
            "the scenarios above. The same fees apply.",
        )
        / 100
    )

    left, middle, right = st.columns(3)
    after_tax = left.toggle("After tax", key="projection_after_tax")
    adjust = middle.toggle("Adjust for inflation", key="projection_adjust")
    index_deposits = right.toggle(
        "Index contributions on inflation",
        key="projection_index",
        help="The regular deposits of the plan grow with inflation, once a year.",
    )
    show_livret = left.toggle("Show Livret A", key="projection_livret")
    inflation = middle.number_input(
        "Inflation (%)",
        min_value=0.0,
        max_value=20.0,
        value=get_float_setting(conn, "inflation", config.DEFAULT_INFLATION) * 100,
        step=0.1,
        format="%.2f",
        key="projection_inflation",
    )
    inflation = inflation / 100

errors = validate_plan(plan)
if errors:
    st.error("This plan cannot be simulated:\n\n" + "\n\n".join(errors))
    st.page_link("views/plan.py", label="Fix it in Plan", icon=":material/edit:")
    st.stop()
# deposits and withdrawals apart: the PEA ceiling and the warnings look at one or the other
flows = build_flow_parts(plan, today, inflation, index_on_inflation=index_deposits)
if flows.empty:
    st.info("This plan ends before next month: choose a later end date in Plan.")
    st.page_link("views/plan.py", label="Go to Plan", icon=":material/edit:")
    st.stop()

# one projection per scenario, with the tax and the net value month by month
open_date = settings.pea_open_date
# gross return of the CTO in each scenario: the one typed, moved by the gap between the scenario
# and the second one (the central one by default)
reference = list(scenarios.values())[1]
cto_gross = {name: round(cto_return + gross - reference, 6) for name, gross in scenarios.items()}
projections = {}
for name, gross in scenarios.items():
    net = net_return(gross, fees)
    if envelope == PEA:
        table = project_with_ceiling(
            start_value,
            start_deposits,
            flows,
            net,
            settings.pea_ceiling,
            counted_in_ceiling,
            net_return(cto_gross[name], fees),
        )
    else:
        table = project(start_value, start_deposits, flows, net)
    projections[name] = add_net_value(table, envelope, open_date, settings)

measure = "net_value" if after_tax else "value"
real = inflation if adjust else None
first = next(iter(projections.values()))
curves = pd.DataFrame({"date": first["date"]})
for name, table in projections.items():
    curves[name] = displayed(table, measure, real)
curves["Contributions"] = displayed(first, "contributions", real)
if show_livret:
    livret_ceiling = get_float_setting(conn, "livret_ceiling", config.DEFAULT_LIVRET_CEILING)
    # a Livret A cannot hold more than its ceiling: the comparison starts with what fits in it
    livret = project_livret(
        min(start_value, livret_ceiling),
        flows,
        get_float_setting(conn, "livret_rate", config.DEFAULT_LIVRET_RATE),
        livret_ceiling,
    )
    curves["Livret A"] = displayed(livret, "value", real)

shown = ("net of tax" if after_tax else "before tax") + (
    ", in today's euros" if adjust else ", in future euros"
)
st.subheader("Value over time")
st.caption(f"Curves shown {shown}.")
st.plotly_chart(charts.projection_chart(curves, list(projections), target), width="stretch")
if show_livret:
    st.caption(
        "Livret A: same starting amount and same plan. Interest is added monthly here; the real "
        "one is computed by fortnight and added once a year. No tax on a Livret A."
    )
    refused = livret["not_invested"].sum() + max(start_value - livret_ceiling, 0.0)
    if refused > 0:
        st.caption(
            f"{format_eur(refused)} could not go in: Livret A ceiling of "
            f"{format_eur(livret_ceiling)} reached. That money is left out of its curve."
        )

# warnings
planned = (flows["deposit"] - flows["withdrawal"]).to_numpy()
limited = [
    name
    for name, table in projections.items()
    if (table["flow"].iloc[1:].to_numpy() > planned + 0.005).any()
]
if limited:
    st.warning(
        "A planned withdrawal is larger than the value at that time, and is limited to it, in: "
        + ", ".join(limited)
        + "."
    )
if envelope == PEA:
    over = first[first["contributions_cto"] > 0]
    if not over.empty:
        st.warning(
            f"PEA ceiling of {format_eur(settings.pea_ceiling)} reached in "
            f"{over['date'].iloc[0]:%B %Y}: deposits above it go to a CTO, with its own return "
            "and the flat tax on its gains. See PEA and CTO apart, below."
        )
    five_years = pea_five_year_date(open_date) if open_date else None
    early = [
        day
        for day, withdrawal in flows["withdrawal"].items()
        if withdrawal > 0 and (not five_years or day.date() < five_years)
    ]
    if early:
        reason = (
            f"before {five_years:%d/%m/%Y}" if five_years else "and the opening date is unknown"
        )
        st.warning(
            f"A withdrawal is planned in {early[0]:%B %Y}, {reason}: before 5 years it closes the "
            "PEA and the gain bears the flat tax. This tool does not simulate the closing."
        )

# kept for the Withdrawal page, which can start from the end of one of these scenarios
st.session_state[f"projection_result_{data_mode()}"] = {
    name: {
        "value": table["value"].iloc[-1],
        "contributions": table["contributions"].iloc[-1],
        "date": table["date"].iloc[-1].date(),
    }
    for name, table in projections.items()
}

# results at the horizon
st.subheader(f"At the horizon ({first['date'].iloc[-1]:%B %Y})")
months = len(flows)
horizon = first["date"].iloc[-1].date()
flat_tax = settings.ir_rate + settings.ps_rate
pocket_rates = {PEA: tax_rate(PEA, open_date, horizon, settings), CTO: flat_tax}
rows, pocket_rows, parts = [], [], []
for name, table in projections.items():
    last = table.iloc[-1]
    target_text = "—"
    if target:
        goal = pd.DataFrame({"date": table["date"], "value": curves[name]})
        reached = date_target_reached(goal, target)
        target_text = f"{reached:%m/%Y}" if reached else "not reached"
    if envelope == PEA:
        cost = fees_cost(
            start_value,
            start_deposits,
            flows,
            scenarios[name],
            fees,
            settings.pea_ceiling,
            counted_in_ceiling,
            cto_gross[name],
        )
    else:
        cost = fees_cost(start_value, start_deposits, flows, scenarios[name], fees)
    rows.append(
        {
            "Scenario": f"{name} ({format_pct(scenarios[name], 2)})",
            "Deposits": format_eur(last["contributions"]),
            "Gains": format_eur(last["gains"]),
            "Before tax": format_eur(last["value"]),
            "Tax": format_eur(last["tax"]),
            "After tax": format_eur(last["net_value"]),
            "After tax, today's euros": format_eur(
                last["net_value"] / (1 + inflation) ** (months / 12)
            ),
            "Cost of fees": format_eur(cost),
            "Target reached": target_text,
        }
    )
    # one line per account, then their total; a portfolio wholly in a CTO has one line only
    returns = {PEA: scenarios[name], CTO: cto_gross[name] if envelope == PEA else scenarios[name]}
    for account in (PEA, CTO) if envelope == PEA else (CTO,):
        pocket = account.lower()
        value, paid = last[f"value_{pocket}"], last[f"contributions_{pocket}"]
        pocket_rows.append(
            {
                "Scenario": name,
                "Account": account,
                "Gross return": format_pct(returns[account], 2),
                "Deposits": format_eur(paid),
                "Gains": format_eur(value - paid),
                "Before tax": format_eur(value),
                "Tax rate": format_pct(pocket_rates[account], 1),
                "Tax": format_eur(last[f"tax_{pocket}"]),
                "After tax": format_eur(last[f"net_{pocket}"]),
            }
        )
    if envelope == PEA:
        pocket_rows.append(
            {
                "Scenario": name,
                "Account": "Total",
                "Gross return": "",
                "Deposits": format_eur(last["contributions"]),
                "Gains": format_eur(last["gains"]),
                "Before tax": format_eur(last["value"]),
                "Tax rate": "",
                "Tax": format_eur(last["tax"]),
                "After tax": format_eur(last["net_value"]),
            }
        )
    parts.append(
        {"scenario": name, "contributions": last["contributions"], "gains": max(last["gains"], 0.0)}
    )
st.dataframe(pd.DataFrame(rows), hide_index=True)
st.caption(
    "Tax: if everything were withdrawn at the horizon, with today's rules and rates. "
    f"Today's euros: with {format_pct(inflation, 2)} of inflation a year. Cost of fees: gap with "
    "the same projection at 0 % of fees. Target: on the curve as shown above."
)

st.subheader("PEA and CTO apart" if envelope == PEA else "CTO")
st.dataframe(pd.DataFrame(pocket_rows), hide_index=True)
if envelope == PEA:
    st.caption(
        f"The PEA takes deposits up to its ceiling of {format_eur(settings.pea_ceiling)}; the "
        "rest goes to the CTO. Deposits: what was put in and not withdrawn. Tax = gains x tax "
        "rate, for each account on its own: a loss in one does not lower the tax of the other. "
        "Withdrawals of the plan are taken from the CTO first, then from the PEA."
    )
else:
    st.caption("The whole portfolio is in a CTO: the flat tax applies to all its gains.")

left, right = st.columns(2)
with left:
    st.subheader("Deposits and gains")
    st.plotly_chart(charts.gains_chart(pd.DataFrame(parts)), width="stretch")
with right:
    st.subheader("Year by year")
    yearly = yearly_table(curves).drop(columns=["date"]).round(2)
    yearly = yearly.rename(columns={"year": "Year"})
    amounts = [name for name in yearly.columns if name != "Year"]
    st.dataframe(
        # the cells stay numbers (sorting works); only their text gets spaces between thousands
        yearly.style.format(format_number, subset=amounts),
        hide_index=True,
        height=360,
        column_config={
            "Year": st.column_config.NumberColumn(format="%d"),
            **{name: st.column_config.Column(f"{name} (€)") for name in amounts},
        },
    )
    st.download_button(
        "Download (CSV)",
        data=yearly.to_csv(index=False),
        file_name="projection.csv",
        mime="text/csv",
        icon=":material/download:",
    )
st.caption(
    "A projection is not a forecast: real returns vary from year to year and can be negative."
)
