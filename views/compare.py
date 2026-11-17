from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from simulator import charts, config
from simulator.comparison import (
    Strategy,
    compare_envelopes,
    compare_strategies,
    dca_flows,
    lump_sum_flows,
    simulate_strategy,
)
from simulator.database import get_float_setting, list_assets, load_tax_settings
from simulator.montecarlo import LEVELS
from simulator.plan import build_flow_parts, plan_to_json, validate_plan
from simulator.portfolio import weighted_fees
from simulator.projection import net_return, to_real
from simulator.taxes import CTO, LIVRET, PEA
from views.common import data_mode, format_eur, format_pct, get_conn
from views.inputs import choose_plan, render_start

ENVELOPE_LABELS = {PEA: "PEA", CTO: "CTO", LIVRET: "Livret A"}
N_STRATEGIES = 4
PLAN, LUMP, SPREAD = "Plan", "Lump sum today", "Spread over N months"


def shown_curves(table: pd.DataFrame, after_tax: bool, inflation: float | None) -> pd.DataFrame:
    """One column per name of a comparison, net of tax and in today's euros if asked."""
    curves = pd.DataFrame()
    for name, rows in table.groupby("name", sort=False):
        values = rows["net_value" if after_tax else "value"].reset_index(drop=True)
        if "date" not in curves:
            curves["date"] = rows["date"].reset_index(drop=True)
        curves[name] = to_real(values, inflation) if inflation is not None else values
    return curves


def horizon_table(table: pd.DataFrame, label: str, inflation: float) -> pd.DataFrame:
    """The last month of each name: deposits, value before tax, tax, value after tax."""
    rows = []
    for name, months in table.groupby("name", sort=False):
        last = months.iloc[-1]
        years = (len(months) - 1) / 12
        rows.append(
            {
                label: name,
                "Deposits": format_eur(last["contributions"]),
                "Before tax": format_eur(last["value"]),
                "Tax": format_eur(last["tax"]),
                "After tax": format_eur(last["net_value"]),
                "After tax, today's euros": format_eur(
                    last["net_value"] / (1 + inflation) ** years
                ),
            }
        )
    return pd.DataFrame(rows)


def render_envelopes(start, flows, settings, inflation, after_tax, adjust) -> None:
    start_value, start_deposits, counted_in_ceiling = start
    conn = get_conn()
    left, right = st.columns(2)
    gross = (
        left.number_input(
            "Yearly gross return, PEA and CTO (%)",
            min_value=-20.0,
            max_value=30.0,
            value=config.DEFAULT_SCENARIOS["Central"] * 100,
            step=0.5,
            format="%.2f",
            key="cmp_env_return",
        )
        / 100
    )
    fees = (
        right.number_input(
            "Yearly fees, PEA and CTO (%)",
            min_value=0.0,
            max_value=5.0,
            value=weighted_fees(list_assets(conn)) * 100,
            step=0.05,
            format="%.3f",
            key="cmp_env_fees",
        )
        / 100
    )
    livret_rate = get_float_setting(conn, "livret_rate", config.DEFAULT_LIVRET_RATE)
    livret_ceiling = get_float_setting(conn, "livret_ceiling", config.DEFAULT_LIVRET_CEILING)
    table = compare_envelopes(
        start_value,
        start_deposits,
        flows,
        net_return(gross, fees),
        settings,
        livret_rate,
        livret_ceiling,
        counted_in_ceiling,
    )
    table["name"] = table["name"].map(ENVELOPE_LABELS)

    curves = shown_curves(table, after_tax, inflation if adjust else None)
    paid = table[table["name"] == "PEA"]["contributions"].reset_index(drop=True)
    curves["Contributions"] = to_real(paid, inflation) if adjust else paid
    st.caption(
        "Same starting point, same plan. Curves shown "
        + ("after tax" if after_tax else "before tax")
        + (", in today's euros." if adjust else ", in future euros.")
        + ("" if after_tax else " Before tax, PEA and CTO are one and the same curve.")
    )
    st.plotly_chart(charts.comparison_chart(curves), width="stretch")

    st.subheader(f"At the horizon ({table['date'].iloc[-1]:%B %Y})")
    st.dataframe(horizon_table(table, "Envelope", inflation), hide_index=True)
    refused = table[table["name"] == "Livret A"]["not_invested"].sum()
    st.caption(
        f"PEA and CTO earn the same {format_pct(net_return(gross, fees), 2)} a year after fees: "
        "the gap between them comes from the tax only. Tax: if everything were withdrawn at the "
        f"horizon, with today's rules. Livret A at {format_pct(livret_rate, 2)}, without tax. "
        f"Above its ceiling of {format_eur(settings.pea_ceiling)}, the PEA sends deposits to a "
        "CTO with the same return here."
    )
    if refused > 0:
        st.warning(
            f"The Livret A is full: {format_eur(refused)} could not go in (ceiling of "
            f"{format_eur(livret_ceiling)}) and is left out of its line. The two other envelopes "
            "hold that money, so the three final values are not the result of the same deposits."
        )


def preset_lump_sum() -> None:
    """`DCA vs lump sum`: the same amount, at once or spread, everything else alike."""
    names = ("Lump sum", "Spread over 12 months")
    for number, (name, kind) in enumerate(zip(names, (LUMP, SPREAD), strict=True)):
        st.session_state[f"cmp_{number}_on"] = True
        st.session_state[f"cmp_{number}_name"] = name
        st.session_state[f"cmp_{number}_kind"] = kind
        st.session_state[f"cmp_{number}_amount"] = 12_000.0
        st.session_state[f"cmp_{number}_months"] = 12
        for field in ("envelope", "return", "volatility", "fees"):
            st.session_state[f"cmp_{number}_{field}"] = st.session_state[f"cmp_0_{field}"]
    for number in range(2, N_STRATEGIES):
        st.session_state[f"cmp_{number}_on"] = False


def preset_mixed() -> None:
    """`100% equities vs 60/40`: the same plan, two return and volatility assumptions."""
    choices = (
        ("100% equities", config.DEFAULT_SCENARIOS["Central"], config.DEFAULT_VOLATILITY),
        ("60/40", config.DEFAULT_MIXED_RETURN, config.DEFAULT_MIXED_VOLATILITY),
    )
    for number, (name, mean, volatility) in enumerate(choices):
        st.session_state[f"cmp_{number}_on"] = True
        st.session_state[f"cmp_{number}_name"] = name
        st.session_state[f"cmp_{number}_kind"] = PLAN
        st.session_state[f"cmp_{number}_return"] = mean * 100
        st.session_state[f"cmp_{number}_volatility"] = volatility * 100
    for number in range(2, N_STRATEGIES):
        st.session_state[f"cmp_{number}_on"] = False


def strategy_inputs(number: int, column, default_fees: float) -> dict | None:
    """The fields of one strategy. Returns what was typed, or None if it is switched off."""
    key = f"cmp_{number}"
    defaults = {
        "on": number < 2,
        "name": f"Strategy {number + 1}",
        "envelope": PEA,
        "return": config.DEFAULT_SCENARIOS["Central"] * 100,
        "volatility": config.DEFAULT_VOLATILITY * 100,
        "fees": default_fees * 100,
        "kind": PLAN,
        "amount": 12_000.0,
        "months": 12,
    }
    for field, value in defaults.items():
        st.session_state.setdefault(f"{key}_{field}", value)

    # every field is always drawn: Streamlit forgets the value of a field it stops drawing
    active = column.checkbox("Compare it", key=f"{key}_on")
    name = column.text_input("Name", key=f"{key}_name", disabled=not active)
    envelope = column.selectbox(
        "Envelope",
        [PEA, CTO, LIVRET],
        format_func=ENVELOPE_LABELS.get,
        key=f"{key}_envelope",
        disabled=not active,
    )
    livret = envelope == LIVRET
    mean = column.number_input(
        "Yearly return (%)",
        min_value=-20.0,
        max_value=30.0,
        step=0.5,
        format="%.2f",
        key=f"{key}_return",
        disabled=not active,
        help="Gross return of the funds. For a Livret A: its interest rate.",
    )
    volatility = column.number_input(
        "Yearly volatility (%)",
        min_value=0.0,
        max_value=80.0,
        step=1.0,
        format="%.2f",
        key=f"{key}_volatility",
        disabled=not active or livret,
        help="Only used by the Monte Carlo run below.",
    )
    fees = column.number_input(
        "Yearly fees (%)",
        min_value=0.0,
        max_value=5.0,
        step=0.05,
        format="%.3f",
        key=f"{key}_fees",
        disabled=not active or livret,
    )
    kind = column.selectbox(
        "Deposits", [PLAN, LUMP, SPREAD], key=f"{key}_kind", disabled=not active
    )
    amount = column.number_input(
        "Amount (€)",
        min_value=0.0,
        step=1000.0,
        key=f"{key}_amount",
        disabled=not active or kind == PLAN,
    )
    months = column.number_input(
        "N (months)",
        min_value=1,
        step=1,
        key=f"{key}_months",
        disabled=not active or kind != SPREAD,
    )
    if not active:
        return None
    return {
        "name": name.strip() or defaults["name"],
        "envelope": envelope,
        "return": round(mean / 100, 6),
        "volatility": 0.0 if livret else volatility / 100,
        "fees": 0.0 if livret else fees / 100,
        "kind": kind,
        "amount": amount,
        "months": int(months),
    }


def build_strategies(choices: list[dict], flows: pd.DataFrame) -> list[Strategy]:
    """Turn what was typed into strategies. Each name is made unique: it becomes a column."""
    strategies = []
    for number, choice in enumerate(choices):
        name = choice["name"]
        if name in [s.name for s in strategies] or name.lower() == "date":
            name = f"{name} ({number + 1})"
        initial, own_flows = 0.0, flows
        if choice["kind"] == LUMP:
            initial, own_flows = lump_sum_flows(choice["amount"], flows.index)
        elif choice["kind"] == SPREAD:
            n_months = min(choice["months"], len(flows))
            initial, own_flows = dca_flows(choice["amount"], n_months, flows.index)
        strategies.append(
            Strategy(
                name,
                choice["envelope"],
                choice["return"],
                choice["volatility"],
                choice["fees"],
                own_flows,
                initial,
            )
        )
    return strategies


def monte_carlo_table(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, paths in results.items():
        levels = np.percentile(paths["value"], LEVELS)
        rows.append(
            {
                "Strategy": name,
                "10 %": format_eur(levels[0]),
                "25 %": format_eur(levels[1]),
                "Median": format_eur(levels[2]),
                "75 %": format_eur(levels[3]),
                "90 %": format_eur(levels[4]),
                "Ends below its deposits": format_pct(
                    float(np.mean(paths["value"] < paths["contributions"]))
                ),
                "Typical worst fall": f"-{format_pct(paths['worst_fall'].median())}",
            }
        )
    return pd.DataFrame(rows)


def render_monte_carlo(strategies: list[Strategy], start, signature: tuple) -> None:
    start_value, start_deposits, _ = start
    st.subheader("Monte Carlo")
    random = [s for s in strategies if s.envelope != LIVRET]
    if not random:
        st.caption("A Livret A has no random part: nothing to draw.")
        return
    left, middle, right = st.columns(3, vertical_alignment="bottom")
    n_paths = left.number_input(
        "Number of paths",
        min_value=100,
        max_value=20_000,
        value=config.DEFAULT_N_PATHS,
        step=1000,
        key="cmp_paths",
    )
    seed = middle.number_input(
        "Random seed", min_value=0, value=config.DEFAULT_SEED, step=1, key="cmp_seed"
    )
    inputs = (signature, int(n_paths), int(seed))
    result_key = f"cmp_mc_{data_mode()}"
    if right.button("Run Monte Carlo", icon=":material/casino:"):
        st.session_state[result_key] = {
            "inputs": inputs,
            "paths": {
                s.name: simulate_strategy(s, start_value, start_deposits, int(n_paths), int(seed))
                for s in random
            },
        }
    result = st.session_state.get(result_key)
    if result is None:
        st.caption(
            "The curves above use one fixed return. Run Monte Carlo to see how spread out the "
            "final value of each strategy can be."
        )
        return
    if result["inputs"] != inputs:
        st.warning("The settings changed since this run: click Run Monte Carlo to update it.")
    st.dataframe(monte_carlo_table(result["paths"]), hide_index=True)
    st.caption(
        "Final values before tax, in future euros. Every strategy gets the same random draws: "
        "only its return, its volatility and its deposits differ. The PEA ceiling is not applied "
        "in this run. Livret A strategies are left out: their result is certain."
    )
    with st.expander("How to read this table?"):
        name, paths = next(iter(result["paths"].items()))
        low, median, high = (format_eur(v) for v in np.percentile(paths["value"], (10, 50, 90)))
        st.markdown(
            f"""
Each line sorts the possible futures of one strategy, from the worst to the best. For
**{name}**:

- **10 %**: in 1 future out of 10, it ends with less than **{low}**.
- **Median**: half of its futures end below **{median}** and half above.
- **90 %**: only 1 future out of 10 ends above **{high}**.
- **Ends below its deposits**: share of the futures where it is worth less at the end than
  what was put in.
- **Typical worst fall**: in half of its futures, the market drops at least that much from an
  earlier high before recovering.

A strategy with a higher median but a much lower 10 % column pays its extra return with more
risk: compare the lines column by column, not only on the median.
"""
        )


def render_strategies(start, flows, plan, settings, inflation, after_tax, adjust) -> None:
    start_value, start_deposits, counted_in_ceiling = start
    conn = get_conn()
    left, right, _ = st.columns([1, 1, 2])
    left.button("Preset: DCA vs lump sum", on_click=preset_lump_sum, width="stretch")
    right.button("Preset: 100% equities vs 60/40", on_click=preset_mixed, width="stretch")
    default_fees = weighted_fees(list_assets(conn))
    columns = st.columns(N_STRATEGIES)
    choices = [strategy_inputs(n, columns[n], default_fees) for n in range(N_STRATEGIES)]
    choices = [choice for choice in choices if choice is not None]
    st.caption(
        "Every strategy starts from the starting point above. `Plan`: the deposits of the plan. "
        "`Lump sum today`: the amount is invested at once, with no other deposit. `Spread over N "
        "months`: the same amount in N equal monthly deposits. The 60/40 preset uses assumptions "
        "for its return and volatility: change them as you see fit."
    )
    if not choices:
        st.info("Tick `Compare it` on at least one strategy.")
        return

    strategies = build_strategies(choices, flows)
    livret_ceiling = get_float_setting(conn, "livret_ceiling", config.DEFAULT_LIVRET_CEILING)
    table = compare_strategies(
        strategies, start_value, start_deposits, settings, livret_ceiling, counted_in_ceiling
    )
    st.caption(
        "One fixed return for each strategy. Curves shown "
        + ("after tax" if after_tax else "before tax")
        + (", in today's euros." if adjust else ", in future euros.")
    )
    curves = shown_curves(table, after_tax, inflation if adjust else None)
    st.plotly_chart(charts.comparison_chart(curves), width="stretch")

    st.subheader(f"At the horizon ({table['date'].iloc[-1]:%B %Y})")
    st.dataframe(horizon_table(table, "Strategy", inflation), hide_index=True)
    if any(choice["kind"] != PLAN for choice in choices):
        st.caption(
            "With a fixed positive return, money invested earlier always ends higher: a lump "
            "sum beats spread deposits by construction. The interest of this comparison is in "
            "the Monte Carlo run below, which shows the spread of the results."
        )
    over = [
        s.name
        for s in strategies
        if s.envelope == PEA and counted_in_ceiling + s.initial_deposit > settings.pea_ceiling
    ]
    if over:
        st.warning(
            "The lump sum goes above the PEA ceiling in: " + ", ".join(over) + ". This page puts "
            "it all in the PEA; in reality the surplus would have to go to a CTO."
        )
    if table["not_invested"].sum() > 0:
        st.warning(
            "A Livret A strategy is full: the deposits above its ceiling of "
            f"{format_eur(livret_ceiling)} are left out of its line."
        )

    signature = (
        plan_to_json(plan),
        start_value,
        start_deposits,
        tuple(tuple(choice.items()) for choice in choices),
    )
    render_monte_carlo(strategies, start, signature)


st.title("Compare")
st.caption("The same money in several envelopes, or invested in several ways, side by side.")

conn = get_conn()
today = date.today()
settings = load_tax_settings(conn)

with st.container(border=True):
    left, right = st.columns(2)
    with left:
        plan = choose_plan("compare")
    with right:
        start = render_start("compare")
    left, middle, right = st.columns(3)
    after_tax = left.toggle("After tax", key="cmp_after_tax")
    adjust = middle.toggle("Adjust for inflation", key="cmp_adjust")
    inflation = (
        right.number_input(
            "Inflation (%)",
            min_value=0.0,
            max_value=20.0,
            value=get_float_setting(conn, "inflation", config.DEFAULT_INFLATION) * 100,
            step=0.1,
            format="%.2f",
            key="cmp_inflation",
        )
        / 100
    )

errors = validate_plan(plan)
if errors:
    st.error("This plan cannot be simulated:\n\n" + "\n\n".join(errors))
    st.page_link("views/plan.py", label="Fix it in Plan", icon=":material/edit:")
    st.stop()
flows = build_flow_parts(plan, today)
if flows.empty:
    st.info("This plan ends before next month: choose a later end date in Plan.")
    st.page_link("views/plan.py", label="Go to Plan", icon=":material/edit:")
    st.stop()

envelopes_tab, strategies_tab = st.tabs(["Envelopes", "Strategies"])
with envelopes_tab:
    render_envelopes(start, flows, settings, inflation, after_tax, adjust)
with strategies_tab:
    render_strategies(start, flows, plan, settings, inflation, after_tax, adjust)
