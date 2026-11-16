from dataclasses import replace
from datetime import date

import pandas as pd
import streamlit as st

from simulator import charts
from simulator.database import delete_plan, list_plan_names, load_plan, save_plan
from simulator.models import Plan
from simulator.plan import (
    add_big_purchase,
    add_child,
    add_home_purchase,
    add_job_change,
    add_unemployment,
    add_windfall,
    build_flows,
    plan_from_frames,
    plan_to_frames,
    validate_plan,
)
from views.common import (
    DEFAULT_HORIZON_YEARS,
    data_mode,
    format_eur,
    get_conn,
    get_plan,
    set_flash,
    show_flash,
)

DATE_FORMAT = "DD/MM/YYYY"
EVENTS = [
    "New job / raise",
    "Unemployment",
    "Child",
    "Home purchase",
    "Car / big purchase",
    "Windfall",
]


def show_plan(plan: Plan) -> None:
    """Replace the plan shown in the page (after a load, a helper, a new plan...).

    The tables keep what the user typed as long as their key is the same. A new key gives new
    tables, filled with the new plan.
    """
    mode = data_mode()
    st.session_state[f"plan_{mode}"] = plan
    st.session_state[f"plan_shown_{mode}"] = plan
    st.session_state[f"plan_version_{mode}"] = st.session_state.get(f"plan_version_{mode}", 0) + 1


def empty_plan(today: date) -> Plan:
    end = date(today.year + DEFAULT_HORIZON_YEARS, 12, 31)
    return Plan(name="My plan", end=end, phases=[], one_offs=[])


def render_saved_plans(today: date) -> None:
    """Load a saved plan, or start a new one."""
    names = list_plan_names(get_conn())
    left, middle, right = st.columns([3, 1, 1], vertical_alignment="bottom")
    chosen = left.selectbox(
        "Saved plans", names, index=None, placeholder="Choose a plan to load", key="plan_to_load"
    )
    if middle.button("Load plan", disabled=chosen is None, width="stretch"):
        show_plan(load_plan(get_conn(), chosen))
        set_flash(f"{chosen} loaded.")
        st.rerun()
    if right.button("New plan", width="stretch"):
        show_plan(empty_plan(today))
        st.rerun()


def render_editors(shown: Plan, key: str, today: date) -> tuple[Plan, int]:
    """Name, horizon and the two editable tables. Returns the plan as edited, and the number
    of rows left out because they are incomplete."""
    left, right = st.columns([2, 1])
    name = left.text_input("Plan name", value=shown.name, key=f"{key}_name")
    end = right.date_input(
        "End of the simulation",
        value=shown.end,
        min_value=today,
        max_value=date(today.year + 80, 12, 31),
        format=DATE_FORMAT,
        key=f"{key}_end",
        help="Last month simulated. 30 years from now by default.",
    )
    phases, one_offs = plan_to_frames(shown)

    st.subheader("Contribution phases")
    st.caption(
        "One line per period of regular deposits. An amount of 0 is a pause; "
        "a month covered by no line is worth 0 too. Two lines cannot cover the same month."
    )
    edited_phases = st.data_editor(
        phases,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key=f"{key}_phases",
        column_config={
            "label": st.column_config.TextColumn("Label"),
            "start": st.column_config.DateColumn("Start", format=DATE_FORMAT),
            "end": st.column_config.DateColumn("End", format=DATE_FORMAT),
            "monthly_amount": st.column_config.NumberColumn(
                "Monthly amount (€)", min_value=0.0, format="%.2f"
            ),
            "increase_pct": st.column_config.NumberColumn(
                "Yearly increase (%)",
                format="%.2f",
                help="Optional. Raises the amount each year, on the anniversary of the line.",
            ),
        },
    )

    st.subheader("One-off events")
    st.caption("Positive: an extra deposit (bonus, gift). Negative: a withdrawal (home, car).")
    edited_one_offs = st.data_editor(
        one_offs,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key=f"{key}_one_offs",
        column_config={
            "label": st.column_config.TextColumn("Label"),
            "date": st.column_config.DateColumn("Date", format=DATE_FORMAT),
            "amount": st.column_config.NumberColumn("Amount (€)", format="%.2f"),
        },
    )
    return plan_from_frames(name, end, edited_phases, edited_one_offs)


def render_preview(plan: Plan, today: date) -> None:
    st.subheader("Preview")
    regular = build_flows(replace(plan, one_offs=[]), today)
    events = build_flows(replace(plan, phases=[]), today)
    if regular.empty:
        st.caption("The simulation ends before next month: nothing to show.")
        return
    months = pd.PeriodIndex(regular.index, freq="M")
    in_horizon = [o for o in plan.one_offs if pd.Period(o.date, freq="M") in months]
    markers = pd.DataFrame(
        {
            "date": [
                pd.Period(o.date, freq="M").to_timestamp(how="end").normalize() for o in in_horizon
            ],
            "amount": [o.amount for o in in_horizon],
            "label": [o.label or "One-off" for o in in_horizon],
        }
    )

    left, middle, right = st.columns(3)
    left.metric(
        "Total deposits",
        format_eur(regular.sum() + events[events > 0].sum()),
        help="Regular deposits plus one-off deposits, until the end of the simulation.",
    )
    middle.metric("Total withdrawals", format_eur(-events[events < 0].sum()))
    right.metric("Months simulated", len(regular))
    st.plotly_chart(charts.contributions_chart(regular, markers), width="stretch")
    skipped = len(plan.one_offs) - len(in_horizon)
    if skipped:
        st.caption(f"{skipped} one-off event(s) outside the simulated period are ignored.")


def apply_event(plan: Plan, event: str, today: date) -> Plan | None:
    """The form of one life event. Returns the changed plan once the form is submitted."""
    with st.form(f"event_{event}", border=False):
        left, middle, right = st.columns(3)
        day = left.date_input("Date", value=today, min_value=today, format=DATE_FORMAT)
        if event == "New job / raise":
            amount = middle.number_input("New monthly amount (€)", min_value=0.0, step=50.0)
        elif event == "Unemployment":
            months = middle.number_input("Duration (months)", min_value=1, value=6, step=1)
        elif event == "Child":
            amount = middle.number_input("Monthly decrease (€)", min_value=0.0, step=50.0)
            years = right.number_input("Duration (years)", min_value=1, value=18, step=1)
        elif event == "Home purchase":
            down_payment = middle.number_input(
                "Down payment withdrawn (€)", min_value=0.0, step=1000.0
            )
            amount = right.number_input("Monthly decrease (€)", min_value=0.0, step=50.0)
            years = left.number_input("Loan duration (years)", min_value=1, value=20, step=1)
        else:
            amount = middle.number_input("Amount (€)", min_value=0.0, step=500.0)
        submitted = st.form_submit_button("Add to the plan", type="primary")
    if not submitted:
        return None
    if event == "New job / raise":
        return add_job_change(plan, day, amount)
    if event == "Unemployment":
        return add_unemployment(plan, day, int(months))
    if event == "Child":
        return add_child(plan, day, amount, int(years))
    if event == "Home purchase":
        return add_home_purchase(plan, day, down_payment, amount, int(years))
    if event == "Car / big purchase":
        return add_big_purchase(plan, day, amount)
    return add_windfall(plan, day, amount)


def render_events(plan: Plan, valid: bool, today: date) -> None:
    with st.expander("Life event helpers"):
        st.caption(
            "A helper writes lines in the two tables for you; you can edit them afterwards. "
            "Life events only change the deposits: this tool does not simulate a budget "
            "(salary, rent, loan payments)."
        )
        if not valid:
            st.info("Fix the plan first: a helper needs a valid plan to work on.")
            return
        event = st.selectbox("Event", EVENTS, key="plan_event")
        changed = apply_event(plan, event, today)
        if changed is None:
            return
        errors = validate_plan(changed)
        if errors:
            st.error("\n\n".join(errors))
            return
        show_plan(changed)
        set_flash(f"{event} added to the plan.")
        st.rerun()


def render_actions(plan: Plan, valid: bool) -> None:
    conn = get_conn()
    saved = load_plan(conn, plan.name)
    left, middle, right, status = st.columns([1, 1, 1, 2], vertical_alignment="center")
    if left.button("Save plan", type="primary", disabled=not valid, width="stretch"):
        save_plan(conn, plan)
        show_plan(plan)
        set_flash(f"{plan.name} saved.")
        st.rerun()
    if middle.button("Duplicate", disabled=not valid, width="stretch"):
        show_plan(replace(plan, name=f"{plan.name} (copy)"))
        set_flash("Copy created: rename it if you wish, then save it.")
        st.rerun()
    with right.popover("Delete", disabled=saved is None, width="stretch"):
        st.write(f"Delete the saved plan {plan.name}? This cannot be undone.")
        if st.button("Yes, delete it", type="primary"):
            delete_plan(conn, plan.name)
            set_flash(f"{plan.name} deleted.")
            st.rerun()
    if saved is None:
        status.caption("Not saved yet.")
    elif saved == plan:
        status.caption("Saved.")
    else:
        status.caption("Changes not saved.")


st.title("Plan")
show_flash()
st.caption(
    "The future of your deposits. The Projection, Monte Carlo, Compare and Withdrawal pages "
    "all use the plan shown here."
)

today = date.today()
mode = data_mode()
if f"plan_shown_{mode}" not in st.session_state:
    show_plan(get_plan())

render_saved_plans(today)
editor_key = f"plan_{mode}_{st.session_state[f'plan_version_{mode}']}"
plan, incomplete = render_editors(st.session_state[f"plan_shown_{mode}"], editor_key, today)
# the other pages read the plan as it is typed here
st.session_state[f"plan_{mode}"] = plan

errors = validate_plan(plan)
if errors:
    st.error("\n\n".join(errors))
if incomplete:
    st.warning(f"{incomplete} line(s) without a date or an amount are left out for now.")

render_actions(plan, valid=not errors)
render_events(plan, valid=not errors, today=today)
if not errors:
    render_preview(plan, today)
