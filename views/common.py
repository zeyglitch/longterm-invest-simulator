import sqlite3
from datetime import date

import pandas as pd
import streamlit as st

from simulator.database import list_plan_names, load_plan
from simulator.models import Plan
from simulator.prices import fetch_history


def format_eur(amount: float | None) -> str:
    """Euros with two decimals and a space between thousands: 12 345.67 €."""
    if amount is None:
        return "—"
    return f"{amount:,.2f} €".replace(",", " ")


def format_number(value: float | None, decimals: int = 2) -> str:
    """A number with a space between thousands, for table cells: 12 345.67."""
    if value is None:
        return "—"
    return f"{value:,.{decimals}f}".replace(",", " ")


def format_pct(rate: float | None, decimals: int = 1) -> str:
    """A decimal rate as a percentage: 0.1337 gives 13.4 %."""
    if rate is None:
        return "—"
    return f"{rate * 100:.{decimals}f} %"


def format_quantity(quantity: float) -> str:
    """A number of shares without useless zeros: 651 or 2.5 or 0.1234 (fractions come to 4)."""
    return f"{quantity:,.4f}".replace(",", " ").rstrip("0").rstrip(".")


def asset_label(asset) -> str:
    """Text of an asset in a dropdown. Streamlit finds the chosen option by its text, so two
    assets must never share one: the ISIN, which is unique, is always part of it."""
    return f"{asset.name} ({asset.isin})"


def get_conn() -> sqlite3.Connection:
    """The database chosen in app.py: the demo one or the real one."""
    return st.session_state["conn"]


def show_demo_banner() -> None:
    if st.session_state["demo"]:
        st.info("Demo data — fictional portfolio", icon=":material/science:")


def set_flash(message: str) -> None:
    """Keep a success message to show at the top of the page after the next rerun."""
    st.session_state["flash"] = message


def show_flash() -> None:
    message = st.session_state.pop("flash", None)
    if message:
        st.success(message)


DEFAULT_HORIZON_YEARS = 30


def data_mode() -> str:
    """`demo` or `real`: session values that depend on the database are stored once per mode."""
    return "demo" if st.session_state["demo"] else "real"


def get_plan() -> Plan:
    """The plan in use, shared by the simulation pages.

    The first time, it is the first saved plan, or an empty 30-year plan if none is saved.
    """
    key = f"plan_{data_mode()}"
    if key not in st.session_state:
        names = list_plan_names(get_conn())
        if names:
            st.session_state[key] = load_plan(get_conn(), names[0])
        else:
            end = date(date.today().year + DEFAULT_HORIZON_YEARS, 12, 31)
            st.session_state[key] = Plan(name="My plan", end=end, phases=[], one_offs=[])
    return st.session_state[key]


def cached_history(ticker: str) -> pd.Series:
    """Full price history of a Yahoo ticker, downloaded once per session. Empty if Yahoo fails."""
    key = f"history_{ticker}"
    if key not in st.session_state:
        history = fetch_history(ticker)
        if history.empty:
            return history  # not kept: the next click tries again
        st.session_state[key] = history
    return st.session_state[key]


def clear_mode_state(mode: str) -> None:
    """Forget what the pages keep for one database (`demo` or `real`): its plan on screen, its
    last simulation results... Needed when that database is replaced or emptied.

    `plan_version_{mode}` is kept: the plan tables get a new key, so they cannot show old edits.
    """
    for key in list(st.session_state):
        if key.endswith(f"_{mode}") and key != f"plan_version_{mode}":
            del st.session_state[key]
