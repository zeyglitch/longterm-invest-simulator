import sqlite3

import streamlit as st


def format_eur(amount: float | None) -> str:
    """Euros with two decimals and a space between thousands: 12 345.67 €."""
    if amount is None:
        return "—"
    return f"{amount:,.2f} €".replace(",", " ")


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
