import os

import streamlit as st

from simulator import config
from simulator.database import get_connection, init_db, list_transactions
from simulator.demo import build_demo_db
from simulator.prices import latest_price_date, refresh_prices
from views.common import show_demo_banner

st.set_page_config(layout="wide", page_title="Long-term Invest Simulator")

# the online demo sets LTIS_DEMO=1: it never opens the real database
DEMO_ONLY = os.environ.get("LTIS_DEMO") == "1"


def choose_database() -> None:
    """Put the demo or the real connection in session_state["conn"], from the sidebar switch."""
    if "demo_conn" not in st.session_state:
        st.session_state["demo_conn"] = build_demo_db()
    if DEMO_ONLY:
        st.session_state["demo"] = True
        st.session_state["conn"] = st.session_state["demo_conn"]
        return

    if "real_conn" not in st.session_state:
        real_conn = get_connection(config.DB_PATH)
        init_db(real_conn)
        st.session_state["real_conn"] = real_conn
        # demo by default until the first real trade is entered
        st.session_state["demo_switch"] = len(list_transactions(real_conn)) == 0

    demo = st.sidebar.toggle(
        "Demo data",
        key="demo_switch",
        help="Fictional portfolio, kept in memory only. Switch off to use your own data.",
    )
    st.session_state["demo"] = demo
    st.session_state["conn"] = st.session_state["demo_conn" if demo else "real_conn"]


def render_price_refresh() -> None:
    """Sidebar button that downloads new prices, and the date of the latest one."""
    conn = st.session_state["conn"]
    if st.sidebar.button("Refresh prices", icon=":material/refresh:", width="stretch"):
        with st.spinner("Downloading prices from Yahoo Finance..."):
            failures = refresh_prices(conn)
        if failures:
            st.sidebar.warning(
                "No new price for: " + ", ".join(failures) + ". The last known prices are kept."
            )
        else:
            st.sidebar.success("Prices updated.")
    latest = latest_price_date(conn)
    st.sidebar.caption(f"Latest price: {latest:%d/%m/%Y}" if latest else "No price yet")


pages = {
    "Tracking": [
        st.Page("views/portfolio.py", title="Portfolio", default=True),
        st.Page("views/transactions.py", title="Transactions"),
    ],
    "Simulation": [
        st.Page("views/plan.py", title="Plan"),
        st.Page("views/projection.py", title="Projection"),
        st.Page("views/monte_carlo.py", title="Monte Carlo"),
        st.Page("views/compare.py", title="Compare"),
        st.Page("views/backtest.py", title="Backtest"),
        st.Page("views/withdrawal.py", title="Withdrawal"),
    ],
    "Configuration": [st.Page("views/settings.py", title="Settings")],
}

page = st.navigation(pages)
choose_database()

with st.sidebar:
    st.caption("Demo data" if st.session_state["demo"] else "My portfolio")
render_price_refresh()
st.sidebar.caption("Educational tool. Not financial advice.")

show_demo_banner()
page.run()
