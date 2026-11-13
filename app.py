import streamlit as st

st.set_page_config(layout="wide", page_title="Long-term Invest Simulator")

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
    "": [st.Page("views/settings.py", title="Settings")],
}

page = st.navigation(pages)

with st.sidebar:
    st.caption("Educational tool. Not financial advice.")

page.run()
