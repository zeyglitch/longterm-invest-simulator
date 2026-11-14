from datetime import date

import pandas as pd
import streamlit as st

from simulator.database import (
    add_cash_movement,
    add_transaction,
    delete_cash_movement,
    delete_transaction,
    list_assets,
    list_cash_movements,
    list_transactions,
    update_transaction,
)
from simulator.importer import export_transactions
from simulator.models import Asset, CashMovement, Transaction
from simulator.portfolio import first_oversold_sale, trade_errors
from views.common import format_eur, get_conn, set_flash, show_flash

KINDS = {"Buy": "BUY", "Sell": "SELL"}
CASH_KINDS = {"Deposit": "DEPOSIT", "Withdrawal": "WITHDRAWAL"}
KIND_LABELS = {"BUY": "Buy", "SELL": "Sell", "DEPOSIT": "Deposit", "WITHDRAWAL": "Withdrawal"}
DATE_FORMAT = "DD/MM/YYYY"


def trade_total(kind: str, quantity: float, unit_price: float, fees: float) -> float:
    """Money leaving the account for a buy, coming back for a sell."""
    if kind == "BUY":
        return quantity * unit_price + fees
    return quantity * unit_price - fees


def save_new_trade() -> None:
    """Called when `Add trade` is clicked: check the fields, save, then empty the form."""
    state = st.session_state
    conn = get_conn()
    asset = state["trade_asset"]
    tx = Transaction(
        id=None,
        date=state["trade_date"],
        asset_id=asset.id,
        kind=KINDS[state["trade_kind"]],
        quantity=state["trade_quantity"] or 0.0,
        unit_price=state["trade_price"] or 0.0,
        fees=state["trade_fees"],
        note=state["trade_note"].strip(),
    )
    errors = trade_errors(tx, list_transactions(conn), date.today())
    if errors:
        state["trade_form_errors"] = errors
        return

    add_transaction(conn, tx)
    total = trade_total(tx.kind, tx.quantity, tx.unit_price, tx.fees)
    message = f"{state['trade_kind']} of {tx.quantity:g} {asset.name} saved ({format_eur(total)})."
    if tx.kind == "BUY" and state["trade_deposit"]:
        deposit = CashMovement(None, tx.date, "DEPOSIT", total, note=f"Buy {asset.name}")
        add_cash_movement(conn, deposit)
        message += " Deposit of the same amount recorded."
    set_flash(message)
    # date, asset and fees are kept: several trades are often typed in a row
    state["trade_quantity"] = None
    state["trade_price"] = None
    state["trade_note"] = ""


def render_add_trade(assets: list[Asset]) -> None:
    left, middle, right = st.columns(3)
    left.date_input("Date", value="today", max_value="today", format=DATE_FORMAT, key="trade_date")
    middle.selectbox("Asset", assets, format_func=lambda asset: asset.name, key="trade_asset")
    kind_label = right.segmented_control(
        "Type", list(KINDS), default="Buy", required=True, key="trade_kind"
    )
    # a second row of columns keeps the three fields aligned whatever the height of the first row
    left, middle, right = st.columns(3)
    quantity = left.number_input(
        "Quantity",
        min_value=0.0,
        value=None,
        step=1.0,
        format="%.6f",
        placeholder="Number of shares",
        key="trade_quantity",
        help="Fractions are allowed, as in savings plans.",
    )
    unit_price = middle.number_input(
        "Unit price (€)",
        min_value=0.0,
        value=None,
        step=0.01,
        format="%.4f",
        placeholder="Price of one share",
        key="trade_price",
    )
    fees = right.number_input(
        "Fees (€)",
        min_value=0.0,
        value=0.0,
        step=1.0,
        format="%.2f",
        key="trade_fees",
        help="Order fee: 0 for a savings plan, usually 1 for a manual order.",
    )
    st.text_input("Note (optional)", key="trade_note")
    is_buy = kind_label == "Buy"
    st.checkbox(
        "Also record a deposit of the same amount",
        value=True,
        disabled=not is_buy,
        key="trade_deposit",
        help="Untick if you paid with cash that was already on the account.",
    )

    total = trade_total(KINDS[kind_label], quantity or 0.0, unit_price or 0.0, fees)
    label = "Total to pay" if is_buy else "Amount received"
    st.markdown(f"**{label}: {format_eur(total)}** — compare it with your broker's statement.")
    st.button("Add trade", type="primary", on_click=save_new_trade)

    errors = st.session_state.pop("trade_form_errors", None)
    if errors:
        st.error("\n\n".join(errors))


def render_add_cash_movement() -> None:
    with st.form("add_cash", clear_on_submit=True, border=False):
        left, middle, right = st.columns(3)
        day = left.date_input("Date", value="today", max_value="today", format=DATE_FORMAT)
        kind_label = middle.segmented_control(
            "Type", list(CASH_KINDS), default="Deposit", required=True
        )
        amount = right.number_input(
            "Amount (€)", min_value=0.01, value=None, step=50.0, format="%.2f"
        )
        note = st.text_input("Note (optional)")
        submitted = st.form_submit_button("Add cash movement", type="primary")
    if not submitted:
        return
    if amount is None:
        st.error("The amount is required.")
        return
    movement = CashMovement(None, day, CASH_KINDS[kind_label], amount, note.strip())
    add_cash_movement(get_conn(), movement)
    set_flash(f"{kind_label} of {format_eur(amount)} saved.")
    st.rerun()


@st.dialog("Edit trade")
def edit_trade_dialog(tx: Transaction, assets: list[Asset]) -> None:
    key = f"edit_trade_{tx.id}"
    current_asset = [asset.id for asset in assets].index(tx.asset_id)
    day = st.date_input(
        "Date", value=tx.date, max_value="today", format=DATE_FORMAT, key=f"{key}_d"
    )
    asset = st.selectbox(
        "Asset", assets, index=current_asset, format_func=lambda a: a.name, key=f"{key}_a"
    )
    kind_label = st.segmented_control(
        "Type", list(KINDS), default=KIND_LABELS[tx.kind], required=True, key=f"{key}_k"
    )
    quantity = st.number_input(
        "Quantity", min_value=0.0, value=tx.quantity, step=1.0, format="%.6f", key=f"{key}_q"
    )
    unit_price = st.number_input(
        "Unit price (€)", min_value=0.0, value=tx.unit_price, format="%.4f", key=f"{key}_p"
    )
    fees = st.number_input(
        "Fees (€)", min_value=0.0, value=tx.fees, step=1.0, format="%.2f", key=f"{key}_f"
    )
    note = st.text_input("Note (optional)", value=tx.note, key=f"{key}_n")
    if tx.kind == "BUY":
        st.caption("A deposit recorded with this buy is not changed: see the cash movements tab.")

    if st.button("Save changes", type="primary"):
        edited = Transaction(
            id=tx.id,
            date=day,
            asset_id=asset.id,
            kind=KINDS[kind_label],
            quantity=quantity,
            unit_price=unit_price,
            fees=fees,
            note=note.strip(),
            source=tx.source,
        )
        others = [other for other in list_transactions(get_conn()) if other.id != tx.id]
        errors = trade_errors(edited, others, date.today())
        if errors:
            st.error("\n\n".join(errors))
        else:
            update_transaction(get_conn(), edited)
            set_flash("Trade updated.")
            st.rerun()


@st.dialog("Delete trade")
def delete_trade_dialog(tx: Transaction, asset_name: str) -> None:
    st.write(
        f"{KIND_LABELS[tx.kind]} of {tx.quantity:g} {asset_name} "
        f"at {format_eur(tx.unit_price)} on {tx.date:%d/%m/%Y}."
    )
    others = [other for other in list_transactions(get_conn()) if other.id != tx.id]
    blocked = first_oversold_sale(others)
    if blocked is not None:
        st.error(
            f"This buy cannot be deleted: the sale of {blocked.date:%d/%m/%Y} would sell more "
            "shares than held on that date. Edit or delete that sale first."
        )
        return
    if tx.kind == "BUY":
        st.caption("A deposit recorded with this buy is not deleted: see the cash movements tab.")
    if st.button("Delete this trade", type="primary"):
        delete_transaction(get_conn(), tx.id)
        set_flash("Trade deleted.")
        st.rerun()


@st.dialog("Delete cash movement")
def delete_cash_dialog(movement: CashMovement) -> None:
    st.write(
        f"{KIND_LABELS[movement.kind]} of {format_eur(movement.amount)} "
        f"on {movement.date:%d/%m/%Y}."
    )
    if st.button("Delete this cash movement", type="primary"):
        delete_cash_movement(get_conn(), movement.id)
        set_flash("Cash movement deleted.")
        st.rerun()


def filter_trades(transactions: list[Transaction], assets: list[Asset]) -> list[Transaction]:
    """Show the three filters and return the matching trades, most recent first."""
    left, middle, right = st.columns(3)
    asset = left.selectbox(
        "Asset",
        assets,
        index=None,
        format_func=lambda a: a.name,
        placeholder="All assets",
        key="history_asset",
    )
    kind_label = middle.segmented_control(
        "Type", ["All", "Buy", "Sell"], default="All", required=True, key="history_kind"
    )
    period = right.date_input(
        "Period",
        value=(transactions[0].date, date.today()),
        format=DATE_FORMAT,
        key="history_period",
    )
    shown = []
    for tx in reversed(transactions):
        if asset is not None and tx.asset_id != asset.id:
            continue
        if kind_label != "All" and tx.kind != KINDS[kind_label]:
            continue
        # while the user is picking the second date, the period has only one
        if len(period) == 2 and not period[0] <= tx.date <= period[1]:
            continue
        shown.append(tx)
    return shown


def render_trade_history(assets: list[Asset]) -> None:
    transactions = list_transactions(get_conn())
    if not transactions:
        st.info("No trade yet.")
        return
    shown = filter_trades(transactions, assets)
    names = {asset.id: asset.name for asset in assets}
    table = pd.DataFrame(
        {
            "Date": [f"{tx.date:%d/%m/%Y}" for tx in shown],
            "Asset": [names[tx.asset_id] for tx in shown],
            "Type": [KIND_LABELS[tx.kind] for tx in shown],
            "Quantity": [f"{tx.quantity:g}" for tx in shown],
            "Unit price": [format_eur(tx.unit_price) for tx in shown],
            "Fees": [format_eur(tx.fees) for tx in shown],
            "Total": [
                format_eur(trade_total(tx.kind, tx.quantity, tx.unit_price, tx.fees))
                for tx in shown
            ],
            "Note": [tx.note for tx in shown],
        }
    )
    st.caption(f"{len(shown)} of {len(transactions)} trades. Select a row to edit or delete it.")
    event = st.dataframe(
        table,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="trade_history",
    )
    if event.selection.rows:
        selected = shown[event.selection.rows[0]]
        with st.container(horizontal=True):
            if st.button("Edit", icon=":material/edit:", key="edit_trade"):
                edit_trade_dialog(selected, assets)
            if st.button("Delete", icon=":material/delete:", key="delete_trade"):
                delete_trade_dialog(selected, names[selected.asset_id])


def render_cash_history() -> None:
    movements = list(reversed(list_cash_movements(get_conn())))
    if not movements:
        st.info("No cash movement yet.")
        return
    table = pd.DataFrame(
        {
            "Date": [f"{movement.date:%d/%m/%Y}" for movement in movements],
            "Type": [KIND_LABELS[movement.kind] for movement in movements],
            "Amount": [format_eur(movement.amount) for movement in movements],
            "Note": [movement.note for movement in movements],
        }
    )
    st.caption(f"{len(movements)} cash movements. Select a row to delete it.")
    event = st.dataframe(
        table,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="cash_history",
    )
    if event.selection.rows:
        selected = movements[event.selection.rows[0]]
        if st.button("Delete", icon=":material/delete:", key="delete_cash"):
            delete_cash_dialog(selected)


st.title("Transactions")
show_flash()

assets = list_assets(get_conn())
if not assets:
    st.info("Create your assets first, then come back to enter your trades.")
    st.page_link("views/settings.py", label="Go to Settings", icon=":material/settings:")
    st.stop()

with st.container(border=True):
    st.subheader("Add a trade")
    render_add_trade(assets)

with st.expander("Add a cash movement"):
    st.caption("Money moved between your bank and the account, without buying anything.")
    render_add_cash_movement()

st.subheader("History")
trades_tab, cash_tab = st.tabs(["Trades", "Cash movements"])
with trades_tab:
    render_trade_history(assets)
    st.download_button(
        "Export all trades (CSV)",
        data=export_transactions(get_conn()),
        file_name="transactions.csv",
        mime="text/csv",
        icon=":material/download:",
        help="One line per trade. Also a simple backup of what you typed.",
    )
with cash_tab:
    render_cash_history()
