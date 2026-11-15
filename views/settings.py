import sqlite3
import time
from datetime import date

import pandas as pd
import streamlit as st

from simulator import config
from simulator.database import (
    add_asset,
    delete_manual_price,
    get_float_setting,
    get_include_fees,
    get_last_price,
    get_price_source,
    list_assets,
    list_manual_prices,
    load_tax_settings,
    reset_database,
    save_prices,
    save_tax_settings,
    set_include_fees,
    set_setting,
    update_asset,
)
from simulator.importer import export_transactions
from simulator.models import Asset, TaxSettings
from simulator.portfolio import asset_errors, target_weights_total
from simulator.prices import fetch_last_price, find_ticker
from views.common import asset_label, format_eur, format_pct, get_conn, set_flash, show_flash


def asset_from_inputs(asset_id, name, isin, ticker, fees_pct, weight_pct) -> Asset:
    """Build an asset from the form fields. Percentages typed by the user become decimals."""
    return Asset(
        id=asset_id,
        isin=isin.strip().upper(),
        name=name.strip(),
        ticker=ticker.strip().upper() or None,
        fees=fees_pct / 100,
        target_weight=None if weight_pct is None else weight_pct / 100,
    )


def asset_inputs(asset: Asset | None, key: str):
    """The five fields of an asset, empty or filled with an existing asset."""
    name = st.text_input("Name", value=asset.name if asset else "", key=f"{key}_name")
    left, right = st.columns(2)
    isin = left.text_input("ISIN", value=asset.isin if asset else "", key=f"{key}_isin")
    ticker = right.text_input(
        "Yahoo ticker",
        value=(asset.ticker or "") if asset else "",
        key=f"{key}_ticker",
        help="Symbol used by Yahoo Finance to fetch prices, for example CW8.PA.",
    )
    fees_pct = left.number_input(
        "Yearly fees (%)",
        min_value=0.0,
        value=asset.fees * 100 if asset else 0.0,
        step=0.05,
        format="%.2f",
        key=f"{key}_fees",
        help="Ongoing charges of the fund, for example 0.20.",
    )
    has_weight = asset is not None and asset.target_weight is not None
    weight_pct = right.number_input(
        "Target weight (%)",
        min_value=0.0,
        max_value=100.0,
        value=asset.target_weight * 100 if has_weight else None,
        step=5.0,
        format="%.1f",
        key=f"{key}_weight",
        help="Share of the portfolio you aim for. Leave empty for an asset outside your plan.",
    )
    return name, isin, ticker, fees_pct, weight_pct


def save_asset(asset: Asset) -> list[str]:
    """Add or update the asset. Returns the error messages, empty if it was saved."""
    others = [other for other in list_assets(get_conn()) if other.id != asset.id]
    errors = asset_errors(asset, others)
    if errors:
        return errors
    try:
        if asset.id is None:
            add_asset(get_conn(), asset)
        else:
            update_asset(get_conn(), asset)
    except sqlite3.IntegrityError:
        return [f"Another asset already has the ISIN {asset.isin}."]
    return []


def render_asset_list(assets: list[Asset]) -> None:
    if not assets:
        st.info("No asset yet. Add the funds you hold below.")
        return
    table = pd.DataFrame(
        {
            "Name": [asset.name for asset in assets],
            "ISIN": [asset.isin for asset in assets],
            "Yahoo ticker": [asset.ticker or "—" for asset in assets],
            "Yearly fees": [format_pct(asset.fees, decimals=2) for asset in assets],
            "Target weight": [format_pct(asset.target_weight) for asset in assets],
        }
    )
    st.dataframe(table, hide_index=True)
    total = target_weights_total(assets)
    if total is not None and abs(total - 1) > 1e-6:
        st.warning(f"Target weights add up to {format_pct(total)} instead of 100 %.")


def add_form_key() -> str:
    """Key prefix of the add form. It changes after each asset added, which gives new, empty
    fields: deleting the old values is not enough, the browser sends them back."""
    return f"add_asset_{st.session_state.get('add_asset_count', 0)}"


def use_ticker_result() -> None:
    """Called when `Use this ticker` is clicked: test the ticker, then fill the add form."""
    result = st.session_state[f"{add_form_key()}_lookup_choice"]
    price = fetch_last_price(result["ticker"])
    if price is None:
        st.session_state["lookup_message"] = (
            "warning",
            f"{result['ticker']} gave no price: Yahoo may be unreachable, or this is not the "
            "right listing. Pick another one, or fill the form by hand.",
        )
        return
    day, close = price
    key = add_form_key()
    st.session_state[f"{key}_isin"] = st.session_state[f"{key}_lookup_isin"].strip().upper()
    st.session_state[f"{key}_ticker"] = result["ticker"]
    # always the name of the fund just found: a name left from another search would be wrong
    st.session_state[f"{key}_name"] = result["name"]
    st.session_state["lookup_message"] = (
        "success",
        f"{result['ticker']} works: last price {format_eur(close)} on {day:%d/%m/%Y}. "
        "The form below is filled in: check it and add the asset.",
    )


def render_ticker_lookup() -> None:
    """Find the Yahoo ticker of an ISIN. Outside the form below: a form cannot hold a search."""
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    isin = left.text_input(
        "Look up a ticker from an ISIN", key=f"{add_form_key()}_lookup_isin", max_chars=12
    )
    if right.button("Search", icon=":material/search:", key="lookup_search") and isin.strip():
        results = find_ticker(isin.strip().upper())
        st.session_state["lookup_results"] = results
        st.session_state["lookup_message"] = (
            None
            if results
            else ("warning", "Yahoo found nothing for this ISIN, or did not answer.")
        )
    results = st.session_state.get("lookup_results")
    if results:
        st.selectbox(
            "Matching listings",
            results,
            format_func=lambda r: f"{r['ticker']} — {r['name']} ({r['exchange']})",
            key=f"{add_form_key()}_lookup_choice",
            help="Paris listings come first. Choose the one in euros that you trade.",
        )
        st.button("Use this ticker", on_click=use_ticker_result)
    message = st.session_state.get("lookup_message")
    if message:
        kind, text = message
        (st.success if kind == "success" else st.warning)(text)


def render_add_asset() -> None:
    with st.form("add_asset", border=False):
        inputs = asset_inputs(None, key=add_form_key())
        submitted = st.form_submit_button("Add asset", type="primary")
    if submitted:
        asset = asset_from_inputs(None, *inputs)
        errors = save_asset(asset)
        if errors:
            st.error("\n\n".join(errors))
        else:
            # new empty fields for the next asset (see add_form_key)
            st.session_state["add_asset_count"] = st.session_state.get("add_asset_count", 0) + 1
            # the ticker search is over too: its message would describe an empty form
            for key in ("lookup_results", "lookup_message"):
                st.session_state.pop(key, None)
            set_flash(f"{asset.name} added.")
            st.rerun()


def render_edit_asset(assets: list[Asset]) -> None:
    asset = st.selectbox("Asset to edit", assets, format_func=asset_label)
    with st.form("edit_asset", border=False):
        inputs = asset_inputs(asset, key=f"edit_asset_{asset.id}")
        submitted = st.form_submit_button("Save changes", type="primary")
    if submitted:
        edited = asset_from_inputs(asset.id, *inputs)
        errors = save_asset(edited)
        if errors:
            st.error("\n\n".join(errors))
        else:
            set_flash(f"{edited.name} updated.")
            st.rerun()


st.title("Settings")
show_flash()

st.subheader("Assets")
assets = list_assets(get_conn())
render_asset_list(assets)
with st.expander("Add an asset", expanded=not assets):
    render_ticker_lookup()
    render_add_asset()
if assets:
    with st.expander("Edit an asset"):
        render_edit_asset(assets)


def render_price_table(assets: list[Asset]) -> None:
    rows = []
    for asset in assets:
        last = get_last_price(get_conn(), asset.id)
        rows.append(
            {
                "Asset": asset.name,
                "Last price": format_eur(last[1]) if last else "—",
                "Date": f"{last[0]:%d/%m/%Y}" if last else "—",
                "Source": get_price_source(get_conn(), asset.id, last[0]) if last else "—",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("`yahoo`: downloaded with Refresh prices. `manual`: typed by you.")


def render_manual_price(assets: list[Asset]) -> None:
    st.caption(
        "A manual price replaces the Yahoo price of that day. Use it when Yahoo has no price "
        "for an asset, or a wrong one."
    )
    with st.form("manual_price", border=False):
        left, middle, right = st.columns(3)
        asset = left.selectbox("Asset", assets, format_func=asset_label, key="price_asset")
        day = middle.date_input(
            "Date", value="today", max_value="today", format="DD/MM/YYYY", key="price_date"
        )
        price = right.number_input(
            "Price (€)", min_value=0.0, value=None, step=0.01, format="%.4f", key="price_value"
        )
        submitted = st.form_submit_button("Save manual price", type="primary")
    if not submitted:
        return
    if not price:
        st.error("The price must be greater than 0.")
        return
    save_prices(get_conn(), asset.id, pd.Series([price], index=[pd.Timestamp(day)]), "manual")
    set_flash(f"Manual price of {format_eur(price)} saved for {asset.name} on {day:%d/%m/%Y}.")
    st.rerun()


def render_manual_price_list(assets: list[Asset]) -> None:
    manual = list_manual_prices(get_conn())
    if not manual:
        return
    names = {asset.id: asset_label(asset) for asset in assets}
    choice = st.selectbox(
        "Manual prices",
        manual,
        format_func=lambda m: f"{names[m[0]]} — {m[1]:%d/%m/%Y} — {format_eur(m[2])}",
        key="manual_price_choice",
    )
    if st.button("Delete this manual price", key="manual_price_delete"):
        delete_manual_price(get_conn(), choice[0], choice[1])
        set_flash("Manual price deleted.")
        st.rerun()


if assets:
    st.subheader("Prices")
    render_price_table(assets)
    with st.expander("Enter a manual price"):
        render_manual_price(assets)
        render_manual_price_list(assets)


def percent_to_decimal(percent: float) -> float:
    """A rate typed as a percentage becomes a decimal, without floating point residue."""
    return round(percent / 100, 6)


def render_account() -> None:
    settings = load_tax_settings(get_conn())
    with st.form("account", border=False):
        left, right = st.columns(2)
        open_date = left.date_input(
            "PEA opening date",
            value=settings.pea_open_date,
            min_value=date(1992, 1, 1),
            max_value="today",
            format="DD/MM/YYYY",
            help="The day the PEA was opened: after 5 years, gains only bear social charges.",
        )
        ceiling = right.number_input(
            "PEA deposit ceiling (€)",
            min_value=1.0,
            value=float(settings.pea_ceiling),
            step=1000.0,
            format="%.0f",
            help="150 000 € in general; 20 000 € for a young adult attached to their parents' "
            "tax household.",
        )
        with_fees = st.checkbox(
            "Include order fees in the cost of shares",
            value=get_include_fees(get_conn()),
            help="The broker shows an average cost with fees included.",
        )
        submitted = st.form_submit_button("Save account settings", type="primary")
    if submitted:
        new = TaxSettings(settings.ps_rate, settings.ir_rate, ceiling, open_date)
        save_tax_settings(get_conn(), new)
        set_include_fees(get_conn(), with_fees)
        set_flash("Account settings saved.")
        st.rerun()


def render_tax_and_rates() -> None:
    conn = get_conn()
    settings = load_tax_settings(conn)
    checked = f"Default {{default}}, checked on {config.RATES_VERIFIED_ON}. Check the current rate."
    with st.form("tax_and_rates", border=False):
        left, right = st.columns(2)
        ps_pct = left.number_input(
            "Social charges (%)",
            min_value=0.0,
            max_value=50.0,
            value=settings.ps_rate * 100,
            step=0.1,
            format="%.2f",
            help=checked.format(default=format_pct(config.DEFAULT_PS_RATE, 2)),
        )
        ir_pct = right.number_input(
            "Income tax part of the flat tax (%)",
            min_value=0.0,
            max_value=50.0,
            value=settings.ir_rate * 100,
            step=0.1,
            format="%.2f",
            help=checked.format(default=format_pct(config.DEFAULT_IR_RATE, 2)),
        )
        livret_pct = left.number_input(
            "Livret A rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=get_float_setting(conn, "livret_rate", config.DEFAULT_LIVRET_RATE) * 100,
            step=0.05,
            format="%.2f",
            help=checked.format(default=format_pct(config.DEFAULT_LIVRET_RATE, 2)),
        )
        livret_ceiling = right.number_input(
            "Livret A ceiling (€)",
            min_value=1.0,
            value=get_float_setting(conn, "livret_ceiling", config.DEFAULT_LIVRET_CEILING),
            step=500.0,
            format="%.0f",
            help=checked.format(default=format_eur(config.DEFAULT_LIVRET_CEILING)),
        )
        inflation_pct = left.number_input(
            "Default inflation (%)",
            min_value=0.0,
            max_value=20.0,
            value=get_float_setting(conn, "inflation", config.DEFAULT_INFLATION) * 100,
            step=0.1,
            format="%.2f",
            help="An assumption, not a measure: used by default in the simulations.",
        )
        submitted = st.form_submit_button("Save rates", type="primary")
    st.caption(
        "Rates change: check them on an official source (service-public.fr, "
        "impots.gouv.fr, banque-france.fr) before relying on a simulation."
    )
    if submitted:
        new = TaxSettings(
            percent_to_decimal(ps_pct),
            percent_to_decimal(ir_pct),
            settings.pea_ceiling,
            settings.pea_open_date,
        )
        save_tax_settings(conn, new)
        set_setting(conn, "livret_rate", str(percent_to_decimal(livret_pct)))
        set_setting(conn, "livret_ceiling", str(livret_ceiling))
        set_setting(conn, "inflation", str(percent_to_decimal(inflation_pct)))
        set_flash("Rates saved.")
        st.rerun()


st.subheader("Account")
render_account()

st.subheader("Tax and rates")
render_tax_and_rates()


RESET_DELAY_SECONDS = 5
RESET_WORD = "ok"


@st.dialog("Delete all my data")
def reset_dialog() -> None:
    """Ask to type `ok`, then make the user wait 5 seconds before the delete button works."""
    st.warning(
        "This deletes your assets, trades, deposits, prices, account settings and rates. "
        "It cannot be undone.",
        icon=":material/warning:",
    )
    st.download_button(
        "Download a backup of my trades first (CSV)",
        data=export_transactions(get_conn()),
        file_name="transactions.csv",
        mime="text/csv",
        icon=":material/download:",
    )
    # one field per opening of the dialog: an `ok` typed in an earlier attempt never counts
    answer = st.text_input(
        f"Type {RESET_WORD} to confirm", key=f"reset_confirm_{st.session_state['reset_attempt']}"
    )

    typed = answer.strip().lower() == RESET_WORD
    if not typed:
        st.session_state.pop("reset_typed_at", None)  # the countdown restarts if `ok` is erased
    elif "reset_typed_at" not in st.session_state:
        st.session_state["reset_typed_at"] = time.monotonic()
    seconds_left = 0.0
    if typed:
        seconds_left = RESET_DELAY_SECONDS - (time.monotonic() - st.session_state["reset_typed_at"])

    if not typed:
        st.caption(f"Type {RESET_WORD} above, then wait {RESET_DELAY_SECONDS} seconds.")
    elif seconds_left > 0:
        st.caption(f"You can confirm in {int(seconds_left) + 1} s.")
    ready = typed and seconds_left <= 0
    if st.button("Delete everything", type="primary", icon=":material/delete:", disabled=not ready):
        reset_database(get_conn())
        # the names of the assets the last refresh could not update are gone too
        for key in ("reset_typed_at", "price_failures_real"):
            st.session_state.pop(key, None)
        set_flash("All your data has been deleted.")
        st.rerun()  # the whole page, which also closes the dialog

    if typed and seconds_left > 0:
        # wait a second, then redraw the dialog only: the countdown goes down
        time.sleep(1)
        st.rerun(scope="fragment")


def render_data() -> None:
    if st.session_state["demo"]:
        st.caption(
            "The demo data is rebuilt each time the app starts. "
            "Switch off Demo data in the sidebar to manage your own data."
        )
        return
    st.caption(
        "Start again from an empty portfolio, for example after trying the app with test numbers."
    )
    if st.button("Delete all my data", icon=":material/delete_forever:"):
        # start from scratch each time: new empty field, countdown not started
        st.session_state["reset_attempt"] = st.session_state.get("reset_attempt", 0) + 1
        st.session_state.pop("reset_typed_at", None)
        reset_dialog()


st.subheader("Data")
render_data()
