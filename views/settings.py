import sqlite3

import pandas as pd
import streamlit as st

from simulator.database import add_asset, list_assets, update_asset
from simulator.models import Asset
from simulator.portfolio import asset_errors, target_weights_total
from views.common import format_pct, get_conn, set_flash, show_flash


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
    errors = asset_errors(asset)
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


def render_add_asset() -> None:
    with st.form("add_asset", border=False):
        inputs = asset_inputs(None, key="add_asset")
        submitted = st.form_submit_button("Add asset", type="primary")
    if submitted:
        asset = asset_from_inputs(None, *inputs)
        errors = save_asset(asset)
        if errors:
            st.error("\n\n".join(errors))
        else:
            # forget the typed values so the form comes back empty
            for field in ("name", "isin", "ticker", "fees", "weight"):
                del st.session_state[f"add_asset_{field}"]
            set_flash(f"{asset.name} added.")
            st.rerun()


def render_edit_asset(assets: list[Asset]) -> None:
    asset = st.selectbox("Asset to edit", assets, format_func=lambda a: a.name)
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
    render_add_asset()
if assets:
    with st.expander("Edit an asset"):
        render_edit_asset(assets)
