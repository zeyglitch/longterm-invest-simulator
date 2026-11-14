"""Positions and average cost from the list of trades (rules in section 1 of the finance notes)."""

from datetime import date

import pandas as pd

from simulator.models import Asset, CashMovement, Position, Transaction

# below this, a quantity is a rounding residue of fractional shares: the position is closed
CLOSED_QUANTITY = 1e-9


def _in_processing_order(transactions: list[Transaction]) -> list[Transaction]:
    """Trades sorted by date, then by id (entry order) on the same day."""
    return sorted(transactions, key=lambda tx: (tx.date, tx.id or 0))


def _replay(transactions: list[Transaction], include_fees: bool) -> tuple[dict[int, dict], list]:
    """Replay the trades with the weighted average cost method.

    Returns the quantity and cost held per asset id, and one record per sale with its gain.
    """
    holdings: dict[int, dict] = {}
    sales = []
    for tx in _in_processing_order(transactions):
        holding = holdings.setdefault(tx.asset_id, {"quantity": 0.0, "cost": 0.0})
        fees = tx.fees if include_fees else 0.0
        if tx.kind == "BUY":
            holding["quantity"] += tx.quantity
            holding["cost"] += tx.quantity * tx.unit_price + fees
            continue

        # a sale removes shares at the average cost: the average cost itself does not change
        avg_cost = holding["cost"] / holding["quantity"]
        gain = tx.quantity * tx.unit_price - fees - tx.quantity * avg_cost
        sales.append((tx, avg_cost, gain))
        holding["quantity"] -= tx.quantity
        holding["cost"] -= tx.quantity * avg_cost
        if holding["quantity"] < CLOSED_QUANTITY:
            holding["quantity"] = 0.0
            holding["cost"] = 0.0
    return holdings, sales


def compute_positions(
    assets: list[Asset],
    transactions: list[Transaction],
    last_prices: dict[int, tuple[date, float]],
    include_fees: bool = True,
) -> list[Position]:
    """Open positions, one per asset still held. `last_prices` maps asset id to (date, price)."""
    holdings, _ = _replay(transactions, include_fees)
    positions = []
    for asset in assets:
        holding = holdings.get(asset.id)
        if holding is None or holding["quantity"] <= 0:
            continue
        quantity, cost = holding["quantity"], holding["cost"]
        price_date, last_price = last_prices.get(asset.id, (None, None))
        value = gain = gain_pct = None
        if last_price is not None:
            value = quantity * last_price
            gain = value - cost
            gain_pct = gain / cost
        positions.append(
            Position(
                asset=asset,
                quantity=quantity,
                avg_cost=cost / quantity,
                invested=cost,
                last_price=last_price,
                price_date=price_date,
                value=value,
                gain=gain,
                gain_pct=gain_pct,
                weight=None,
            )
        )

    # weights are shares of the positions whose value is known
    total_value = sum(p.value for p in positions if p.value is not None)
    for position in positions:
        if position.value is not None and total_value > 0:
            position.weight = position.value / total_value
    return positions


def realised_gains(
    assets: list[Asset], transactions: list[Transaction], include_fees: bool = True
) -> pd.DataFrame:
    """One row per sale: date, asset, quantity, sale price, average cost, realised gain."""
    names = {asset.id: asset.name for asset in assets}
    _, sales = _replay(transactions, include_fees)
    rows = [
        {
            "date": tx.date,
            "asset": names.get(tx.asset_id, ""),
            "quantity": tx.quantity,
            "sale_price": tx.unit_price,
            "avg_cost": avg_cost,
            "gain": gain,
        }
        for tx, avg_cost, gain in sales
    ]
    columns = ["date", "asset", "quantity", "sale_price", "avg_cost", "gain"]
    return pd.DataFrame(rows, columns=columns)


def quantity_held(transactions: list[Transaction], asset_id: int, on_date: date) -> float:
    """Number of shares of an asset held at the end of `on_date` (trades of that day included)."""
    quantity = 0.0
    for tx in transactions:
        if tx.asset_id != asset_id or tx.date > on_date:
            continue
        quantity += tx.quantity if tx.kind == "BUY" else -tx.quantity
    return quantity


def total_deposits(cash_movements: list[CashMovement]) -> float:
    """Net money put on the account: deposits minus withdrawals (used for tax)."""
    total = 0.0
    for movement in cash_movements:
        total += movement.amount if movement.kind == "DEPOSIT" else -movement.amount
    return total


def gross_deposits(cash_movements: list[CashMovement]) -> float:
    """Sum of deposits only, for the PEA ceiling: a withdrawal gives no capacity back."""
    return sum(m.amount for m in cash_movements if m.kind == "DEPOSIT")
