"""Positions, average cost and realised gains, with the weighted average cost method."""

from datetime import date

import pandas as pd

from simulator.models import Asset, CashMovement, Position, Transaction

# below this, a quantity is a rounding residue of fractional shares: the position is closed
CLOSED_QUANTITY = 1e-9


def _in_processing_order(transactions: list[Transaction]) -> list[Transaction]:
    """Trades sorted by date, then by id (entry order) on the same day.

    A trade not saved yet has no id: it comes last on its day, as it will once saved.
    """
    return sorted(
        transactions, key=lambda tx: (tx.date, tx.id if tx.id is not None else float("inf"))
    )


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


def _price_on(prices: pd.Series | None, day: pd.Timestamp, fallback: float) -> float:
    """Last known price on or before `day`, never a later one; `fallback` if there is none."""
    if prices is None or prices.empty:
        return fallback
    price = prices.asof(day)
    return fallback if pd.isna(price) else float(price)


def portfolio_history(
    transactions: list[Transaction],
    price_history: dict[int, pd.Series],
    freq: str = "W",
    include_fees: bool = True,
    end: date | None = None,
) -> pd.DataFrame:
    """Value and invested amount of the portfolio on a weekly grid, as `date`, `value`, `invested`.

    The grid goes from the first trade to `end` (default: today); both ends are always in it.
    `price_history` maps an asset id to its closing prices. On a day without a price, the last
    earlier price is used. Before any price exists, the last trade price stands in for it.
    """
    columns = ["date", "value", "invested"]
    ordered = _in_processing_order(transactions)
    if not ordered:
        return pd.DataFrame(columns=columns)

    start = ordered[0].date
    stop = max(end or date.today(), start)
    weekly = pd.date_range(start, stop, freq=freq)
    days = sorted({pd.Timestamp(start), pd.Timestamp(stop), *weekly})

    rows = []
    for day in days:
        known = [tx for tx in ordered if tx.date <= day.date()]
        holdings, _ = _replay(known, include_fees)
        last_trade_price = {tx.asset_id: tx.unit_price for tx in known}
        value = invested = 0.0
        for asset_id, holding in holdings.items():
            if holding["quantity"] <= 0:
                continue
            invested += holding["cost"]
            price = _price_on(price_history.get(asset_id), day, last_trade_price[asset_id])
            value += holding["quantity"] * price
        rows.append((day, value, invested))
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


def first_oversold_sale(transactions: list[Transaction]) -> Transaction | None:
    """The first sale that sells more shares than held at that moment, or None if all are fine."""
    held: dict[int, float] = {}
    for tx in _in_processing_order(transactions):
        change = tx.quantity if tx.kind == "BUY" else -tx.quantity
        held[tx.asset_id] = held.get(tx.asset_id, 0.0) + change
        # the tolerance accepts 0.3 - 0.1 - 0.2, which is not exactly 0 in floating point
        if held[tx.asset_id] < -CLOSED_QUANTITY:
            return tx
    return None


def trade_errors(tx: Transaction, other_transactions: list[Transaction], today: date) -> list[str]:
    """Messages for what is wrong in a trade typed by the user (empty list if it is valid).

    `other_transactions` are the saved trades, without the one being edited.
    """
    errors = []
    if tx.quantity <= 0:
        errors.append("The quantity must be greater than 0.")
    if tx.unit_price <= 0:
        errors.append("The unit price must be greater than 0.")
    if tx.fees < 0:
        errors.append("Fees cannot be negative.")
    if tx.date > today:
        errors.append("The date cannot be in the future.")
    if errors:
        return errors

    oversold = first_oversold_sale(other_transactions + [tx])
    if oversold is tx:
        errors.append("You cannot sell more shares than you hold on that date.")
    elif oversold is not None:
        errors.append(
            f"With this trade, the sale of {oversold.date:%d/%m/%Y} would sell more shares "
            "than held on that date."
        )
    return errors


def asset_errors(asset: Asset) -> list[str]:
    """Messages for what is wrong in an asset typed by the user (empty list if it is valid)."""
    errors = []
    if not asset.name.strip():
        errors.append("The name is required.")
    if len(asset.isin) != 12 or not asset.isin.isalnum():
        errors.append("An ISIN has 12 letters and digits, for example FR0000000000.")
    if not 0 <= asset.fees < 0.05:
        errors.append("Yearly fees must be between 0 % and 5 %.")
    if asset.target_weight is not None and not 0 < asset.target_weight <= 1:
        errors.append("The target weight must be between 0 % and 100 %, or left empty.")
    return errors


def target_weights_total(assets: list[Asset]) -> float | None:
    """Sum of the target weights that are set, or None if no asset has one."""
    weights = [asset.target_weight for asset in assets if asset.target_weight is not None]
    return sum(weights) if weights else None
