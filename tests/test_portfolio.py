from datetime import date

import pytest

from simulator.models import Asset, CashMovement, Transaction
from simulator.portfolio import (
    compute_positions,
    gross_deposits,
    quantity_held,
    realised_gains,
    total_deposits,
)

ASSET_A = Asset(id=1, isin="XX0000000001", name="Asset A", ticker=None)
ASSET_B = Asset(id=2, isin="XX0000000002", name="Asset B", ticker=None)


def trade(day, asset_id, kind, quantity, price, fees=0.0, tx_id=None) -> Transaction:
    return Transaction(
        id=tx_id,
        date=date(2026, 1, day),
        asset_id=asset_id,
        kind=kind,
        quantity=quantity,
        unit_price=price,
        fees=fees,
    )


# P1: 10 shares at 5.00 (fee 1) then 20 shares at 6.00 (fee 1)
P1_TRADES = [trade(5, 1, "BUY", 10, 5.0, 1.0), trade(12, 1, "BUY", 20, 6.0, 1.0)]


def test_average_cost_includes_fees():
    # reference P1: cost 50 + 1 + 120 + 1 = 172, avg 172 / 30
    position = compute_positions([ASSET_A], P1_TRADES, {})[0]

    assert position.quantity == 30
    assert position.invested == pytest.approx(172.00, abs=0.01)
    assert position.avg_cost == pytest.approx(5.7333, abs=0.0001)


def test_average_cost_without_fees():
    # reference P1 without fees: cost 170, avg 170 / 30
    position = compute_positions([ASSET_A], P1_TRADES, {}, include_fees=False)[0]

    assert position.invested == pytest.approx(170.00, abs=0.01)
    assert position.avg_cost == pytest.approx(5.6667, abs=0.0001)


def test_fractional_buy():
    # reference P2: 2.5 shares at 40.00, no fee
    position = compute_positions([ASSET_B], [trade(5, 2, "BUY", 2.5, 40.0)], {})[0]

    assert position.invested == pytest.approx(100.00, abs=0.01)
    assert position.avg_cost == pytest.approx(40.00, abs=0.0001)


def test_asset_without_trades_has_no_position():
    assert compute_positions([ASSET_A, ASSET_B], P1_TRADES, {})[0].asset == ASSET_A
    assert len(compute_positions([ASSET_A, ASSET_B], P1_TRADES, {})) == 1


# P3: after P1, sell 10 shares at 7.00 (fee 1)
P3_TRADES = P1_TRADES + [trade(19, 1, "SELL", 10, 7.0, 1.0)]


def test_sell_gives_realised_gain_and_keeps_average_cost():
    # reference P3: gain 70 - 1 - 10 * 172/30 = 11.6667 ; cost 172 - 57.3333 = 114.6667
    position = compute_positions([ASSET_A], P3_TRADES, {})[0]
    gains = realised_gains([ASSET_A], P3_TRADES)

    assert gains["gain"].tolist() == [pytest.approx(11.6667, abs=0.0001)]
    assert position.quantity == 20
    assert position.invested == pytest.approx(114.6667, abs=0.0001)
    assert position.avg_cost == pytest.approx(5.7333, abs=0.0001)


def test_closed_position_restarts_average_cost_on_next_buy():
    # reference P4: sell the 20 remaining at 7.00 without fee, then buy 5 at 8.00
    sold_out = P3_TRADES + [trade(20, 1, "SELL", 20, 7.0)]
    rebought = sold_out + [trade(26, 1, "BUY", 5, 8.0)]

    assert compute_positions([ASSET_A], sold_out, {}) == []
    assert realised_gains([ASSET_A], sold_out)["gain"].iloc[-1] == pytest.approx(25.3333, abs=1e-4)
    position = compute_positions([ASSET_A], rebought, {})[0]
    assert position.quantity == 5
    assert position.avg_cost == pytest.approx(8.00, abs=0.0001)


def test_fractional_shares_sold_out_leave_no_residue():
    # 0.1 + 0.2 - 0.3 is not exactly 0 in floating point
    trades = [
        trade(5, 1, "BUY", 0.1, 10.0),
        trade(6, 1, "BUY", 0.2, 10.0),
        trade(7, 1, "SELL", 0.3, 12.0),
    ]

    assert compute_positions([ASSET_A], trades, {}) == []


def test_trades_given_out_of_order_are_processed_by_date():
    # same trades as P3, listed with the sale first: the result must not change
    shuffled = [P3_TRADES[2], P3_TRADES[1], P3_TRADES[0]]

    position = compute_positions([ASSET_A], shuffled, {})[0]

    assert position.avg_cost == pytest.approx(5.7333, abs=0.0001)
    assert realised_gains([ASSET_A], shuffled)["gain"].iloc[0] == pytest.approx(11.6667, abs=1e-4)


def test_no_sale_gives_empty_realised_gains():
    assert realised_gains([ASSET_A], P1_TRADES).empty


def test_quantity_held_before_between_and_after_trades():
    # P3 trades: +10 on the 5th, +20 on the 12th, -10 on the 19th
    assert quantity_held(P3_TRADES, 1, date(2026, 1, 4)) == 0
    assert quantity_held(P3_TRADES, 1, date(2026, 1, 5)) == 10
    assert quantity_held(P3_TRADES, 1, date(2026, 1, 15)) == 30
    assert quantity_held(P3_TRADES, 1, date(2026, 2, 1)) == 20
    assert quantity_held(P3_TRADES, 2, date(2026, 2, 1)) == 0


def test_value_gain_and_weights_from_last_prices():
    # reference P5: P1 priced at 6.50, P2 priced at 42.00
    trades = P1_TRADES + [trade(5, 2, "BUY", 2.5, 40.0)]
    prices = {1: (date(2026, 1, 30), 6.50), 2: (date(2026, 1, 30), 42.00)}

    position_a, position_b = compute_positions([ASSET_A, ASSET_B], trades, prices)

    # A: 30 * 6.50 = 195, gain 195 - 172 = 23, 23 / 172 = 13.37 %
    assert position_a.value == pytest.approx(195.00, abs=0.01)
    assert position_a.gain == pytest.approx(23.00, abs=0.01)
    assert position_a.gain_pct == pytest.approx(0.1337, abs=0.0001)
    assert position_a.last_price == 6.50
    assert position_a.price_date == date(2026, 1, 30)
    # B: 2.5 * 42 = 105 ; weights 195 / 300 and 105 / 300
    assert position_b.value == pytest.approx(105.00, abs=0.01)
    assert position_a.weight == pytest.approx(0.65)
    assert position_b.weight == pytest.approx(0.35)


def test_position_without_price_has_no_value():
    # reference P6: no price for B, no exception
    trades = P1_TRADES + [trade(5, 2, "BUY", 2.5, 40.0)]
    prices = {1: (date(2026, 1, 30), 6.50)}

    position_a, position_b = compute_positions([ASSET_A, ASSET_B], trades, prices)

    assert position_b.value is None
    assert position_b.gain is None
    assert position_b.gain_pct is None
    assert position_b.weight is None
    assert position_b.invested == pytest.approx(100.00, abs=0.01)
    # A is the only priced position
    assert position_a.weight == pytest.approx(1.0)


CASH = [
    CashMovement(id=None, date=date(2026, 1, 2), kind="DEPOSIT", amount=500.0),
    CashMovement(id=None, date=date(2026, 2, 2), kind="DEPOSIT", amount=300.0),
    CashMovement(id=None, date=date(2026, 3, 2), kind="WITHDRAWAL", amount=200.0),
]


def test_total_deposits_subtracts_withdrawals():
    # 500 + 300 - 200
    assert total_deposits(CASH) == pytest.approx(600.00, abs=0.01)


def test_gross_deposits_ignores_withdrawals():
    # 500 + 300: a withdrawal does not give ceiling capacity back
    assert gross_deposits(CASH) == pytest.approx(800.00, abs=0.01)


def test_deposit_totals_are_zero_without_movements():
    assert total_deposits([]) == 0
    assert gross_deposits([]) == 0
