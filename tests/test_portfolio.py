from datetime import date

import pandas as pd
import pytest

from simulator.models import Asset, CashMovement, Transaction
from simulator.portfolio import (
    asset_errors,
    compute_positions,
    first_oversold_sale,
    gross_deposits,
    portfolio_history,
    quantity_held,
    realised_gains,
    target_weights_total,
    total_deposits,
    trade_errors,
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
    positions = compute_positions([ASSET_A, ASSET_B], P1_TRADES, {})

    assert [position.asset for position in positions] == [ASSET_A]


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


def test_valid_asset_has_no_error():
    asset = Asset(None, "XX0000000001", "Asset A", "A.PA", fees=0.002, target_weight=0.75)

    assert asset_errors(asset) == []


@pytest.mark.parametrize(
    "changes",
    [
        {"name": "  "},
        {"isin": "XX00001"},
        {"isin": "XX00000000-1"},
        {"fees": 0.2},  # 0.2 means 20 %: the user typed a percentage as a decimal
        {"fees": -0.001},
        {"target_weight": 0},
        {"target_weight": 1.5},
    ],
)
def test_invalid_asset_gives_one_error(changes):
    values = dict(id=None, isin="XX0000000001", name="Asset A", ticker=None, fees=0.002)
    values.update(changes)

    assert len(asset_errors(Asset(**values))) == 1


def test_target_weights_total_ignores_assets_without_target():
    assets = [
        Asset(1, "XX0000000001", "A", None, target_weight=0.75),
        Asset(2, "XX0000000002", "B", None, target_weight=0.25),
        Asset(3, "XX0000000003", "C", None),
    ]

    assert target_weights_total(assets) == pytest.approx(1.0)
    assert target_weights_total(assets[2:]) is None


TODAY = date(2026, 2, 1)


def test_valid_buy_has_no_error():
    assert trade_errors(trade(5, 1, "BUY", 2.5, 40.0), [], TODAY) == []


@pytest.mark.parametrize(
    "new_trade",
    [
        trade(5, 1, "BUY", 0, 40.0),
        trade(5, 1, "BUY", 2, 0),
        trade(5, 1, "BUY", 2, 40.0, fees=-1.0),
        Transaction(None, date(2026, 2, 2), 1, "BUY", 2, 40.0),  # the day after TODAY
    ],
)
def test_invalid_trade_fields_give_one_error(new_trade):
    assert len(trade_errors(new_trade, [], TODAY)) == 1


def test_sale_above_quantity_held_on_that_date_is_refused():
    # P1: 10 shares on the 5th, 20 more on the 12th
    assert trade_errors(trade(8, 1, "SELL", 10, 7.0), P1_TRADES, TODAY) == []
    assert len(trade_errors(trade(8, 1, "SELL", 11, 7.0), P1_TRADES, TODAY)) == 1
    assert len(trade_errors(trade(20, 1, "SELL", 31, 7.0), P1_TRADES, TODAY)) == 1
    assert len(trade_errors(trade(20, 2, "SELL", 1, 7.0), P1_TRADES, TODAY)) == 1


def test_selling_the_rest_of_fractional_shares_is_accepted():
    # held is 0.3 - 0.1 = 0.19999999999999998, slightly under the 0.2 sold
    saved = [trade(5, 1, "BUY", 0.3, 10.0, tx_id=1), trade(6, 1, "SELL", 0.1, 10.0, tx_id=2)]

    assert trade_errors(trade(7, 1, "SELL", 0.2, 10.0), saved, TODAY) == []


def test_new_sale_on_the_day_of_a_buy_comes_after_it():
    saved = [trade(5, 1, "BUY", 10, 5.0, tx_id=1)]

    assert trade_errors(trade(5, 1, "SELL", 10, 6.0), saved, TODAY) == []


def test_earlier_sale_that_empties_a_later_sale_is_refused():
    # P3 sells 10 on the 19th out of 30 held: selling 25 on the 15th leaves only 5 for it
    errors = trade_errors(trade(15, 1, "SELL", 25, 7.0), P3_TRADES, TODAY)

    assert len(errors) == 1
    assert "19/01/2026" in errors[0]


def test_removing_a_buy_needed_by_a_sale_is_detected():
    # P3 without its second buy: 10 held, and the sale of 10 on the 19th is still fine
    assert first_oversold_sale([P3_TRADES[0], P3_TRADES[2]]) is None
    # without its first buy, 20 held: fine too ; without both, the sale has nothing to sell
    assert first_oversold_sale([P3_TRADES[2]]) is P3_TRADES[2]


# H1, H2: 10 shares at 5.00 on Monday 05/01, 10 more at 6.00 on Monday 12/01
H_TRADES = [trade(5, 1, "BUY", 10, 5.0), trade(12, 1, "BUY", 10, 6.0)]


def closes(values: dict[str, float]) -> pd.Series:
    return pd.Series(list(values.values()), index=pd.to_datetime(list(values)), dtype=float)


def history_on(history: pd.DataFrame, day: str) -> pd.Series:
    return history[history["date"] == pd.Timestamp(day)].iloc[0]


def test_history_values_the_portfolio_with_the_price_of_the_day():
    prices = {1: closes({"2026-01-05": 5.0, "2026-01-12": 6.0})}

    history = portfolio_history(H_TRADES, prices, end=date(2026, 1, 12))

    last = history_on(history, "2026-01-12")
    assert last["value"] == pytest.approx(120.0)  # 20 shares at 6.00
    assert last["invested"] == pytest.approx(110.0)  # 50 + 60
    assert history["date"].iloc[0] == pd.Timestamp("2026-01-05")  # first trade
    assert history["date"].iloc[-1] == pd.Timestamp("2026-01-12")  # end


def test_history_uses_the_last_earlier_price_when_a_day_has_none():
    # no price on Sunday 11/01 and the 6.00 of the 12th must not leak backwards
    prices = {1: closes({"2026-01-05": 5.0, "2026-01-12": 6.0})}

    history = portfolio_history(H_TRADES, prices, end=date(2026, 1, 18))

    sunday = history_on(history, "2026-01-11")
    assert sunday["value"] == pytest.approx(50.0)  # 10 shares, still at the 5.00 of the 5th
    assert sunday["invested"] == pytest.approx(50.0)
    assert history_on(history, "2026-01-18")["value"] == pytest.approx(120.0)  # price of the 12th


def test_history_falls_back_on_the_last_trade_price_before_any_price_exists():
    history = portfolio_history(H_TRADES, {}, end=date(2026, 1, 12))

    assert history_on(history, "2026-01-12")["value"] == pytest.approx(120.0)  # 20 at 6.00


def test_history_follows_sales_and_fees():
    trades = [trade(5, 1, "BUY", 10, 5.0, 1.0), trade(12, 1, "SELL", 4, 7.0, 1.0)]
    prices = {1: closes({"2026-01-05": 5.0, "2026-01-12": 7.0})}

    history = portfolio_history(trades, prices, end=date(2026, 1, 12))

    assert history_on(history, "2026-01-05")["invested"] == pytest.approx(51.0)
    after_sale = history_on(history, "2026-01-12")
    assert after_sale["value"] == pytest.approx(42.0)  # 6 shares at 7.00
    assert after_sale["invested"] == pytest.approx(51.0 * 0.6)  # 4 of 10 shares left at avg cost


def test_history_adds_the_values_of_several_assets():
    trades = [trade(5, 1, "BUY", 10, 5.0), trade(5, 2, "BUY", 2.5, 40.0)]
    prices = {1: closes({"2026-01-05": 6.0}), 2: closes({"2026-01-05": 42.0})}

    history = portfolio_history(trades, prices, end=date(2026, 1, 5))

    assert len(history) == 1
    assert history["value"].iloc[0] == pytest.approx(165.0)
    assert history["invested"].iloc[0] == pytest.approx(150.0)


def test_history_without_trade_is_empty():
    history = portfolio_history([], {})

    assert history.empty
    assert list(history.columns) == ["date", "value", "invested"]
