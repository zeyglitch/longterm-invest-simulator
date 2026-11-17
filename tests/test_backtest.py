from datetime import date

import pandas as pd
import pytest

from simulator.backtest import (
    annualised_return,
    backtest_dca,
    drawdowns,
    max_drawdown,
    monthly_prices,
)

CENT = 0.01


def monthly(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="MS"))


def test_each_month_buys_shares_at_the_price_of_the_month():
    # B1: prices 10, 20, 20, 40 and 100 a month
    table = backtest_dca(monthly([10.0, 20.0, 20.0, 40.0]), 100)

    assert table["shares"].iloc[-1] == pytest.approx(22.5)
    assert table["value"].iloc[-1] == pytest.approx(900.00, abs=CENT)
    assert table["invested"].iloc[-1] == pytest.approx(400.00, abs=CENT)
    assert table["invested"].tolist() == [100, 200, 300, 400]


def test_the_order_fee_is_taken_from_each_deposit():
    # B1, second half: 1 euro of fee per order
    table = backtest_dca(monthly([10.0, 20.0, 20.0, 40.0]), 100, order_fee=1)

    assert table["shares"].iloc[-1] == pytest.approx(22.275)
    assert table["invested"].iloc[-1] == 400  # the fee is part of what was paid


def test_an_initial_amount_is_one_more_order_at_the_first_price():
    table = backtest_dca(monthly([10.0, 20.0]), 100, initial_amount=1000, order_fee=1)

    # first month: 99 / 10 + 999 / 10 ; second month: 99 / 20
    assert table["shares"].tolist() == pytest.approx([109.8, 114.75])
    assert table["invested"].tolist() == [1100, 1200]


def test_a_lump_sum_alone_buys_once_and_pays_one_fee():
    table = backtest_dca(monthly([10.0, 20.0, 40.0]), 0, initial_amount=300, order_fee=1)

    assert table["shares"].tolist() == pytest.approx([29.9, 29.9, 29.9])
    assert table["value"].iloc[-1] == pytest.approx(1196.0)
    assert table["invested"].tolist() == [300, 300, 300]


def test_annualised_return_finds_the_rate_of_a_known_projection():
    # B2: 120 deposits of 100 that end at 16 247.34 earned 6 % a year
    assert annualised_return([100.0] * 120, 16247.34) == pytest.approx(0.06, abs=0.0001)


def test_annualised_return_of_a_single_deposit_and_of_a_loss():
    # 1 000 that become 1 210 in 24 months: 10 % a year
    assert annualised_return([1000.0] + [0.0] * 24, 1210.0) == pytest.approx(0.10, abs=0.0001)
    # 12 deposits that are worth less than their sum: a negative rate
    assert annualised_return([100.0] * 12, 1100.0) < 0
    # worth exactly their sum: 0 %
    assert annualised_return([100.0] * 12, 1200.0) == pytest.approx(0.0, abs=0.0001)


def test_annualised_return_needs_a_deposit():
    with pytest.raises(ValueError):
        annualised_return([0.0, 0.0], 100.0)


def test_max_drawdown_is_the_worst_fall_from_an_earlier_high():
    # B3: from 120 down to 90 is -25 %; the later fall from 130 to 104 is only -20 %
    prices = monthly([100.0, 120.0, 90.0, 95.0, 130.0, 104.0])

    fall, high_day, low_day = max_drawdown(prices)

    assert fall == pytest.approx(-0.25)
    assert high_day == date(2020, 2, 1)
    assert low_day == date(2020, 3, 1)


def test_prices_that_only_rise_never_fall():
    prices = monthly([10.0, 11.0, 12.0])

    assert max_drawdown(prices)[0] == 0
    assert drawdowns(prices).tolist() == [0, 0, 0]


def test_monthly_prices_keep_the_first_price_of_each_month():
    days = pd.to_datetime(["2020-01-03", "2020-01-15", "2020-02-04", "2020-02-28", "2020-04-01"])
    prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=days)

    firsts = monthly_prices(prices)

    assert firsts.tolist() == [10.0, 12.0, 14.0]  # no price in March: no row
    assert firsts.index[1] == pd.Timestamp("2020-02-04")
    assert monthly_prices(pd.Series(dtype=float)).empty
