import numpy as np
import pytest

from simulator.withdrawal import (
    depletion_month,
    simulate_withdrawals,
    success_rate,
    withdrawal_paths,
    withdrawal_schedule,
)

CENT = 0.01


def test_the_three_withdrawal_rules():
    # W1
    fixed = withdrawal_schedule("fixed", 500, 300_000, 0.02, 36)
    indexed = withdrawal_schedule("indexed", 500, 300_000, 0.02, 36)
    four_percent = withdrawal_schedule("rate", 0.04, 300_000, 0.02, 36)

    assert fixed.tolist() == [500] * 36
    assert indexed[:12].tolist() == [500] * 12
    assert indexed[12:24] == pytest.approx([510] * 12)
    assert indexed[24] == pytest.approx(520.2)
    assert four_percent[:12] == pytest.approx([1000] * 12)
    assert four_percent[12:24] == pytest.approx([1020] * 12)


def test_an_unknown_rule_is_refused():
    with pytest.raises(ValueError):
        withdrawal_schedule("weekly", 500, 300_000, 0.02, 12)


def test_capital_runs_out_when_withdrawals_beat_the_return():
    # W2: 100 000 at 3 % with 500 a month
    schedule = withdrawal_schedule("fixed", 500, 100_000, 0.0, 400)

    table = simulate_withdrawals(100_000, 100_000, schedule, 0.03, 0.0)

    assert depletion_month(table) == 276
    assert table["value"].iloc[275] > 0
    assert table["value"].iloc[276] == 0
    # the last withdrawal takes what is left, then nothing more leaves
    assert 0 < table["withdrawal"].iloc[276] < 500
    assert table["withdrawal"].iloc[277:].sum() == 0
    assert len(table) == 401


def test_each_withdrawal_is_taxed_on_its_share_of_gain():
    # W3: 100 000 for 60 000 deposited, no return, 1 000 a month, social charges of 18.6 %
    schedule = withdrawal_schedule("fixed", 1000, 100_000, 0.0, 120)

    table = simulate_withdrawals(100_000, 60_000, schedule, 0.0, 0.186)

    paid = table.iloc[1:101]
    assert paid["tax"].tolist() == pytest.approx([74.40] * 100)
    assert paid["net"].tolist() == pytest.approx([925.60] * 100)
    assert depletion_month(table) == 100
    assert table["net"].sum() == pytest.approx(92_560.00, abs=CENT)
    assert table["contributions"].iloc[100] == pytest.approx(0, abs=CENT)


def test_capital_that_lasts_has_no_depletion_month():
    schedule = withdrawal_schedule("fixed", 100, 100_000, 0.0, 120)

    table = simulate_withdrawals(100_000, 100_000, schedule, 0.03, 0.0)

    assert depletion_month(table) is None
    assert table["value"].iloc[-1] > 100_000  # 3 % a year is more than 1 200 a year


def test_the_tax_rate_can_change_from_one_month_to_the_next():
    # flat tax the first month, social charges only the second
    table = simulate_withdrawals(10_000, 5000, [1000.0, 1000.0], 0.0, [0.314, 0.186])

    assert table["tax"].tolist() == pytest.approx([0.0, 157.0, 93.0])


def test_no_tax_on_a_capital_in_loss():
    table = simulate_withdrawals(8000, 10_000, [1000.0] * 3, 0.0, 0.314)

    assert table["tax"].sum() == 0
    assert table["net"].sum() == 3000


def test_success_rate_without_volatility_is_all_or_nothing():
    # W4: the capital of W2 lasts 275 months
    lasts = withdrawal_schedule("fixed", 500, 100_000, 0.0, 240)
    runs_out = withdrawal_schedule("fixed", 500, 100_000, 0.0, 300)

    assert success_rate(100_000, lasts, 0.03, 0.0, n_paths=50, seed=1) == 1.0
    assert success_rate(100_000, runs_out, 0.03, 0.0, n_paths=50, seed=1) == 0.0


def test_random_paths_start_at_the_capital_and_never_go_below_zero():
    schedule = withdrawal_schedule("fixed", 500, 100_000, 0.0, 240)

    paths = withdrawal_paths(100_000, schedule, 0.05, 0.15, n_paths=500, seed=3)

    assert paths.shape == (500, 241)
    assert (paths[:, 0] == 100_000).all()
    assert (paths >= 0).all()
    rate = success_rate(100_000, schedule, 0.05, 0.15, n_paths=500, seed=3)
    assert rate == pytest.approx(np.mean(paths[:, -1] > 0))
    assert 0 < rate < 1


def test_without_volatility_the_paths_follow_the_fixed_return():
    schedule = withdrawal_schedule("indexed", 400, 100_000, 0.02, 120)

    paths = withdrawal_paths(100_000, schedule, 0.04, 0.0, n_paths=2, seed=1)
    table = simulate_withdrawals(100_000, 100_000, schedule, 0.04, 0.0)

    assert paths[0] == pytest.approx(table["value"].to_numpy())
