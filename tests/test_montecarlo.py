import numpy as np
import pandas as pd
import pytest

from simulator.montecarlo import (
    estimate_mu_sigma,
    net_final_values,
    percentiles,
    probability_above,
    random_unemployment,
    run_paths,
    simulate_bootstrap,
    simulate_normal,
)
from simulator.prices import monthly_returns
from simulator.projection import project


def monthly_flows(amounts: list[float]) -> pd.Series:
    return pd.Series(amounts, index=pd.date_range("2027-01-31", periods=len(amounts), freq="ME"))


FLOWS = monthly_flows([100.0] * 11 + [-2000.0] + [150.0] * 48)  # 60 months with a withdrawal


def test_without_volatility_every_path_is_the_deterministic_projection():
    # M1
    paths = simulate_normal(10_000, FLOWS, mu=0.06, sigma=0.0, n_paths=5, seed=42)

    expected = project(10_000, 10_000, FLOWS, 0.06)["value"].to_numpy()
    assert paths.shape == (5, 61)
    for path in paths:
        assert path == pytest.approx(expected)


def test_contributions_follow_the_projection_too():
    returns = np.full((3, 60), np.log(1.06) / 12)

    _, contributions = run_paths(10_000, FLOWS, returns, start_contributions=10_000)

    expected = project(10_000, 10_000, FLOWS, 0.06)["contributions"].iloc[-1]
    assert contributions == pytest.approx([expected] * 3)


def test_the_average_path_follows_the_mean_return():
    # M2: 10 000 without deposit, 7 % a year, 15 % of volatility, 10 years
    paths = simulate_normal(10_000, np.zeros(120), mu=0.07, sigma=0.15, n_paths=20_000, seed=42)

    final = paths[:, -1]
    assert final.mean() == pytest.approx(19671.51, rel=0.02)  # 10 000 x 1.07 ^ 10
    assert np.median(final) < final.mean()  # a few very high paths pull the mean up


def test_same_seed_same_paths_and_another_seed_other_paths():
    # M3
    first = simulate_normal(1000, np.zeros(24), 0.06, 0.15, n_paths=10, seed=7)
    again = simulate_normal(1000, np.zeros(24), 0.06, 0.15, n_paths=10, seed=7)
    other = simulate_normal(1000, np.zeros(24), 0.06, 0.15, n_paths=10, seed=8)

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_probability_and_percentiles():
    # M4
    assert probability_above([1, 2, 3, 4], 3) == 0.5

    paths = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])
    table = percentiles(paths)

    assert list(table.columns) == [10, 25, 50, 75, 90]
    assert table.loc[1].tolist() == pytest.approx(
        np.percentile([10, 20, 30, 40, 50], [10, 25, 50, 75, 90])
    )
    assert table.loc[0, 50] == 3.0


def test_mu_and_sigma_of_prices_that_rise_by_one_percent_a_month():
    # M5
    days = pd.date_range("2020-01-31", periods=37, freq="ME")
    prices = pd.Series(100 * 1.01 ** np.arange(37), index=days)

    mu, sigma = estimate_mu_sigma(prices)

    assert sigma == pytest.approx(0.0, abs=1e-9)
    assert mu == pytest.approx(0.126825, abs=1e-6)  # 1.01 ^ 12 - 1


def test_monthly_returns_use_the_last_price_of_each_month():
    days = pd.to_datetime(["2026-01-15", "2026-01-30", "2026-02-10", "2026-02-27", "2026-03-31"])
    prices = pd.Series([90.0, 100.0, 105.0, 110.0, 121.0], index=days)

    returns = monthly_returns(prices)

    assert returns.tolist() == pytest.approx([np.log(1.1), np.log(1.1)])


def test_too_short_a_history_is_refused():
    prices = pd.Series([100.0, 101.0], index=pd.to_datetime(["2026-01-31", "2026-02-28"]))

    with pytest.raises(ValueError):
        estimate_mu_sigma(prices)


def test_bootstrap_with_identical_past_returns_is_deterministic():
    # M6
    past = [np.log(1.01)] * 30

    paths = simulate_bootstrap(1000, np.zeros(12), past, n_paths=4, seed=3)

    for path in paths:
        assert path == pytest.approx(1000 * 1.01 ** np.arange(13))


def test_bootstrap_only_draws_returns_that_happened():
    past = [np.log(1.10), np.log(0.90)]

    paths = simulate_bootstrap(100, np.zeros(1), past, n_paths=200, seed=1)

    assert set(np.round(paths[:, 1], 6)) == {90.0, 110.0}


def test_unemployment_never_or_every_year():
    # M7
    flows = np.array([100.0] * 11 + [-500.0] + [100.0] * 12)

    never = random_unemployment(flows, n_paths=50, annual_prob=0.0, duration_months=6, seed=42)
    always = random_unemployment(flows, n_paths=50, annual_prob=1.0, duration_months=6, seed=42)

    assert never.shape == (50, 24)
    assert (never == flows).all()
    deposits = np.where(always > 0, always, 0).sum(axis=1)
    assert (deposits < 2300).all()  # every path lost some deposits
    assert (always[:, 11] == -500.0).all()  # the planned withdrawal is untouched
    # a period of 6 months drawn in the first year: at least 5 deposits lost (6 months, one of
    # which can be the month of the withdrawal, which holds no deposit)
    assert ((always[:, :17] == 0).sum(axis=1) >= 5).all()


def test_paths_have_the_right_shape_and_never_go_below_zero():
    # M8
    flows = np.array([-400.0] * 36)

    paths = simulate_normal(5000, flows, mu=0.05, sigma=0.25, n_paths=300, seed=42)

    assert paths.shape == (300, 37)
    assert (paths[:, 0] == 5000).all()
    assert (paths >= 0).all()
    assert (paths[:, -1] == 0).any()  # 14 400 withdrawn from 5 000: some paths are emptied


def test_each_path_can_have_its_own_flows():
    flows = np.array([[100.0, 100.0], [0.0, 0.0]])

    paths, contributions = run_paths(0, flows, np.zeros((2, 2)))

    assert paths[:, -1].tolist() == [200.0, 0.0]
    assert contributions.tolist() == [200.0, 0.0]


def test_final_values_are_taxed_on_their_gain_only():
    final = np.array([2000.0, 800.0])
    contributions = np.array([1000.0, 1000.0])

    net = net_final_values(final, contributions, 0.186)

    assert net.tolist() == pytest.approx([1814.0, 800.0])  # gain of 1 000 taxed; a loss is not


def test_no_prices_give_no_returns():
    # an empty answer of Yahoo must not crash the pages that estimate from history
    assert monthly_returns(pd.Series(dtype=float)).empty


def test_the_month_in_progress_is_left_out_of_the_returns():
    days = pd.to_datetime(["2026-08-31", "2026-09-30", "2026-10-30", "2026-11-16"])
    prices = pd.Series([100.0, 110.0, 121.0, 122.0], index=days)

    returns = monthly_returns(prices)

    # 30/10/2026 is a Friday, the last business day of October: October counts, November does not
    assert returns.tolist() == pytest.approx([np.log(1.1), np.log(1.1)])


def test_paths_and_projection_agree_on_a_losing_path_with_a_capped_withdrawal():
    flows = monthly_flows([0.0] * 5 + [-800.0] + [0.0] * 3 + [-50_000.0] + [200.0] * 2)
    returns = np.full((2, 12), np.log(0.70) / 12)  # -30 % a year

    paths, contributions = run_paths(10_000, flows, returns, start_contributions=10_000)

    expected = project(10_000, 10_000, flows, -0.30)
    assert paths[0] == pytest.approx(expected["value"].to_numpy())
    assert contributions[0] == pytest.approx(expected["contributions"].iloc[-1])


def test_a_month_can_have_a_deposit_and_a_withdrawal():
    parts = pd.DataFrame(
        {"deposit": [300.0, 300.0, 300.0], "withdrawal": [0.0, 500.0, 0.0]},
        index=pd.date_range("2027-01-31", periods=3, freq="ME"),
    )
    returns = np.full((1, 3), np.log(1.06) / 12)

    paths, contributions = run_paths(1000, parts["deposit"], returns, 1000, parts["withdrawal"])

    expected = project(1000, 1000, parts, 0.06)
    assert paths[0] == pytest.approx(expected["value"].to_numpy())
    assert contributions[0] == pytest.approx(expected["contributions"].iloc[-1])


def test_unemployment_on_deposits_leaves_the_withdrawal_of_the_same_month():
    deposits = np.full(12, 300.0)
    withdrawals = np.zeros(12)
    withdrawals[11] = 500.0  # a withdrawal in December, a month that also has a deposit

    lost = random_unemployment(deposits, n_paths=5, annual_prob=1.0, duration_months=12, seed=1)
    paths, _ = run_paths(1000, lost, np.zeros((5, 12)), 1000, withdrawals)

    # the period starts during the year and lasts beyond it: December never gets its deposit,
    # yet the 500 are withdrawn on every path
    assert (lost[:, -1] == 0).all()
    assert paths[:, -1].tolist() == pytest.approx((1000 + lost.sum(axis=1) - 500).tolist())
