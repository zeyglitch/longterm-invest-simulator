from datetime import date

import pandas as pd
import pytest

from simulator.comparison import (
    Strategy,
    compare_envelopes,
    compare_strategies,
    dca_flows,
    lump_sum_flows,
    simulate_strategy,
)
from simulator.models import TaxSettings
from simulator.projection import project

CENT = 0.01
# a PEA opened long ago: social charges only
TAX = TaxSettings(ps_rate=0.186, ir_rate=0.128, pea_ceiling=150_000, pea_open_date=date(2010, 1, 1))
LIVRET_CEILING = 22_950


def months(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2027-01-31", periods=n, freq="ME")


def monthly_flows(amounts: list[float]) -> pd.Series:
    return pd.Series(amounts, index=months(len(amounts)))


def last(table: pd.DataFrame, name: str) -> pd.Series:
    return table[table["name"] == name].iloc[-1]


def test_same_plan_in_the_three_envelopes():
    # E1: 100 a month for 10 years at 6 %, Livret A at 1.7 %
    table = compare_envelopes(0, 0, monthly_flows([100.0] * 120), 0.06, TAX, 0.017, LIVRET_CEILING)

    assert last(table, "PEA")["value"] == pytest.approx(16247.34, abs=CENT)
    assert last(table, "CTO")["value"] == pytest.approx(16247.34, abs=CENT)
    assert last(table, "PEA")["net_value"] == pytest.approx(15457.33, abs=CENT)
    assert last(table, "CTO")["net_value"] == pytest.approx(14913.68, abs=CENT)
    assert last(table, "LIVRET")["value"] == pytest.approx(13061.56, abs=CENT)
    assert last(table, "LIVRET")["tax"] == 0
    assert len(table) == 3 * 121


def test_lump_sum_against_spread_deposits():
    # E2: 12 000 at once against 1 000 a month for 12 months, 6 %, after 12 months
    index = months(12)
    lump = Strategy("Lump sum", "CTO", 0.06, 0.15, 0.0, *reversed(lump_sum_flows(12_000, index)))
    spread = Strategy("Spread", "CTO", 0.06, 0.15, 0.0, *reversed(dca_flows(12_000, 12, index)))

    table = compare_strategies([lump, spread], 0, 0, TAX, LIVRET_CEILING)

    assert last(table, "Lump sum")["value"] == pytest.approx(12720.00, abs=CENT)
    assert last(table, "Spread")["value"] == pytest.approx(12326.53, abs=CENT)
    assert last(table, "Lump sum")["contributions"] == pytest.approx(12_000)
    assert last(table, "Spread")["contributions"] == pytest.approx(12_000)


def test_spread_deposits_stop_after_their_last_month():
    initial, flows = dca_flows(6000, 3, months(5))

    assert initial == 0
    assert flows.tolist() == [2000, 2000, 2000, 0, 0]
    assert lump_sum_flows(6000, months(5))[0] == 6000
    assert lump_sum_flows(6000, months(5))[1].sum() == 0


def test_a_livret_cannot_start_above_its_ceiling():
    table = compare_envelopes(30_000, 30_000, monthly_flows([0.0]), 0.0, TAX, 0.0, LIVRET_CEILING)

    livret = table[table["name"] == "LIVRET"]
    assert livret["value"].iloc[-1] == LIVRET_CEILING
    assert livret["not_invested"].sum() == pytest.approx(30_000 - LIVRET_CEILING)


def test_above_the_pea_ceiling_the_pea_is_taxed_like_a_cto_on_the_surplus():
    # ceiling already reached: the 1 200 deposited go to the CTO pocket of the PEA
    settings = TaxSettings(0.186, 0.128, pea_ceiling=1000, pea_open_date=date(2010, 1, 1))
    flows = monthly_flows([1200.0] + [0.0] * 12)

    table = compare_envelopes(1000, 1000, flows, 1.0, settings, 0.0, LIVRET_CEILING)

    # the 1 000 of the PEA double in 12 months and grow one month more; the 1 200 double
    pea_gain = 1000 * 2 ** (13 / 12) - 1000
    assert last(table, "PEA")["tax"] == pytest.approx(pea_gain * 0.186 + 1200 * 0.314)
    assert last(table, "CTO")["tax"] == pytest.approx((pea_gain + 1200) * 0.314)


def test_fees_lower_a_strategy_but_not_a_livret():
    flows = monthly_flows([100.0] * 120)
    strategies = [
        Strategy("No fees", "CTO", 0.06, 0.15, 0.0, flows),
        Strategy("Fees", "CTO", 0.06, 0.15, 0.002, flows),
        Strategy("Livret", "LIVRET", 0.017, 0.0, 0.002, flows),
    ]

    table = compare_strategies(strategies, 0, 0, TAX, LIVRET_CEILING)

    assert last(table, "No fees")["value"] == pytest.approx(16247.34, abs=CENT)
    assert last(table, "Fees")["value"] == pytest.approx(16071.58, abs=CENT)  # R3
    assert last(table, "Livret")["value"] == pytest.approx(13061.56, abs=CENT)


def test_a_strategy_without_volatility_is_its_projection():
    flows = monthly_flows([100.0] * 11 + [-500.0] + [100.0] * 12)
    strategy = Strategy("Calm", "CTO", 0.06, 0.0, 0.0, flows, initial_deposit=1000)

    paths = simulate_strategy(strategy, 5000, 4000, n_paths=3, seed=1)

    expected = project(6000, 5000, flows, 0.06).iloc[-1]
    assert paths["value"].tolist() == pytest.approx([expected["value"]] * 3)
    assert paths["contributions"].tolist() == pytest.approx([expected["contributions"]] * 3)
    assert paths["worst_fall"].tolist() == pytest.approx([0.0] * 3)


def test_more_volatility_gives_a_wider_spread_and_deeper_falls():
    flows = monthly_flows([100.0] * 120)
    calm = simulate_strategy(Strategy("Calm", "CTO", 0.05, 0.08, 0.0, flows), 0, 0, 2000, 7)
    rough = simulate_strategy(Strategy("Rough", "CTO", 0.05, 0.20, 0.0, flows), 0, 0, 2000, 7)

    assert rough["value"].std() > calm["value"].std()
    assert rough["worst_fall"].median() > calm["worst_fall"].median()


def test_deposits_and_withdrawals_can_come_as_two_columns():
    parts = pd.DataFrame({"deposit": [300.0, 300.0], "withdrawal": [0.0, 100.0]}, index=months(2))
    strategy = Strategy("Parts", "CTO", 0.0, 0.0, 0.0, parts)

    paths = simulate_strategy(strategy, 0, 0, n_paths=2, seed=1)

    assert paths["value"].tolist() == pytest.approx([500.0, 500.0])
