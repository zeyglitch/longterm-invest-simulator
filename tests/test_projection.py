from datetime import date

import pandas as pd
import pytest

from simulator.models import TaxSettings
from simulator.projection import (
    add_net_value,
    date_target_reached,
    fees_cost,
    monthly_rate,
    net_return,
    project,
    project_livret,
    project_with_ceiling,
    to_real,
    yearly_table,
)

CENT = 0.01


def monthly_flows(amounts: list[float]) -> pd.Series:
    """One flow per month, from January 2027."""
    return pd.Series(amounts, index=pd.date_range("2027-01-31", periods=len(amounts), freq="ME"))


def final(projection: pd.DataFrame, column: str = "value") -> float:
    return projection[column].iloc[-1]


def test_monthly_rate_compounds_to_the_yearly_rate():
    assert (1 + monthly_rate(0.06)) ** 12 == pytest.approx(1.06)
    assert monthly_rate(0.06) < 0.06 / 12  # dividing by 12 would overstate it


def test_capital_alone_grows_with_compound_interest():
    # R1: 1 000 at 5 % for 120 months
    projection = project(1000, 1000, monthly_flows([0.0] * 120), 0.05)

    assert final(projection) == pytest.approx(1628.89, abs=CENT)
    assert len(projection) == 121  # the start, then one row per month
    assert projection["value"].iloc[0] == 1000


def test_constant_deposit_matches_the_closed_formula():
    # R2: 100 a month at 6 % for 120 months
    projection = project(0, 0, monthly_flows([100.0] * 120), 0.06)

    m = monthly_rate(0.06)
    closed_formula = 100 * ((1 + m) ** 120 - 1) / m
    assert final(projection) == pytest.approx(16247.34, abs=CENT)
    assert final(projection) == pytest.approx(closed_formula)
    assert final(projection, "contributions") == pytest.approx(12000)
    assert final(projection, "gains") == pytest.approx(4247.34, abs=CENT)


def test_target_is_reached_on_the_first_month_above_it():
    # R2: 10 000 is reached at month 82
    projection = project(0, 0, monthly_flows([100.0] * 120), 0.06)

    assert date_target_reached(projection, 10_000) == projection["date"].iloc[82].date()
    assert projection["value"].iloc[81] < 10_000 <= projection["value"].iloc[82]
    assert date_target_reached(projection, 1_000_000) is None


def test_fees_lower_the_return_and_the_final_value():
    # R3: 6 % gross with 0.2 % of fees
    assert net_return(0.06, 0.002) == pytest.approx(0.05788)

    projection = project(0, 0, monthly_flows([100.0] * 120), net_return(0.06, 0.002))

    assert final(projection) == pytest.approx(16071.58, abs=CENT)


def test_fees_cost_is_the_gap_between_the_two_final_values():
    # R7: 16 247.34 - 16 071.58
    cost = fees_cost(0, 0, monthly_flows([100.0] * 120), 0.06, 0.002)

    assert cost == pytest.approx(175.76, abs=CENT)


def test_withdrawal_takes_capital_and_gain_in_proportion():
    # R5: 10 000 at 6 %, 2 000 withdrawn at month 12, 24 months in all
    flows = monthly_flows([0.0] * 11 + [-2000.0] + [0.0] * 12)

    projection = project(10_000, 10_000, flows, 0.06)

    assert projection["value"].iloc[12] + 2000 == pytest.approx(10600.00, abs=CENT)  # before
    # gain part of the withdrawal: 2 000 x 600 / 10 600 = 113.21
    assert projection["contributions"].iloc[12] == pytest.approx(8113.21, abs=CENT)
    assert final(projection) == pytest.approx(9116.00, abs=CENT)


def test_withdrawal_is_capped_at_the_value():
    # R8: asking for more than there is
    projection = project(1000, 1000, monthly_flows([-5000.0, 0.0, 100.0]), 0.06)

    assert projection["value"].iloc[1] == 0
    assert projection["contributions"].iloc[1] == pytest.approx(0)
    assert projection["flow"].iloc[1] == pytest.approx(-1000 * (1 + monthly_rate(0.06)))
    assert (projection["value"] >= 0).all()
    assert final(projection) == pytest.approx(100)  # deposits work again afterwards


def test_projection_without_flow_is_just_the_start():
    projection = project(1000, 800, monthly_flows([]), 0.06)

    assert len(projection) == 1
    assert projection["gains"].iloc[0] == 200
    assert date_target_reached(projection, 500) is None  # no date to give


def test_real_value_removes_inflation():
    # R6: 100 000 after 360 months at 2 % of inflation
    values = pd.Series([100_000.0] * 361)

    real = to_real(values, 0.02)

    assert real.iloc[-1] == pytest.approx(55207.09, abs=CENT)
    assert real.iloc[0] == 100_000  # today's euros are already today's


def test_livret_grows_at_its_rate():
    # L1: 5 000 + 100 a month at 1.7 % for 120 months, far below the ceiling
    projection = project_livret(5000, monthly_flows([100.0] * 120), 0.017, 22_950)

    assert final(projection) == pytest.approx(18979.62, abs=CENT)
    assert final(projection, "contributions") == pytest.approx(17000)
    assert projection["not_invested"].sum() == 0


def test_livret_refuses_deposits_above_its_ceiling():
    # L2: 22 000 + 500 a month, ceiling 22 950
    projection = project_livret(22_000, monthly_flows([500.0] * 3), 0.017, 22_950)

    assert projection["flow"].tolist()[1:] == pytest.approx([500.00, 387.40, 0.0], abs=CENT)
    assert projection["value"].tolist()[1:] == pytest.approx(
        [22530.93, 22950.00, 22982.26], abs=CENT
    )
    # what was refused: nothing, then 112.60, then the whole 500
    assert projection["not_invested"].tolist()[1:] == pytest.approx([0.0, 112.60, 500.0], abs=CENT)


def test_deposits_above_the_pea_ceiling_go_to_the_cto():
    # C1: ceiling 20 000, 19 000 already deposited, 500 a month for 4 months, no return
    projection = project_with_ceiling(
        19_000, 19_000, monthly_flows([500.0] * 4), 0.0, ceiling=20_000, already_deposited=19_000
    )

    assert projection["contributions_pea"].tolist() == [19_000, 19_500, 20_000, 20_000, 20_000]
    assert projection["contributions_cto"].tolist() == [0, 0, 0, 500, 1000]
    assert final(projection, "value_pea") == 20_000
    assert final(projection, "value_cto") == 1000
    assert final(projection) == 21_000


def test_a_deposit_can_be_split_between_pea_and_cto():
    projection = project_with_ceiling(
        0, 0, monthly_flows([500.0]), 0.0, ceiling=20_000, already_deposited=19_800
    )

    assert final(projection, "contributions_pea") == pytest.approx(200)
    assert final(projection, "contributions_cto") == pytest.approx(300)


def test_withdrawals_empty_the_cto_before_the_pea_and_give_no_capacity_back():
    flows = monthly_flows([500.0, -700.0, 400.0])

    projection = project_with_ceiling(1000, 1000, flows, 0.0, ceiling=1200, already_deposited=1000)

    # month 1: 200 to the PEA (ceiling reached), 300 to the CTO
    assert projection["value_pea"].iloc[1] == 1200
    assert projection["value_cto"].iloc[1] == 300
    # month 2: 300 from the CTO, then 400 from the PEA
    assert projection["value_cto"].iloc[2] == 0
    assert projection["value_pea"].iloc[2] == 800
    # month 3: the PEA stays closed to deposits, the withdrawal gave no capacity back
    assert projection["value_pea"].iloc[3] == 800
    assert projection["value_cto"].iloc[3] == 400


def test_without_reaching_the_ceiling_the_two_projections_agree():
    flows = monthly_flows([100.0] * 60)

    plain = project(5000, 4000, flows, 0.06)
    capped = project_with_ceiling(5000, 4000, flows, 0.06, 150_000, 4000)

    assert final(capped) == pytest.approx(final(plain))
    assert final(capped, "value_cto") == 0


def test_the_start_row_is_dated_one_month_before_the_first_flow():
    projection = project(0, 0, monthly_flows([100.0]), 0.06)

    assert projection["date"].iloc[0].date() == date(2026, 12, 31)


TAX = TaxSettings(ps_rate=0.186, ir_rate=0.128, pea_ceiling=150_000, pea_open_date=None)


def test_net_value_of_a_pea_changes_rate_on_its_fifth_birthday():
    # opened on 15/02/2022: the months up to January 2027 bear 31.4 %, the next ones 18.6 %
    projection = project(1000, 600, monthly_flows([0.0, 0.0]), 0.0)  # gain of 400 throughout

    table = add_net_value(projection, "PEA", date(2022, 2, 15), TAX)

    assert table["tax"].tolist() == pytest.approx([125.6, 125.6, 74.4])  # Dec, Jan, Feb
    assert table["net_value"].iloc[-1] == pytest.approx(925.6)


def test_pea_without_opening_date_and_cto_bear_the_flat_tax():
    projection = project(1000, 600, monthly_flows([0.0]), 0.0)

    unknown = add_net_value(projection, "PEA", None, TAX)
    cto = add_net_value(projection, "CTO", date(2010, 1, 1), TAX)

    assert unknown["tax"].iloc[-1] == pytest.approx(125.6)
    assert cto["tax"].iloc[-1] == pytest.approx(125.6)


def test_the_surplus_above_the_ceiling_is_taxed_like_a_cto():
    # 1 000 on an old PEA (gain 400), then 500 above the ceiling that double in a CTO
    projection = project_with_ceiling(1000, 600, monthly_flows([500.0, 0.0]), 0.0, 600, 600)
    projection.loc[2, "value_cto"] = 1000.0
    projection.loc[2, "value"] = 2000.0

    table = add_net_value(projection, "PEA", date(2010, 1, 1), TAX)

    # PEA: 400 x 18.6 % = 74.40 ; CTO: 500 x 31.4 % = 157.00
    assert table["tax"].iloc[-1] == pytest.approx(231.4)
    assert table["net_value"].iloc[-1] == pytest.approx(1768.6)


def test_no_tax_on_a_loss():
    projection = project(800, 1000, monthly_flows([0.0]), 0.0)

    table = add_net_value(projection, "CTO", None, TAX)

    assert table["tax"].tolist() == [0.0, 0.0]
    assert table["net_value"].tolist() == [800.0, 800.0]


def test_yearly_table_keeps_the_last_month_of_each_year():
    projection = project(0, 0, monthly_flows([100.0] * 30), 0.0)  # January 2027 to June 2029

    table = yearly_table(projection)

    assert table["year"].tolist() == [2026, 2027, 2028, 2029]
    assert table["contributions"].tolist() == [0, 1200, 2400, 3000]
    assert table["date"].iloc[-1] == pd.Timestamp("2029-06-30")


def flow_parts(deposits: list[float], withdrawals: list[float]) -> pd.DataFrame:
    days = pd.date_range("2027-01-31", periods=len(deposits), freq="ME")
    return pd.DataFrame({"deposit": deposits, "withdrawal": withdrawals}, index=days)


def test_the_ceiling_counts_deposits_before_the_withdrawal_of_the_same_month():
    # 900 deposited in all under a ceiling of 800: the withdrawal of 100 gives no room back
    parts = flow_parts([300.0, 300.0, 300.0], [0.0, 100.0, 0.0])

    projection = project_with_ceiling(0, 0, parts, 0.0, ceiling=800, already_deposited=0)

    assert final(projection, "contributions_pea") + 100 == pytest.approx(800)  # 800 in, 100 out
    assert final(projection, "value_cto") == pytest.approx(100)
    assert projection["flow"].tolist() == [0.0, 300.0, 200.0, 300.0]


def test_a_net_flow_and_its_two_parts_give_the_same_value():
    parts = flow_parts([300.0, 300.0, 300.0], [0.0, 100.0, 0.0])
    net = parts["deposit"] - parts["withdrawal"]

    assert final(project(1000, 1000, parts, 0.06)) == pytest.approx(
        final(project(1000, 1000, net, 0.06))
    )


def test_a_withdrawal_takes_gain_in_proportion_in_each_pocket():
    # PEA: 1 000 for 600 deposited (gain 40 %); the surplus of 400 goes to a CTO without gain
    parts = flow_parts([400.0, 0.0], [0.0, 700.0])

    projection = project_with_ceiling(1000, 600, parts, 0.0, ceiling=600, already_deposited=600)

    # the CTO is emptied first (400, all capital), then 300 leave the PEA: 180 of capital
    assert final(projection, "value_cto") == 0
    assert final(projection, "contributions_cto") == pytest.approx(0)
    assert final(projection, "value_pea") == pytest.approx(700)
    assert final(projection, "contributions_pea") == pytest.approx(420)


def test_livret_accepts_a_deposit_up_to_the_ceiling_then_lets_the_withdrawal_out():
    parts = flow_parts([500.0], [200.0])

    projection = project_livret(1000, parts, 0.0, ceiling=1300)

    assert final(projection) == pytest.approx(1100)  # 300 accepted, 200 refused, 200 out
    assert final(projection, "not_invested") == pytest.approx(200)


def test_the_cto_can_have_its_own_return():
    # ceiling already reached: 1 000 stay in the PEA at 0 %, 1 200 go to a CTO at 12 %
    flows = monthly_flows([1200.0] + [0.0] * 12)

    projection = project_with_ceiling(1000, 1000, flows, 0.0, 1000, 1000, cto_return=0.12)

    assert final(projection, "value_pea") == pytest.approx(1000)
    assert final(projection, "value_cto") == pytest.approx(1344)
    assert final(projection) == pytest.approx(2344)


def test_tax_and_net_value_are_given_for_each_pocket():
    # old PEA: 1 000 for 600 deposited ; CTO: 500 deposited that become 1 000
    projection = project_with_ceiling(1000, 600, monthly_flows([500.0, 0.0]), 0.0, 600, 600)
    projection.loc[2, "value_cto"] = 1000.0
    projection.loc[2, "value"] = 2000.0

    last = add_net_value(projection, "PEA", date(2010, 1, 1), TAX).iloc[-1]

    assert last["tax_pea"] == pytest.approx(74.4)  # 400 x 18.6 %
    assert last["tax_cto"] == pytest.approx(157.0)  # 500 x 31.4 %
    assert last["net_pea"] == pytest.approx(925.6)
    assert last["net_cto"] == pytest.approx(843.0)
    assert last["net_pea"] + last["net_cto"] == pytest.approx(last["net_value"])


def test_a_projection_without_pockets_is_wholly_in_its_envelope():
    projection = project(1000, 600, monthly_flows([0.0]), 0.0)

    pea = add_net_value(projection, "PEA", date(2010, 1, 1), TAX).iloc[-1]
    cto = add_net_value(projection, "CTO", date(2010, 1, 1), TAX).iloc[-1]

    assert (pea["value_pea"], pea["value_cto"], pea["tax_cto"]) == (1000, 0, 0)
    assert pea["tax_pea"] == pytest.approx(74.4)
    assert (cto["value_cto"], cto["value_pea"], cto["tax_pea"]) == (1000, 0, 0)
    assert cto["tax_cto"] == pytest.approx(125.6)


def test_a_loss_in_one_pocket_does_not_lower_the_tax_of_the_other():
    projection = project_with_ceiling(1000, 600, monthly_flows([500.0, 0.0]), 0.0, 600, 600)
    projection.loc[2, "value_cto"] = 300.0  # the CTO lost 200

    last = add_net_value(projection, "PEA", date(2010, 1, 1), TAX).iloc[-1]

    assert last["tax_cto"] == 0
    assert last["tax"] == pytest.approx(74.4)


def test_fees_cost_with_a_ceiling_takes_the_fees_on_both_pockets():
    flows = monthly_flows([100.0] * 120)

    # same return in both pockets: the split does not change the cost (R7)
    assert fees_cost(0, 0, flows, 0.06, 0.002, ceiling=5000) == pytest.approx(175.76, abs=CENT)
    # a CTO that earns more loses more to the same fees
    assert fees_cost(0, 0, flows, 0.06, 0.002, ceiling=5000, cto_gross=0.09) > 175.76
