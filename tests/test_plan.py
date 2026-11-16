from datetime import date

import pandas as pd
import pytest

from simulator.database import delete_plan, list_plan_names, load_plan, save_plan
from simulator.models import OneOff, Phase, Plan
from simulator.plan import build_flows, plan_from_json, plan_to_json, validate_plan
from simulator.projection import project

START = date(2026, 12, 15)  # month 1 is January 2027


def plan_with(phases=(), one_offs=(), end=date(2029, 12, 31), name="Test plan") -> Plan:
    return Plan(name=name, end=end, phases=list(phases), one_offs=list(one_offs))


def test_flows_start_the_month_after_the_start_date_and_end_with_the_plan():
    flows = build_flows(plan_with(end=date(2027, 3, 10)), START)

    assert list(flows.index) == list(pd.to_datetime(["2027-01-31", "2027-02-28", "2027-03-31"]))
    assert flows.tolist() == [0.0, 0.0, 0.0]


def test_two_phases_give_the_reference_projection():
    # R4: 100 a month for 24 months, then 300 a month for 36 months, at 6 %
    plan = plan_with(
        [
            Phase("Studies", date(2027, 1, 1), date(2028, 12, 31), 100.0),
            Phase("First job", date(2029, 1, 1), date(2031, 12, 31), 300.0),
        ],
        end=date(2031, 12, 31),
    )

    flows = build_flows(plan, START)
    projection = project(0, 0, flows, 0.06)

    assert flows.tolist() == [100.0] * 24 + [300.0] * 36
    assert projection["value"].iloc[-1] == pytest.approx(14797.13, abs=0.01)
    assert projection["contributions"].iloc[-1] == pytest.approx(13200)


def test_yearly_increase_applies_at_each_anniversary_of_the_phase():
    # F1: 100 a month, +10 % a year, 36 months
    plan = plan_with([Phase("Job", date(2027, 1, 1), date(2029, 12, 31), 100.0, 0.10)])

    flows = build_flows(plan, START)

    assert flows.tolist() == pytest.approx([100.0] * 12 + [110.0] * 12 + [121.0] * 12)


def test_anniversary_counts_from_the_phase_start_not_from_january():
    plan = plan_with([Phase("Job", date(2027, 7, 1), date(2028, 12, 31), 100.0, 0.10)])

    flows = build_flows(plan, START)

    assert flows.tolist()[:6] == [0.0] * 6  # January to June 2027: no phase yet
    assert flows.tolist()[6:18] == pytest.approx([100.0] * 12)  # July 2027 to June 2028
    assert flows.tolist()[18:24] == pytest.approx([110.0] * 6)  # raise in July 2028


def test_phases_can_follow_inflation():
    # F2: 100 a month indexed on 2 % of inflation
    plan = plan_with([Phase("Job", date(2027, 1, 1), date(2028, 12, 31), 100.0)])

    indexed = build_flows(plan, START, inflation=0.02, index_on_inflation=True)
    not_indexed = build_flows(plan, START, inflation=0.02)

    assert indexed.tolist()[:24] == pytest.approx([100.0] * 12 + [102.0] * 12)
    assert not_indexed.tolist()[:24] == [100.0] * 24


def test_one_offs_are_added_to_their_month_and_never_indexed():
    plan = plan_with(
        [Phase("Job", date(2027, 1, 1), date(2029, 12, 31), 100.0)],
        [
            OneOff("Bonus", date(2027, 3, 20), 1000.0),
            OneOff("Car", date(2028, 2, 5), -500.0),
            OneOff("Too late", date(2035, 1, 1), 999.0),
            OneOff("Too early", date(2026, 12, 20), 999.0),
        ],
    )

    flows = build_flows(plan, START, inflation=0.02, index_on_inflation=True)

    assert flows["2027-03-31"] == pytest.approx(1100.0)
    assert flows["2028-02-29"] == pytest.approx(102.0 - 500.0)
    assert flows.sum() == pytest.approx(1200 + 1224 + 1248.48 + 1000 - 500)


def test_a_gap_between_two_phases_is_worth_zero():
    # F3, second half
    plan = plan_with(
        [
            Phase("Before", date(2027, 1, 1), date(2027, 3, 31), 100.0),
            Phase("After", date(2027, 6, 1), date(2027, 8, 31), 200.0),
        ],
        end=date(2027, 8, 31),
    )

    assert validate_plan(plan) == []
    assert build_flows(plan, START).tolist() == [100.0] * 3 + [0.0] * 2 + [200.0] * 3


def test_overlapping_phases_are_refused():
    # F3, first half: both cover March 2027
    plan = plan_with(
        [
            Phase("Before", date(2027, 1, 1), date(2027, 3, 15), 100.0),
            Phase("After", date(2027, 3, 20), date(2027, 8, 31), 200.0),
        ]
    )

    errors = validate_plan(plan)

    assert len(errors) == 1
    assert "Before and After overlap" in errors[0]


@pytest.mark.parametrize(
    "phase",
    [
        Phase("Job", date(2027, 6, 1), date(2027, 1, 1), 100.0),  # ends before it starts
        Phase("Job", date(2027, 1, 1), date(2027, 6, 1), -50.0),  # negative amount
        Phase("Job", date(2027, 1, 1), date(2027, 6, 1), 100.0, -1.0),  # -100 % a year
    ],
)
def test_invalid_phases_are_reported(phase):
    assert len(validate_plan(plan_with([phase]))) == 1


def test_a_plan_needs_a_name():
    assert validate_plan(plan_with(name="  ")) == ["The plan needs a name."]


def full_plan() -> Plan:
    return plan_with(
        [Phase("Job", date(2027, 1, 1), date(2029, 12, 31), 100.0, 0.02)],
        [OneOff("Bonus", date(2027, 3, 20), 1000.0)],
    )


def test_a_plan_survives_json():
    # F4
    plan = full_plan()

    assert plan_from_json(plan_to_json(plan)) == plan


def test_plans_are_saved_by_name_and_can_be_replaced(conn):
    plan = full_plan()
    save_plan(conn, plan)
    assert load_plan(conn, "Test plan") == plan

    plan.phases[0].monthly_amount = 250.0
    save_plan(conn, plan)

    assert load_plan(conn, "Test plan").phases[0].monthly_amount == 250.0
    assert list_plan_names(conn) == ["Test plan"]


def test_unknown_or_deleted_plan_is_none(conn):
    save_plan(conn, full_plan())

    delete_plan(conn, "Test plan")

    assert load_plan(conn, "Test plan") is None
    assert list_plan_names(conn) == []
