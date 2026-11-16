from datetime import date

import pandas as pd
import pytest

from simulator.database import delete_plan, list_plan_names, load_plan, save_plan
from simulator.models import OneOff, Phase, Plan
from simulator.plan import (
    add_big_purchase,
    add_child,
    add_home_purchase,
    add_job_change,
    add_unemployment,
    add_windfall,
    build_flows,
    plan_from_frames,
    plan_from_json,
    plan_to_frames,
    plan_to_json,
    validate_plan,
)
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


# Life event helpers, on a simple plan: 300 a month from 2027 to 2036


def simple_plan() -> Plan:
    return plan_with(
        [Phase("Job", date(2027, 1, 1), date(2036, 12, 31), 300.0)], end=date(2036, 12, 31)
    )


def phase_summary(plan: Plan) -> list[tuple]:
    return [(p.label, p.start, p.end, p.monthly_amount) for p in plan.phases]


def test_job_change_ends_the_current_phase_and_opens_a_new_one():
    plan = add_job_change(simple_plan(), date(2030, 9, 12), 500.0)

    assert phase_summary(plan) == [
        ("Job", date(2027, 1, 1), date(2030, 8, 31), 300.0),
        ("New job", date(2030, 9, 1), date(2036, 12, 31), 500.0),
    ]
    assert validate_plan(plan) == []


def test_job_change_outside_any_phase_runs_to_the_next_phase_or_the_end():
    later = plan_with([Phase("Later", date(2030, 1, 1), date(2031, 12, 31), 300.0)])

    before = add_job_change(later, date(2028, 3, 1), 150.0)
    after = add_job_change(later, date(2033, 3, 1), 600.0)  # the plan ends on 31/12/2029

    assert phase_summary(before)[0] == ("New job", date(2028, 3, 1), date(2029, 12, 31), 150.0)
    assert phase_summary(after)[-1] == ("New job", date(2033, 3, 1), date(2033, 3, 31), 600.0)
    assert validate_plan(before) == []
    assert validate_plan(after) == []


def test_unemployment_pauses_the_deposits_then_they_resume():
    plan = add_unemployment(simple_plan(), date(2029, 4, 10), 6)

    assert phase_summary(plan) == [
        ("Job", date(2027, 1, 1), date(2029, 3, 31), 300.0),
        ("Unemployment", date(2029, 4, 1), date(2029, 9, 30), 0.0),
        ("Job", date(2029, 10, 1), date(2036, 12, 31), 300.0),
    ]
    assert validate_plan(plan) == []
    assert build_flows(plan, START).sum() == pytest.approx(300.0 * (120 - 6))


def test_a_phase_with_raises_resumes_at_the_amount_it_had_reached():
    raised = plan_with([Phase("Job", date(2027, 1, 1), date(2036, 12, 31), 100.0, 0.10)])

    plan = add_unemployment(raised, date(2029, 1, 1), 12)  # the third year is lost

    assert plan.phases[-1].start == date(2030, 1, 1)
    assert plan.phases[-1].monthly_amount == pytest.approx(133.1)  # 100 x 1.1 ^ 3


def test_child_lowers_the_deposits_for_some_years():
    plan = add_child(simple_plan(), date(2030, 6, 20), 100.0, 3)

    assert phase_summary(plan) == [
        ("Job", date(2027, 1, 1), date(2030, 5, 31), 300.0),
        ("Job (child)", date(2030, 6, 1), date(2033, 5, 31), 200.0),
        ("Job", date(2033, 6, 1), date(2036, 12, 31), 300.0),
    ]
    assert validate_plan(plan) == []


def test_a_decrease_never_makes_a_deposit_negative_and_spans_several_phases():
    two = plan_with(
        [
            Phase("Small", date(2027, 1, 1), date(2028, 12, 31), 50.0),
            Phase("Big", date(2029, 1, 1), date(2030, 12, 31), 400.0),
        ],
        end=date(2030, 12, 31),
    )

    plan = add_child(two, date(2028, 7, 1), 100.0, 1)  # July 2028 to June 2029

    assert phase_summary(plan) == [
        ("Small", date(2027, 1, 1), date(2028, 6, 30), 50.0),
        ("Small (child)", date(2028, 7, 1), date(2028, 12, 31), 0.0),
        ("Big (child)", date(2029, 1, 1), date(2029, 6, 30), 300.0),
        ("Big", date(2029, 7, 1), date(2030, 12, 31), 400.0),
    ]
    assert validate_plan(plan) == []


def test_home_purchase_withdraws_the_down_payment_and_lowers_the_deposits():
    plan = add_home_purchase(simple_plan(), date(2032, 3, 15), 20_000.0, 150.0, 20)

    assert plan.one_offs == [OneOff("Home down payment", date(2032, 3, 15), -20_000.0)]
    assert phase_summary(plan) == [
        ("Job", date(2027, 1, 1), date(2032, 2, 29), 300.0),
        ("Job (home loan)", date(2032, 3, 1), date(2036, 12, 31), 150.0),  # to the phase end
    ]
    assert validate_plan(plan) == []


def test_big_purchase_and_windfall_add_one_off_events():
    plan = add_windfall(
        add_big_purchase(simple_plan(), date(2030, 5, 2), 8000.0), date(2031, 1, 9), 2500
    )

    assert [(o.label, o.amount) for o in plan.one_offs] == [
        ("Big purchase", -8000.0),
        ("Windfall", 2500),
    ]
    assert phase_summary(plan) == phase_summary(simple_plan())


def test_helpers_leave_the_original_plan_untouched():
    original = simple_plan()

    changed = add_child(add_windfall(original, date(2030, 1, 1), 100.0), date(2030, 1, 1), 50.0, 1)
    changed.phases[0].monthly_amount = 999.0

    assert original == simple_plan()


def test_a_plan_survives_the_editable_tables():
    plan = full_plan()

    phases, one_offs = plan_to_frames(plan)
    rebuilt, skipped = plan_from_frames(plan.name, plan.end, phases, one_offs)

    assert rebuilt == plan
    assert skipped == 0
    assert phases["increase_pct"].tolist() == pytest.approx([2.0])  # shown as a percentage


def test_rows_being_typed_are_left_out_and_counted():
    phases, one_offs = plan_to_frames(full_plan())
    phases.loc[1] = ["Half typed", pd.Timestamp("2030-01-01"), pd.NaT, 100.0, None]
    phases.loc[2] = [None, pd.Timestamp("2031-01-01"), pd.Timestamp("2031-12-31"), 50.0, None]
    one_offs.loc[1] = ["No amount", pd.Timestamp("2030-01-01"), None]

    plan, skipped = plan_from_frames(" Mine ", date(2040, 12, 31), phases, one_offs)

    assert skipped == 2
    assert plan.name == "Mine"
    assert [(p.label, p.annual_increase) for p in plan.phases] == [("Job", 0.02), ("", 0.0)]
    assert len(plan.one_offs) == 1


def test_an_empty_plan_gives_empty_tables_with_the_right_columns():
    phases, one_offs = plan_to_frames(plan_with())

    assert list(phases.columns) == ["label", "start", "end", "monthly_amount", "increase_pct"]
    assert list(one_offs.columns) == ["label", "date", "amount"]
    assert plan_from_frames("Empty", date(2040, 1, 1), phases, one_offs)[0].phases == []
