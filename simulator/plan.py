"""A plan of future deposits turned into one flow per month. See 03-finance-rules, section 2.3."""

import json
from dataclasses import asdict
from datetime import date

import pandas as pd

from simulator.models import OneOff, Phase, Plan


def _month(day: date) -> pd.Period:
    return pd.Period(day, freq="M")


def build_flows(
    plan: Plan, start: date, inflation: float = 0.0, index_on_inflation: bool = False
) -> pd.Series:
    """One flow per month, from the month after `start` to the month of `plan.end` included.

    The index holds the month ends. A phase applies from the month of its start to the month of
    its end; a month without phase is worth 0. A one-off is added to the month of its date.
    With `index_on_inflation`, the phase amounts follow inflation, by yearly steps.
    """
    months = pd.period_range(_month(start) + 1, _month(plan.end), freq="M")
    flows = pd.Series(0.0, index=months)

    for phase in plan.phases:
        first, last = _month(phase.start), _month(phase.end)
        for number, month in enumerate(months):
            if not first <= month <= last:
                continue
            # the raise comes at each anniversary of the phase, not on 1 January
            years_in_phase = (month - first).n // 12
            amount = phase.monthly_amount * (1 + phase.annual_increase) ** years_in_phase
            if index_on_inflation:
                amount *= (1 + inflation) ** (number // 12)
            flows[month] += amount

    for one_off in plan.one_offs:
        month = _month(one_off.date)
        if month in flows.index:
            flows[month] += one_off.amount

    flows.index = months.to_timestamp(how="end").normalize()
    return flows


def validate_plan(plan: Plan) -> list[str]:
    """Messages for what is wrong in a plan (empty list if it is valid)."""
    errors = []
    if not plan.name.strip():
        errors.append("The plan needs a name.")
    for phase in plan.phases:
        name = phase.label or "A phase"
        if phase.end < phase.start:
            errors.append(f"{name}: the end date is before the start date.")
        if phase.monthly_amount < 0:
            errors.append(f"{name}: the monthly amount cannot be negative. Use a one-off.")
        if phase.annual_increase <= -1:
            errors.append(f"{name}: the yearly increase must be above -100 %.")

    # a phase covers whole months, so two phases overlap as soon as they share a month
    ordered = sorted(plan.phases, key=lambda phase: phase.start)
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        if _month(later.start) <= _month(earlier.end):
            errors.append(
                f"{earlier.label or 'A phase'} and {later.label or 'another phase'} overlap: "
                "two phases cannot cover the same month."
            )
    return errors


def plan_to_json(plan: Plan) -> str:
    """The plan as JSON text, with dates written as YYYY-MM-DD."""
    return json.dumps(asdict(plan), default=lambda day: day.isoformat())


def plan_from_json(text: str) -> Plan:
    data = json.loads(text)
    phases = [
        Phase(
            label=phase["label"],
            start=date.fromisoformat(phase["start"]),
            end=date.fromisoformat(phase["end"]),
            monthly_amount=phase["monthly_amount"],
            annual_increase=phase["annual_increase"],
        )
        for phase in data["phases"]
    ]
    one_offs = [
        OneOff(label=one["label"], date=date.fromisoformat(one["date"]), amount=one["amount"])
        for one in data["one_offs"]
    ]
    return Plan(
        name=data["name"], end=date.fromisoformat(data["end"]), phases=phases, one_offs=one_offs
    )
