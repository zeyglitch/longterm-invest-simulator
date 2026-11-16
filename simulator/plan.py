"""A plan of future deposits turned into one flow per month. See 03-finance-rules, section 2.3."""

import json
from dataclasses import asdict, replace
from datetime import date

import pandas as pd

from simulator.models import OneOff, Phase, Plan


def _month(day: date) -> pd.Period:
    return pd.Period(day, freq="M")


def _amount_in(phase: Phase, month: pd.Period) -> float:
    """Monthly amount of a phase in a given month, with its yearly raises."""
    # the raise comes at each anniversary of the phase, not on 1 January
    years_in_phase = (month - _month(phase.start)).n // 12
    return phase.monthly_amount * (1 + phase.annual_increase) ** years_in_phase


def build_flow_parts(
    plan: Plan, start: date, inflation: float = 0.0, index_on_inflation: bool = False
) -> pd.DataFrame:
    """Deposits and withdrawals of each month, from the month after `start` to the month of
    `plan.end` included, as two columns `deposit` and `withdrawal` (both positive or zero).

    The index holds the month ends. A phase applies from the month of its start to the month of
    its end; a month without phase is worth 0. A one-off goes to the month of its date: to
    `deposit` if positive, to `withdrawal` if negative. With `index_on_inflation`, the phase
    amounts follow inflation, by yearly steps. The two are kept apart because several rules look
    at one only: the PEA ceiling counts deposits, unemployment stops deposits, not withdrawals.
    """
    months = pd.period_range(_month(start) + 1, _month(plan.end), freq="M")
    deposits = pd.Series(0.0, index=months)
    withdrawals = pd.Series(0.0, index=months)

    for phase in plan.phases:
        first, last = _month(phase.start), _month(phase.end)
        for number, month in enumerate(months):
            if not first <= month <= last:
                continue
            amount = _amount_in(phase, month)
            if index_on_inflation:
                amount *= (1 + inflation) ** (number // 12)
            deposits[month] += amount

    for one_off in plan.one_offs:
        month = _month(one_off.date)
        if month not in months:
            continue
        if one_off.amount >= 0:
            deposits[month] += one_off.amount
        else:
            withdrawals[month] -= one_off.amount

    parts = pd.DataFrame({"deposit": deposits, "withdrawal": withdrawals})
    parts.index = months.to_timestamp(how="end").normalize()
    return parts


def build_flows(
    plan: Plan, start: date, inflation: float = 0.0, index_on_inflation: bool = False
) -> pd.Series:
    """One net flow per month: deposits minus withdrawals (see `build_flow_parts`)."""
    parts = build_flow_parts(plan, start, inflation, index_on_inflation)
    return parts["deposit"] - parts["withdrawal"]


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


def plan_to_frames(plan: Plan) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The phases and the one-offs as two tables, for the editable tables of the Plan page.

    The yearly increase is shown as a percentage (2 for 2 %).
    """
    phases = pd.DataFrame(
        {
            "label": pd.Series([p.label for p in plan.phases], dtype="object"),
            "start": pd.to_datetime([p.start for p in plan.phases]),
            "end": pd.to_datetime([p.end for p in plan.phases]),
            "monthly_amount": pd.Series([p.monthly_amount for p in plan.phases], dtype="float"),
            "increase_pct": pd.Series(
                [p.annual_increase * 100 for p in plan.phases], dtype="float"
            ),
        }
    )
    one_offs = pd.DataFrame(
        {
            "label": pd.Series([o.label for o in plan.one_offs], dtype="object"),
            "date": pd.to_datetime([o.date for o in plan.one_offs]),
            "amount": pd.Series([o.amount for o in plan.one_offs], dtype="float"),
        }
    )
    return phases, one_offs


def plan_from_frames(
    name: str, end: date, phases: pd.DataFrame, one_offs: pd.DataFrame
) -> tuple[Plan, int]:
    """Build a plan from the two edited tables. Also returns the number of rows left out
    because a date or an amount is still missing (a row being typed)."""
    skipped = 0
    plan = Plan(name=name.strip(), end=end, phases=[], one_offs=[])
    for row in phases.to_dict("records"):
        if pd.isna(row["start"]) or pd.isna(row["end"]) or pd.isna(row["monthly_amount"]):
            skipped += 1
            continue
        increase = 0.0 if pd.isna(row["increase_pct"]) else row["increase_pct"] / 100
        plan.phases.append(
            Phase(
                label="" if pd.isna(row["label"]) else str(row["label"]).strip(),
                start=pd.Timestamp(row["start"]).date(),
                end=pd.Timestamp(row["end"]).date(),
                monthly_amount=float(row["monthly_amount"]),
                annual_increase=round(float(increase), 6),
            )
        )
    for row in one_offs.to_dict("records"):
        if pd.isna(row["date"]) or pd.isna(row["amount"]):
            skipped += 1
            continue
        plan.one_offs.append(
            OneOff(
                label="" if pd.isna(row["label"]) else str(row["label"]).strip(),
                date=pd.Timestamp(row["date"]).date(),
                amount=float(row["amount"]),
            )
        )
    return plan, skipped


# Life event helpers. Each one returns a new plan and leaves the given plan untouched.
# They only write phases and one-offs: the user can then edit them like any other line.


def _first_day(month: pd.Period) -> date:
    return month.start_time.date()


def _last_day(month: pd.Period) -> date:
    return month.end_time.date()


def _months_from(day: date, months: int) -> tuple[pd.Period, pd.Period]:
    """First and last month of a period of `months` months starting in the month of `day`."""
    first = _month(day)
    return first, first + max(months, 1) - 1


def _next_anniversary(phase: Phase, month: pd.Period) -> pd.Period:
    """First month, from `month` on, in which the phase gets its yearly raise."""
    months_in_phase = (month - _month(phase.start)).n
    return month + (-months_in_phase % 12)


def _fixed_pieces(
    phase: Phase, first: pd.Period, last: pd.Period, label: str, decrease: float = 0.0
) -> list[Phase]:
    """The months `first` to `last` of a phase, as phases without raise, `decrease` euros lower.

    A phase with a yearly raise is cut at each of its anniversaries, so that every piece has the
    amount the phase had that year. Without raise, one piece is enough.
    """
    pieces = []
    month = first
    while month <= last:
        end = last
        if phase.annual_increase != 0:
            end = min(_next_anniversary(phase, month + 1) - 1, last)
        amount = max(0.0, _amount_in(phase, month) - decrease)
        pieces.append(Phase(label, _first_day(month), _last_day(end), amount))
        month = end + 1
    return pieces


def _without_months(phases: list[Phase], first: pd.Period, last: pd.Period) -> list[Phase]:
    """The phases with the months `first` to `last` cut out. A phase that spans the window is
    split in two. The part after the window keeps the amounts and the raise dates the phase
    would have had: if it does not resume on an anniversary, a first piece without raise runs
    to the next anniversary."""
    kept = []
    for phase in phases:
        start, end = _month(phase.start), _month(phase.end)
        if end < first or start > last:
            kept.append(replace(phase))
            continue
        if start < first:
            kept.append(replace(phase, end=_last_day(first - 1)))
        if end > last:
            resumed = last + 1
            anniversary = (
                resumed if phase.annual_increase == 0 else _next_anniversary(phase, resumed)
            )
            if anniversary > resumed:
                before = min(anniversary - 1, end)
                kept.extend(_fixed_pieces(phase, resumed, before, phase.label))
            if anniversary <= end:
                kept.append(
                    replace(
                        phase,
                        start=_first_day(anniversary),
                        monthly_amount=_amount_in(phase, anniversary),
                    )
                )
    return kept


def _sorted(phases: list[Phase]) -> list[Phase]:
    return sorted(phases, key=lambda phase: phase.start)


def _reduced(plan: Plan, day: date, months: int, decrease: float, reason: str) -> list[Phase]:
    """The phases with `decrease` euros less per month during `months` months (never below 0)."""
    first, last = _months_from(day, months)
    phases = _without_months(plan.phases, first, last)
    for phase in plan.phases:
        start, end = max(_month(phase.start), first), min(_month(phase.end), last)
        if start > end:
            continue
        label = f"{phase.label} ({reason})".strip()
        phases.extend(_fixed_pieces(phase, start, end, label, decrease))
    return _sorted(phases)


def add_job_change(plan: Plan, day: date, new_monthly_amount: float) -> Plan:
    """New job or raise: the phase in progress ends, a new one starts in the month of `day`.

    The new phase runs to the end of the phase it replaces. If no phase covers that month, it
    runs to the next phase, or to the end of the plan.
    """
    first = _month(day)
    current = [p for p in plan.phases if _month(p.start) <= first <= _month(p.end)]
    later = [_month(p.start) for p in plan.phases if _month(p.start) > first]
    if current:
        last = _month(current[0].end)
    elif later:
        last = min(later) - 1
    else:
        last = max(_month(plan.end), first)
    phases = _without_months(plan.phases, first, last)
    phases.append(Phase("New job", _first_day(first), _last_day(last), new_monthly_amount))
    return replace(plan, phases=_sorted(phases), one_offs=list(plan.one_offs))


def add_unemployment(plan: Plan, start: date, months: int) -> Plan:
    """A pause: nothing is deposited during `months` months, then the phases resume."""
    first, last = _months_from(start, months)
    phases = _without_months(plan.phases, first, last)
    phases.append(Phase("Unemployment", _first_day(first), _last_day(last), 0.0))
    return replace(plan, phases=_sorted(phases), one_offs=list(plan.one_offs))


def add_child(plan: Plan, day: date, monthly_decrease: float, years: int) -> Plan:
    """A child: the monthly deposits are lowered by `monthly_decrease` during `years` years."""
    phases = _reduced(plan, day, years * 12, monthly_decrease, "child")
    return replace(plan, phases=phases, one_offs=list(plan.one_offs))


def add_home_purchase(
    plan: Plan, day: date, down_payment: float, monthly_decrease: float, loan_years: int
) -> Plan:
    """A home: the down payment is withdrawn, then deposits are lowered while the loan runs."""
    phases = _reduced(plan, day, loan_years * 12, monthly_decrease, "home loan")
    one_offs = [*plan.one_offs, OneOff("Home down payment", day, -abs(down_payment))]
    return replace(plan, phases=phases, one_offs=one_offs)


def add_big_purchase(plan: Plan, day: date, amount: float, label: str = "Big purchase") -> Plan:
    """A car or another large expense: one withdrawal."""
    one_offs = [*plan.one_offs, OneOff(label, day, -abs(amount))]
    return replace(plan, phases=[replace(p) for p in plan.phases], one_offs=one_offs)


def add_windfall(plan: Plan, day: date, amount: float, label: str = "Windfall") -> Plan:
    """A bonus or a gift: one extra deposit."""
    one_offs = [*plan.one_offs, OneOff(label, day, abs(amount))]
    return replace(plan, phases=[replace(p) for p in plan.phases], one_offs=one_offs)
