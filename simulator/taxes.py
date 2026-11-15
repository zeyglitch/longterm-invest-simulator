"""Tax rules of the PEA, the CTO and the Livret A. Rates are decimals: 0.186 for 18.6%."""

from datetime import date

from simulator.models import TaxSettings

PEA = "PEA"
CTO = "CTO"
LIVRET = "LIVRET"
PEA_YEARS = 5


def pea_five_year_date(open_date: date) -> date:
    """The day from which a PEA is five years old."""
    try:
        return open_date.replace(year=open_date.year + PEA_YEARS)
    except ValueError:  # opened on 29 February, and the fifth year has no 29th: take 1 March
        return date(open_date.year + PEA_YEARS, 3, 1)


def tax_rate(envelope: str, open_date: date | None, on_date: date, settings: TaxSettings) -> float:
    """Tax rate on the gain of a withdrawal made on `on_date`.

    PEA: social charges only once five years old, otherwise the flat tax (PFU). Without an
    opening date we cannot tell, so the flat tax applies. CTO: flat tax. Livret A: none.
    """
    if envelope == LIVRET:
        return 0.0
    if envelope not in (PEA, CTO):
        raise ValueError(f"Unknown envelope: {envelope}")
    flat_tax = settings.ir_rate + settings.ps_rate
    if envelope == PEA and open_date is not None and on_date >= pea_five_year_date(open_date):
        return settings.ps_rate
    return flat_tax


def withdrawal_tax(
    amount: float, value: float, contributions: float, rate: float
) -> tuple[float, float]:
    """Tax and capital part of a withdrawal, by the proportional rule (03-finance-rules 4.2).

    `value` is the value before the withdrawal and `contributions` the net deposits. A withdrawal
    contains capital and gain in the same proportion as the whole plan. Returns (tax, capital).
    """
    if value <= 0:
        return 0.0, amount
    gain_part = amount * max(0.0, value - contributions) / value
    return gain_part * rate, amount - gain_part


def net_value(value: float, contributions: float, rate: float) -> float:
    """What would be left after tax if everything were withdrawn."""
    return value - max(0.0, value - contributions) * rate
