from datetime import date

import pytest

from simulator.models import TaxSettings
from simulator.taxes import (
    CTO,
    LIVRET,
    PEA,
    net_value,
    pea_five_year_date,
    tax_rate,
    withdrawal_tax,
)

SETTINGS = TaxSettings(ps_rate=0.186, ir_rate=0.128, pea_ceiling=150_000, pea_open_date=None)


def test_net_value_after_social_charges_only():
    # X1: gain 67 855.70 taxed at 18.6 %
    assert net_value(120855.70, 53000, 0.186) == pytest.approx(108234.54, abs=0.01)


def test_net_value_after_the_flat_tax():
    # X2: the same gain taxed at 31.4 %
    assert net_value(120855.70, 53000, 0.314) == pytest.approx(99549.01, abs=0.01)


def test_net_value_without_gain_is_the_value():
    assert net_value(9000, 10000, 0.314) == 9000


def test_withdrawal_tax_splits_capital_and_gain_in_proportion():
    # X3: gain 8 000 out of 20 000, so 40 % of the 5 000 withdrawn is gain
    tax, capital = withdrawal_tax(5000, 20000, 12000, 0.186)

    assert tax == pytest.approx(372.00)
    assert capital == pytest.approx(3000.00)
    assert 5000 - tax == pytest.approx(4628.00)  # net


def test_no_tax_on_a_loss():
    # X4: value below the deposits
    tax, capital = withdrawal_tax(1000, 9000, 10000, 0.186)

    assert tax == 0
    assert capital == pytest.approx(1000.00)


def test_withdrawal_from_an_empty_plan_has_no_tax():
    assert withdrawal_tax(100, 0, 0, 0.186) == (0.0, 100)


def test_tax_rate_of_a_pea_changes_on_its_fifth_anniversary():
    # X5: opened on 15/01/2020
    opened = date(2020, 1, 15)

    assert tax_rate(PEA, opened, date(2025, 1, 14), SETTINGS) == pytest.approx(0.314)
    assert tax_rate(PEA, opened, date(2025, 1, 15), SETTINGS) == pytest.approx(0.186)
    assert tax_rate(CTO, opened, date(2025, 1, 15), SETTINGS) == pytest.approx(0.314)
    assert tax_rate(LIVRET, opened, date(2025, 1, 15), SETTINGS) == 0


def test_pea_without_opening_date_gets_the_flat_tax():
    assert tax_rate(PEA, None, date(2030, 1, 1), SETTINGS) == pytest.approx(0.314)


def test_unknown_envelope_is_refused():
    with pytest.raises(ValueError):
        tax_rate("ASSURANCE", None, date(2030, 1, 1), SETTINGS)


def test_five_year_date_of_a_pea_opened_on_29_february():
    assert pea_five_year_date(date(2020, 1, 15)) == date(2025, 1, 15)
    assert pea_five_year_date(date(2020, 2, 29)) == date(2025, 3, 1)
