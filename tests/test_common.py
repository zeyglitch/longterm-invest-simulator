from views.common import format_eur, format_pct


def test_euros_have_two_decimals_and_spaces_between_thousands():
    assert format_eur(12345.67) == "12 345.67 €"
    assert format_eur(1234567.891) == "1 234 567.89 €"
    assert format_eur(-5) == "-5.00 €"
    assert format_eur(None) == "—"


def test_percentages_have_one_decimal_by_default():
    assert format_pct(0.1337) == "13.4 %"
    assert format_pct(0.186, decimals=2) == "18.60 %"
    assert format_pct(None) == "—"
