from simulator.models import Asset
from views.common import asset_label, format_eur, format_pct, format_quantity


def test_euros_have_two_decimals_and_spaces_between_thousands():
    assert format_eur(12345.67) == "12 345.67 €"
    assert format_eur(1234567.891) == "1 234 567.89 €"
    assert format_eur(-5) == "-5.00 €"
    assert format_eur(None) == "—"


def test_percentages_have_one_decimal_by_default():
    assert format_pct(0.1337) == "13.4 %"
    assert format_pct(0.186, decimals=2) == "18.60 %"
    assert format_pct(None) == "—"


def test_quantities_drop_useless_zeros():
    assert format_quantity(651.0) == "651"
    assert format_quantity(2.5) == "2.5"
    assert format_quantity(0.12345) == "0.1235"
    assert format_quantity(1234.0) == "1 234"
    assert format_quantity(100.0) == "100"


def test_two_assets_with_the_same_name_have_different_labels():
    first = Asset(1, "XX0000000001", "Same name", None)
    second = Asset(2, "XX0000000002", "Same name", None)

    assert asset_label(first) == "Same name (XX0000000001)"
    assert asset_label(first) != asset_label(second)
