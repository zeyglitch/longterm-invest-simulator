from simulator import config


def test_default_rates_are_between_0_and_1():
    rates = [
        config.DEFAULT_PS_RATE,
        config.DEFAULT_IR_RATE,
        config.DEFAULT_LIVRET_RATE,
        config.DEFAULT_INFLATION,
        config.DEFAULT_VOLATILITY,
        *config.DEFAULT_SCENARIOS.values(),
    ]
    assert all(0 < rate < 1 for rate in rates)
