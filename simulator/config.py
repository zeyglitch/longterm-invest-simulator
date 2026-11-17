"""Default values and paths. Rates are decimals: 0.06 for 6%."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
# LTIS_DB points to another file, to try the app on a copy without touching the real data
DB_PATH = Path(os.environ.get("LTIS_DB") or ROOT_DIR / "data" / "portfolio.db")

DEFAULT_PS_RATE = 0.186
DEFAULT_IR_RATE = 0.128
DEFAULT_PEA_CEILING = 150_000
DEFAULT_LIVRET_RATE = 0.017
DEFAULT_LIVRET_CEILING = 22_950
DEFAULT_INFLATION = 0.02

DEFAULT_SCENARIOS = {"Pessimistic": 0.03, "Central": 0.06, "Optimistic": 0.08}
# gross yearly return of the CTO that takes the deposits above the PEA ceiling
DEFAULT_CTO_RETURN = 0.07
DEFAULT_VOLATILITY = 0.15
DEFAULT_N_PATHS = 5_000
DEFAULT_SEED = 42
DEFAULT_BENCHMARK = "CW8.PA"

# the default rates above were last checked on this date
RATES_VERIFIED_ON = "13/11/2026"
