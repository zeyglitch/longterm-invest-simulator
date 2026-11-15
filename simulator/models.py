from dataclasses import dataclass
from datetime import date


@dataclass
class Asset:
    id: int | None
    isin: str
    name: str
    ticker: str | None
    fees: float = 0.0  # yearly fees as a decimal: 0.002 for 0.20%
    target_weight: float | None = None  # decimal: 0.75 for 75%


@dataclass
class Transaction:
    id: int | None
    date: date
    asset_id: int
    kind: str  # "BUY" or "SELL"
    quantity: float
    unit_price: float
    fees: float = 0.0
    note: str = ""
    source: str = "manual"  # "manual" or "import"


@dataclass
class CashMovement:
    id: int | None
    date: date
    kind: str  # "DEPOSIT" or "WITHDRAWAL"
    amount: float
    note: str = ""


@dataclass
class Position:
    asset: Asset
    quantity: float
    avg_cost: float
    invested: float
    last_price: float | None
    price_date: date | None
    value: float | None
    gain: float | None
    gain_pct: float | None  # decimal: 0.1337 for 13.37%
    weight: float | None  # share of the total value of priced positions


@dataclass
class TaxSettings:
    ps_rate: float  # social charges, decimal: 0.186 for 18.6%
    ir_rate: float  # income tax part of the flat tax (PFU)
    pea_ceiling: float  # maximum total of deposits on the PEA, in euros
    pea_open_date: date | None  # needed to know when the 5 years are over
