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
