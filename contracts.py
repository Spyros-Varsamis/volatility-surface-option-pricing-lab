"""Descriptions and payoff rules for the supported stock options."""

from dataclasses import dataclass
from typing import Literal, Optional

PayoffKind = Literal["call", "put", "digital_call", "digital_put"]
ExerciseKind = Literal["european", "american"]
BarrierKind = Literal["up_and_out", "down_and_out"]


@dataclass(frozen=True)
class OptionContract:
    strike: float
    maturity: float
    payoff: PayoffKind = "call"
    exercise: ExerciseKind = "european"
    cash_payout: float = 1.0
    barrier_kind: Optional[BarrierKind] = None
    barrier: Optional[float] = None

    def __post_init__(self) -> None:
        if self.strike <= 0 or self.maturity <= 0:
            raise ValueError("strike and maturity must be positive")
        if self.cash_payout <= 0:
            raise ValueError("cash payout must be positive")
        if (self.barrier_kind is None) != (self.barrier is None):
            raise ValueError("barrier kind and level must be supplied together")
        if self.barrier is not None and self.barrier <= 0:
            raise ValueError("barrier must be positive")
        if self.exercise == "american" and self.payoff.startswith("digital"):
            raise ValueError("American digitals are outside this project's scope")
        if self.exercise == "american" and self.barrier_kind is not None:
            raise ValueError("American barriers are a later extension")

    def payoff_at(self, stock_price):
        import numpy as np

        prices = np.asarray(stock_price)
        if self.payoff == "call":
            return np.maximum(prices - self.strike, 0.0)
        if self.payoff == "put":
            return np.maximum(self.strike - prices, 0.0)
        if self.payoff == "digital_call":
            return self.cash_payout * (prices > self.strike)
        return self.cash_payout * (prices < self.strike)

