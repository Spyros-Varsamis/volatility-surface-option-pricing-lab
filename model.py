"""The stock-price model used by every pricing method in the project.

Under the risk-neutral measure, geometric Brownian motion is

    dS = (r - q) S dt + sigma S dW.

It keeps a positive starting stock price positive. Under the pricing measure,
the expected growth rate is the risk-free rate minus dividend yield—not an
investor's forecast of the stock's real-world return.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BlackScholesModel:
    rate: float
    dividend: float
    volatility: float

    def __post_init__(self) -> None:
        if self.volatility <= 0:
            raise ValueError("volatility must be positive")

