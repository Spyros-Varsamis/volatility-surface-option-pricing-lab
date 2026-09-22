"""Black-Scholes closed forms used as trusted European benchmarks."""

from math import erf, exp, log, pi, sqrt

from contracts import OptionContract
from model import BlackScholesModel


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def normal_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def black_scholes_price(
    spot: float, model: BlackScholesModel, contract: OptionContract
) -> float:
    """Price a non-barrier European vanilla or cash-digital option."""
    if spot <= 0:
        raise ValueError("spot must be positive")
    if contract.exercise != "european" or contract.barrier_kind is not None:
        raise ValueError("This closed form supports plain European contracts")

    maturity = contract.maturity
    vol_sqrt_t = model.volatility * sqrt(maturity)
    d1 = (
        log(spot / contract.strike)
        + (model.rate - model.dividend + 0.5 * model.volatility**2) * maturity
    ) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    stock_discount = exp(-model.dividend * maturity)
    cash_discount = exp(-model.rate * maturity)

    if contract.payoff == "call":
        return spot * stock_discount * normal_cdf(d1) - contract.strike * cash_discount * normal_cdf(d2)
    if contract.payoff == "put":
        return contract.strike * cash_discount * normal_cdf(-d2) - spot * stock_discount * normal_cdf(-d1)
    if contract.payoff == "digital_call":
        return contract.cash_payout * cash_discount * normal_cdf(d2)
    return contract.cash_payout * cash_discount * normal_cdf(-d2)
