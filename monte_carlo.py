"""Risk-neutral Monte Carlo and Longstaff-Schwartz pricing."""

from dataclasses import dataclass
from math import exp, sqrt
from typing import Optional

import numpy as np

from contracts import OptionContract
from model import BlackScholesModel


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    standard_error: float

    @property
    def confidence_interval_95(self):
        width = 1.96 * self.standard_error
        return self.price - width, self.price + width


def simulate_paths(
    spot: float,
    model: BlackScholesModel,
    maturity: float,
    paths: int,
    steps: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate exact GBM transitions; return shape (steps + 1, paths)."""
    if spot <= 0 or paths < 2 or steps < 1:
        raise ValueError("Use a positive spot, at least 2 paths, and at least 1 step")
    rng = np.random.default_rng(seed)
    dt = maturity / steps
    drift = (model.rate - model.dividend - 0.5 * model.volatility**2) * dt
    diffusion = model.volatility * sqrt(dt)
    samples = np.empty((steps + 1, paths), dtype=float)
    samples[0] = spot
    for n in range(1, steps + 1):
        samples[n] = samples[n - 1] * np.exp(
            drift + diffusion * rng.standard_normal(paths)
        )
    return samples


def price_european_or_barrier(
    spot: float,
    model: BlackScholesModel,
    contract: OptionContract,
    paths: int = 100_000,
    steps: int = 1,
    seed: Optional[int] = 7,
) -> MonteCarloResult:
    if contract.exercise != "european":
        raise ValueError("Use least_squares_american for American exercise")
    if contract.barrier_kind is not None and steps < 2:
        raise ValueError("Barrier options need multiple monitoring dates")

    samples = simulate_paths(spot, model, contract.maturity, paths, steps, seed)
    payoffs = contract.payoff_at(samples[-1]).astype(float)
    if contract.barrier_kind == "up_and_out":
        # End-point checks miss crossings between dates. Conditional on two GBM
        # end points, log-price is a Brownian bridge, whose crossing probability
        # is known. Multiplying conditional survival probabilities prices a
        # continuously monitored barrier without requiring tiny time steps.
        log_barrier = np.log(contract.barrier)
        survival = np.ones(paths)
        variance_step = model.volatility**2 * contract.maturity / steps
        for n in range(steps):
            left = np.log(samples[n])
            right = np.log(samples[n + 1])
            below = (left < log_barrier) & (right < log_barrier)
            crossing_probability = np.ones(paths)
            crossing_probability[below] = np.exp(
                -2.0
                * (log_barrier - left[below])
                * (log_barrier - right[below])
                / variance_step
            )
            survival *= 1.0 - crossing_probability
        payoffs *= survival
    elif contract.barrier_kind == "down_and_out":
        log_barrier = np.log(contract.barrier)
        survival = np.ones(paths)
        variance_step = model.volatility**2 * contract.maturity / steps
        for n in range(steps):
            left = np.log(samples[n])
            right = np.log(samples[n + 1])
            above = (left > log_barrier) & (right > log_barrier)
            crossing_probability = np.ones(paths)
            crossing_probability[above] = np.exp(
                -2.0
                * (left[above] - log_barrier)
                * (right[above] - log_barrier)
                / variance_step
            )
            survival *= 1.0 - crossing_probability
        payoffs *= survival

    discounted = exp(-model.rate * contract.maturity) * payoffs
    return MonteCarloResult(
        float(np.mean(discounted)),
        float(np.std(discounted, ddof=1) / sqrt(paths)),
    )


def least_squares_american(
    spot: float,
    model: BlackScholesModel,
    contract: OptionContract,
    paths: int = 80_000,
    steps: int = 80,
    seed: Optional[int] = 7,
) -> MonteCarloResult:
    """Price an American call/put with Longstaff-Schwartz regression."""
    if contract.exercise != "american":
        raise ValueError("contract must use American exercise")
    samples = simulate_paths(spot, model, contract.maturity, paths, steps, seed)
    dt = contract.maturity / steps
    step_discount = exp(-model.rate * dt)
    cashflows = contract.payoff_at(samples[-1]).astype(float)
    exercise_step = np.full(paths, steps, dtype=int)

    for n in range(steps - 1, 0, -1):
        intrinsic = contract.payoff_at(samples[n]).astype(float)
        candidates = (exercise_step > n) & (intrinsic > 0.0)
        if np.count_nonzero(candidates) < 3:
            continue
        x = samples[n, candidates] / contract.strike
        remaining = exercise_step[candidates] - n
        future_value = cashflows[candidates] * np.power(step_discount, remaining)
        coefficients = np.polynomial.polynomial.polyfit(x, future_value, deg=2)
        continuation = np.polynomial.polynomial.polyval(x, coefficients)
        exercise = intrinsic[candidates] > continuation
        chosen = np.flatnonzero(candidates)[exercise]
        cashflows[chosen] = intrinsic[chosen]
        exercise_step[chosen] = n

    discounted = cashflows * np.power(step_discount, exercise_step)
    return MonteCarloResult(
        float(np.mean(discounted)),
        float(np.std(discounted, ddof=1) / sqrt(paths)),
    )
