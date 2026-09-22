"""Finite-difference solution of the Black-Scholes pricing PDE.

Using time remaining ``tau = T - t``, the PDE is

    V_tau = 0.5 sigma^2 S^2 V_SS + (r-q) S V_S - r V.

The theta scheme gives explicit (theta=0), implicit (theta=1), and
Crank-Nicolson (theta=0.5) stepping from the expiry payoff back to today.
"""

from dataclasses import dataclass
from math import exp
from typing import Literal, Optional

import numpy as np

from contracts import OptionContract
from model import BlackScholesModel

Scheme = Literal["explicit", "implicit", "crank_nicolson"]


@dataclass(frozen=True)
class FiniteDifferenceResult:
    price: float
    stock_grid: np.ndarray
    tau_grid: np.ndarray
    surface: np.ndarray
    stable_explicit_weights: bool
    psor_iterations: int = 0


def thomas_solve(lower, diagonal, upper, rhs):
    """Solve a tridiagonal linear system in O(n) work."""
    lower = np.asarray(lower, dtype=float)
    diagonal = np.asarray(diagonal, dtype=float).copy()
    upper = np.asarray(upper, dtype=float)
    answer = np.asarray(rhs, dtype=float).copy()
    n = diagonal.size
    for i in range(1, n):
        multiplier = lower[i - 1] / diagonal[i - 1]
        diagonal[i] -= multiplier * upper[i - 1]
        answer[i] -= multiplier * answer[i - 1]
    answer[-1] /= diagonal[-1]
    for i in range(n - 2, -1, -1):
        answer[i] = (answer[i] - upper[i] * answer[i + 1]) / diagonal[i]
    return answer


def projected_sor(
    lower,
    diagonal,
    upper,
    rhs,
    obstacle,
    initial,
    omega=1.2,
    tolerance=1e-9,
    max_iterations=10_000,
):
    """Solve the American-option system subject to value >= intrinsic value."""
    value = np.maximum(np.asarray(initial, dtype=float).copy(), obstacle)
    for iteration in range(1, max_iterations + 1):
        largest_change = 0.0
        for i in range(value.size):
            left = lower[i - 1] * value[i - 1] if i else 0.0
            right = upper[i] * value[i + 1] if i < value.size - 1 else 0.0
            raw = (rhs[i] - left - right) / diagonal[i]
            updated = max(value[i] + omega * (raw - value[i]), obstacle[i])
            largest_change = max(largest_change, abs(updated - value[i]))
            value[i] = updated
        if largest_change < tolerance:
            return value, iteration
    raise RuntimeError("PSOR did not converge")


def _boundaries(
    stock_min: float,
    stock_max: float,
    tau: float,
    model: BlackScholesModel,
    contract: OptionContract,
):
    stock_discount = exp(-model.dividend * tau)
    cash_discount = exp(-model.rate * tau)
    if contract.exercise == "american":
        lower = float(contract.payoff_at(stock_min))
        upper = float(contract.payoff_at(stock_max))
    elif contract.payoff == "call":
        lower = 0.0
        upper = max(
            stock_max * stock_discount - contract.strike * cash_discount, 0.0
        )
    elif contract.payoff == "put":
        lower = max(
            contract.strike * cash_discount - stock_min * stock_discount, 0.0
        )
        upper = 0.0
    elif contract.payoff == "digital_call":
        lower, upper = 0.0, contract.cash_payout * cash_discount
    else:
        lower, upper = contract.cash_payout * cash_discount, 0.0

    if contract.barrier_kind == "down_and_out":
        lower = 0.0
    elif contract.barrier_kind == "up_and_out":
        upper = 0.0
    return lower, upper


def solve_finite_difference(
    spot: float,
    model: BlackScholesModel,
    contract: OptionContract,
    stock_min: float = 0.0,
    stock_max: Optional[float] = None,
    space_steps: int = 240,
    time_steps: int = 500,
    scheme: Scheme = "crank_nicolson",
) -> FiniteDifferenceResult:
    if spot <= 0:
        raise ValueError("spot must be positive")
    if space_steps < 3 or time_steps < 1:
        raise ValueError("grid is too small")
    if scheme not in {"explicit", "implicit", "crank_nicolson"}:
        raise ValueError("unknown scheme")

    if stock_max is None:
        stock_max = max(4.0 * spot, 4.0 * contract.strike)
    if contract.barrier_kind == "up_and_out":
        stock_max = float(contract.barrier)
    elif contract.barrier_kind == "down_and_out":
        stock_min = float(contract.barrier)
    if not stock_min < spot < stock_max:
        raise ValueError("spot must be inside the live pricing domain")

    theta = {"explicit": 0.0, "implicit": 1.0, "crank_nicolson": 0.5}[scheme]
    stock_grid = np.linspace(stock_min, stock_max, space_steps + 1)
    tau_grid = np.linspace(0.0, contract.maturity, time_steps + 1)
    ds = (stock_max - stock_min) / space_steps
    dt = contract.maturity / time_steps
    interior_stock = stock_grid[1:-1]

    diffusion = 0.5 * model.volatility**2 * interior_stock**2 / ds**2
    drift = 0.5 * (model.rate - model.dividend) * interior_stock / ds
    a = diffusion - drift
    b = -2.0 * diffusion - model.rate
    c = diffusion + drift
    stable = bool(
        np.all(dt * a >= 0.0)
        and np.all(1.0 + dt * b >= 0.0)
        and np.all(dt * c >= 0.0)
    )

    surface = np.empty((time_steps + 1, space_steps + 1), dtype=float)
    surface[0] = contract.payoff_at(stock_grid)
    if contract.payoff.startswith("digital"):
        at_strike = np.isclose(stock_grid, contract.strike, atol=ds * 1e-10)
        surface[0, at_strike] = 0.5 * contract.cash_payout
    if contract.barrier_kind is not None:
        edge = 0 if contract.barrier_kind == "down_and_out" else -1
        surface[0, edge] = 0.0

    lower = -theta * dt * a[1:]
    diagonal = 1.0 - theta * dt * b
    upper = -theta * dt * c[:-1]
    total_psor_iterations = 0

    for n in range(time_steps):
        old = surface[n]
        old_inner = old[1:-1]
        rhs = old_inner + (1.0 - theta) * dt * (
            a * old[:-2] + b * old_inner + c * old[2:]
        )
        left_value, right_value = _boundaries(
            stock_min, stock_max, tau_grid[n + 1], model, contract
        )
        if theta:
            rhs[0] += theta * dt * a[0] * left_value
            rhs[-1] += theta * dt * c[-1] * right_value

        if theta == 0.0:
            new_inner = rhs
            if contract.exercise == "american":
                new_inner = np.maximum(
                    new_inner, contract.payoff_at(interior_stock)
                )
        elif contract.exercise == "american":
            new_inner, iterations = projected_sor(
                lower,
                diagonal,
                upper,
                rhs,
                contract.payoff_at(interior_stock),
                old_inner,
            )
            total_psor_iterations += iterations
        else:
            new_inner = thomas_solve(lower, diagonal, upper, rhs)

        surface[n + 1, 0] = left_value
        surface[n + 1, -1] = right_value
        surface[n + 1, 1:-1] = new_inner

    return FiniteDifferenceResult(
        price=float(np.interp(spot, stock_grid, surface[-1])),
        stock_grid=stock_grid,
        tau_grid=tau_grid,
        surface=surface,
        stable_explicit_weights=stable,
        psor_iterations=total_psor_iterations,
    )
