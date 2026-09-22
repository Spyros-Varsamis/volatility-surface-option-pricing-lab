"""Simple functions from the volatility-surface notebook, adapted for Streamlit."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm


def black_scholes_call(S0, K, T, r, sigma, q=0.0):
    """Black-Scholes European call with continuous dividend yield."""
    if T <= 0.0:
        return max(S0 - K, 0.0)

    sigma = max(float(sigma), 1e-12)
    sigma_sqrt_T = sigma * np.sqrt(T)
    d1 = (
        np.log(S0 / K)
        + (r - q + 0.5 * sigma * sigma) * T
    ) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T

    discounted_spot = S0 * np.exp(-q * T)
    discounted_strike = K * np.exp(-r * T)
    return float(
        discounted_spot * norm.cdf(d1)
        - discounted_strike * norm.cdf(d2)
    )


def sigma_true(S0, K, T):
    """The volatility formula used by the original notebook."""
    moneyness = K / S0
    base = 0.20
    smile_strength = 0.16 / np.sqrt(max(T, 1e-6))
    smile = smile_strength * (moneyness - 1.0) ** 2
    skew = -0.10 * (moneyness - 1.0)
    term = 0.03 * np.exp(-1.5 * T)
    volatility = base + smile + skew + term
    return float(np.clip(volatility, 0.05, 0.90))


def generate_notebook_prices(
    S0,
    r,
    q,
    strikes,
    maturities,
    noise_std=0.0,
    seed=42,
):
    """Create the controlled call prices used by the notebook experiment."""
    market_prices = pd.DataFrame(index=strikes, columns=maturities, dtype=float)
    true_volatility = pd.DataFrame(index=strikes, columns=maturities, dtype=float)

    random_generator = np.random.RandomState(seed)

    for K in strikes:
        for T in maturities:
            volatility = sigma_true(S0, float(K), float(T))
            call_price = black_scholes_call(
                S0,
                float(K),
                float(T),
                r,
                volatility,
                q=q,
            )
            noise = (
                random_generator.normal(0.0, noise_std)
                if noise_std > 0.0
                else 0.0
            )
            noisy_price = call_price * (1.0 + noise)
            lower_bound = max(
                S0 * np.exp(-q * T) - float(K) * np.exp(-r * T),
                0.0,
            )
            market_prices.loc[K, T] = max(noisy_price, lower_bound)
            true_volatility.loc[K, T] = volatility

    return market_prices, true_volatility


def implied_vol_bisection(
    market_price,
    S0,
    K,
    T,
    r,
    q=0.0,
    sigma_low=1e-8,
    sigma_high=2.0,
    tolerance=1e-12,
    maximum_iterations=250,
):
    """Recover implied volatility with a robust bisection search."""
    if T <= 0.0:
        return np.nan

    def difference(volatility):
        return (
            black_scholes_call(S0, K, T, r, volatility, q=q)
            - float(market_price)
        )

    low = float(sigma_low)
    high = float(sigma_high)
    difference_low = difference(low)
    difference_high = difference(high)

    attempts = 0
    while difference_low * difference_high > 0.0 and attempts < 25:
        high *= 1.5
        difference_high = difference(high)
        attempts += 1

    if difference_low * difference_high > 0.0:
        return np.nan

    midpoint = 0.5 * (low + high)
    for _ in range(maximum_iterations):
        midpoint = 0.5 * (low + high)
        difference_midpoint = difference(midpoint)

        if (
            abs(difference_midpoint) < tolerance
            or high - low < tolerance
        ):
            return float(midpoint)

        if difference_low * difference_midpoint <= 0.0:
            high = midpoint
            difference_high = difference_midpoint
        else:
            low = midpoint
            difference_low = difference_midpoint

    return float(midpoint)


def recover_implied_surface(market_prices, S0, r, q):
    """Invert every call price into an implied volatility."""
    strikes = market_prices.index.to_numpy(dtype=float)
    maturities = market_prices.columns.to_numpy(dtype=float)
    implied_surface = pd.DataFrame(
        index=strikes,
        columns=maturities,
        dtype=float,
    )

    for K in strikes:
        for T in maturities:
            implied_surface.loc[K, T] = implied_vol_bisection(
                market_price=float(market_prices.loc[K, T]),
                S0=S0,
                K=float(K),
                T=float(T),
                r=r,
                q=q,
            )

    return implied_surface


def fit_surface_spline(implied_surface, S0, smoothing=0.0):
    """Fit the notebook's bicubic spline in moneyness and maturity."""
    strikes = implied_surface.index.to_numpy(dtype=float)
    maturities = implied_surface.columns.to_numpy(dtype=float)
    moneyness = strikes / S0
    values = implied_surface.to_numpy(dtype=float)
    return RectBivariateSpline(
        moneyness,
        maturities,
        values,
        kx=3,
        ky=3,
        s=smoothing,
    )


def make_surface_figure(implied_surface, S0, spline, resolution=200):
    """Create the notebook's interactive 3D implied-volatility surface."""
    strikes = implied_surface.index.to_numpy(dtype=float)
    maturities = implied_surface.columns.to_numpy(dtype=float)

    strike_grid = np.linspace(strikes.max(), strikes.min(), resolution)
    time_grid = np.linspace(maturities.min(), maturities.max(), resolution)
    strike_mesh, time_mesh = np.meshgrid(strike_grid, time_grid)
    moneyness_mesh = strike_mesh / S0

    volatility_mesh = spline.ev(
        moneyness_mesh.ravel(),
        time_mesh.ravel(),
    ).reshape(strike_mesh.shape)
    volatility_mesh = np.clip(volatility_mesh, 0.01, 3.0)

    surface = go.Surface(
        x=time_mesh,
        y=strike_mesh,
        z=volatility_mesh * 100.0,
        opacity=0.92,
        colorbar={"title": "Implied vol (percent)", "len": 0.75},
        name="Spline surface",
    )

    point_times = []
    point_strikes = []
    point_volatilities = []
    for T in np.sort(maturities):
        for K in strikes:
            point_times.append(float(T))
            point_strikes.append(float(K))
            point_volatilities.append(
                float(implied_surface.loc[K, T]) * 100.0
            )

    points = go.Scatter3d(
        x=point_times,
        y=point_strikes,
        z=point_volatilities,
        mode="markers",
        marker={"size": 4},
        name="IV grid points",
    )

    figure = go.Figure(data=[surface, points])
    figure.update_layout(
        title={
            "text": "Implied Volatility Surface (x=Time, y=Strike, z=Vol)",
            "x": 0.5,
            "xanchor": "center",
            "pad": {"t": 2, "b": 2},
        },
        height=820,
        margin={"l": 0, "r": 0, "t": 35, "b": 0},
        scene={
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "xaxis": {
                "title": "Time to maturity (years)",
                "autorange": "reversed",
            },
            "yaxis": {"title": "Strike (K)"},
            "zaxis": {"title": "Implied volatility (percent)"},
            "camera": {"eye": {"x": 1.35, "y": 1.35, "z": 0.9}},
        },
    )
    return figure


def make_smile_figure(implied_surface, spot):
    """Create implied volatility versus strike for every maturity."""
    figure = go.Figure()
    for maturity in implied_surface.columns:
        figure.add_trace(
            go.Scatter(
                x=implied_surface.index.to_numpy(dtype=float),
                y=implied_surface[maturity].to_numpy(dtype=float) * 100.0,
                mode="lines+markers",
                name=f"Time to maturity = {float(maturity):.2f} years",
            )
        )

    figure.add_vline(x=spot, line_dash="dash")
    figure.update_layout(
        title="Volatility Smile (Implied volatility versus strike)",
        xaxis_title="Strike",
        yaxis_title="Implied volatility (percent)",
        height=480,
    )
    return figure


def make_term_figure(implied_surface, spot):
    """Create implied volatility versus maturity near the at-the-money strike."""
    strikes = implied_surface.index.to_numpy(dtype=float)
    at_money_index = int(np.argmin(np.abs(strikes - spot)))
    selected_strikes = strikes[
        max(0, at_money_index - 2) : min(len(strikes), at_money_index + 3)
    ]

    figure = go.Figure()
    for strike in selected_strikes:
        figure.add_trace(
            go.Scatter(
                x=implied_surface.columns.to_numpy(dtype=float),
                y=implied_surface.loc[strike].to_numpy(dtype=float) * 100.0,
                mode="lines+markers",
                name=f"Strike = {strike:.0f}",
            )
        )

    figure.update_layout(
        title="Volatility Term Structure (Implied volatility versus maturity)",
        xaxis_title="Time to maturity (years)",
        yaxis_title="Implied volatility (percent)",
        height=480,
    )
    return figure


def reprice_surface(implied_surface, S0, r, q):
    """Reprice calls with recovered IV and return the price table."""
    repriced = pd.DataFrame(
        index=implied_surface.index,
        columns=implied_surface.columns,
        dtype=float,
    )

    for K in implied_surface.index:
        for T in implied_surface.columns:
            volatility = float(implied_surface.loc[K, T])
            repriced.loc[K, T] = black_scholes_call(
                S0,
                float(K),
                float(T),
                r,
                volatility,
                q=q,
            )

    return repriced
