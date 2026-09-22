"""Simple Streamlit project for volatility surfaces and option pricing."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from analytics import black_scholes_price
from contracts import OptionContract
from finite_difference import solve_finite_difference
from model import BlackScholesModel
from monte_carlo import least_squares_american, price_european_or_barrier
from volatility_surface import (
    fit_surface_spline,
    generate_notebook_prices,
    make_smile_figure,
    make_surface_figure,
    make_term_figure,
    recover_implied_surface,
    reprice_surface,
)


st.set_page_config(page_title="Volatility Surface Project", layout="wide")


def make_strike_grid(center_strike, range_percent=30.0, number_of_strikes=13):
    """Create evenly spaced strikes below and above the selected strike."""
    lowest_strike = center_strike * (1.0 - range_percent / 100.0)
    highest_strike = center_strike * (1.0 + range_percent / 100.0)
    return np.linspace(lowest_strike, highest_strike, number_of_strikes)


def show_option_formula(option_family, option_side):
    """Display the main formula or payoff for the selected option."""
    if option_family == "European":
        if option_side == "Call":
            st.latex(r"C=S_0e^{-qT}N(d_1)-Ke^{-rT}N(d_2)")
        else:
            st.latex(r"P=Ke^{-rT}N(-d_2)-S_0e^{-qT}N(-d_1)")
        st.latex(
            r"d_1=\frac{\ln(S_0/K)+(r-q+\frac12\sigma^2)T}"
            r"{\sigma\sqrt{T}},\qquad d_2=d_1-\sigma\sqrt{T}"
        )
    elif option_family == "American":
        payoff = r"\max(S-K,0)" if option_side == "Call" else r"\max(K-S,0)"
        st.latex(r"\text{payoff}=" + payoff)
        st.latex(r"V(S,t)\geq \text{intrinsic payoff}(S)")
        st.write("An American option may be exercised before expiration.")
    elif option_family == "Digital":
        if option_side == "Call":
            st.latex(r"V_0=Qe^{-rT}N(d_2)")
        else:
            st.latex(r"V_0=Qe^{-rT}N(-d_2)")
        st.write("Q is the fixed cash payout.")
    else:
        st.latex(r"V(H,t)=0")
        st.write("The option becomes worthless when the stock touches barrier H.")


def make_contract(
    option_family,
    option_side,
    strike,
    maturity,
    cash_payout,
    barrier_kind,
    barrier,
):
    """Create the option selected on the page."""
    payoff = option_side.lower()
    exercise = "european"

    if option_family == "Digital":
        payoff = f"digital_{payoff}"
    elif option_family == "American":
        exercise = "american"

    return OptionContract(
        strike=strike,
        maturity=maturity,
        payoff=payoff,
        exercise=exercise,
        cash_payout=cash_payout,
        barrier_kind=barrier_kind,
        barrier=barrier,
    )


def run_pricers(
    spot,
    rate,
    dividend,
    volatility,
    contract,
    option_family,
    method,
    stock_steps,
    time_steps,
    paths,
    seed,
):
    """Run finite differences first, followed by Monte Carlo."""
    model = BlackScholesModel(
        rate=rate,
        dividend=dividend,
        volatility=volatility,
    )
    stock_maximum = max(4.0 * spot, 4.0 * contract.strike)
    scheme = method.lower().replace("-", "_")

    finite_difference = solve_finite_difference(
        spot=spot,
        model=model,
        contract=contract,
        stock_max=stock_maximum,
        space_steps=stock_steps,
        time_steps=time_steps,
        scheme=scheme,
    )

    if option_family == "American":
        monte_carlo = least_squares_american(
            spot=spot,
            model=model,
            contract=contract,
            paths=paths,
            steps=min(time_steps, 120),
            seed=seed,
        )
    else:
        simulation_steps = min(time_steps, 250) if option_family == "Barrier" else 1
        monte_carlo = price_european_or_barrier(
            spot=spot,
            model=model,
            contract=contract,
            paths=paths,
            steps=simulation_steps,
            seed=seed,
        )

    formula_price = None
    if option_family in {"European", "Digital"}:
        formula_price = black_scholes_price(spot, model, contract)

    return finite_difference, monte_carlo, formula_price


def make_price_curve(finite_difference, contract, spot):
    """Plot the finite-difference value today and payoff at expiration."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=finite_difference.stock_grid,
            y=finite_difference.surface[-1],
            name="Finite-difference value today",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=finite_difference.stock_grid,
            y=contract.payoff_at(finite_difference.stock_grid),
            name="Payoff at expiration",
            line={"dash": "dash"},
        )
    )
    figure.add_vline(x=spot, line_dash="dot", annotation_text="Spot")
    figure.add_vline(
        x=contract.strike,
        line_dash="dash",
        annotation_text="Strike",
    )
    figure.update_layout(
        title="Finite-Difference Option Value",
        xaxis_title="Stock price",
        yaxis_title="Option value ($)",
        height=520,
    )
    return figure


def select_display_grid(finite_difference, spot, strike):
    """Select the useful part of the finite-difference grid for plotting."""
    lower = 0.4 * min(spot, strike)
    upper = 2.0 * max(spot, strike)
    keep = (
        (finite_difference.stock_grid >= lower)
        & (finite_difference.stock_grid <= upper)
    )
    if np.count_nonzero(keep) < 4:
        keep = np.ones_like(finite_difference.stock_grid, dtype=bool)

    stock_grid = finite_difference.stock_grid[keep]
    value_grid = finite_difference.surface[:, keep]
    time_step = max(1, len(finite_difference.tau_grid) // 140)
    stock_step = max(1, len(stock_grid) // 140)
    return stock_grid, value_grid, time_step, stock_step


def make_value_surface(finite_difference, spot, strike):
    """Create the 3D finite-difference value surface."""
    stock_grid, value_grid, time_step, stock_step = select_display_grid(
        finite_difference, spot, strike
    )
    figure = go.Figure(
        go.Surface(
            x=stock_grid[::stock_step],
            y=finite_difference.tau_grid[::time_step],
            z=value_grid[::time_step, ::stock_step],
            colorscale="Plasma",
            colorbar={"title": "Option value ($)"},
        )
    )
    figure.update_layout(
        title="Finite-Difference Value Surface",
        height=720,
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
        scene={
            "xaxis_title": "Stock price",
            "yaxis_title": "Time remaining (years)",
            "zaxis_title": "Option value ($)",
        },
    )
    return figure


def make_value_heatmap(finite_difference, spot, strike):
    """Create a heatmap from the finite-difference value grid."""
    stock_grid, value_grid, time_step, stock_step = select_display_grid(
        finite_difference, spot, strike
    )
    figure = go.Figure(
        go.Heatmap(
            x=stock_grid[::stock_step],
            y=finite_difference.tau_grid[::time_step],
            z=value_grid[::time_step, ::stock_step],
            colorscale="Plasma",
            colorbar={"title": "Option value ($)"},
        )
    )
    figure.update_layout(
        title="Finite-Difference Value Heatmap",
        xaxis_title="Stock price",
        yaxis_title="Time remaining (years)",
        height=580,
    )
    return figure


def show_project_explanation():
    """Explain the project in simple language."""
    st.header("How the complete project works")
    st.markdown(
        "1. Choose the option framework: European, American, digital, or barrier.\n"
        "2. Choose whether it is a call or put.\n"
        "3. Enter the market inputs. The program automatically creates strikes "
        "30% below and 30% above your selected strike.\n"
        "4. The notebook formula creates example call prices and recovers their "
        "implied volatilities.\n"
        "5. The spline creates the volatility surface, smile, and term structure.\n"
        "6. Finite differences calculate the selected option on a stock/time grid.\n"
        "7. Monte Carlo simulates possible future stock prices.\n"
        "8. The site compares and displays the results."
    )
    st.info(
        "The volatility-surface prices are controlled teaching prices, not live "
        "exchange prices. This project is educational and is not trading advice."
    )


st.title("Volatility Surface and Option Pricing Project")
st.caption(
    "Implied-volatility surface · finite differences · Monte Carlo simulation"
)

# 1. The user chooses the option framework.
st.header("1. Choose the option framework")
option_family = st.selectbox(
    "Option framework",
    ["European", "American", "Digital", "Barrier"],
)

# 2. The user chooses call or put.
st.header("2. Choose call or put")
option_side = st.radio("Option side", ["Call", "Put"], horizontal=True)

# 3. Market inputs and the automatic strike range.
st.header("3. Enter the market and option inputs")
first_row = st.columns(4)
with first_row[0]:
    spot = st.number_input(
        "Spot price ($)", min_value=0.01, value=100.0, step=1.0
    )
with first_row[1]:
    strike = st.number_input(
        "Selected strike price ($)", min_value=0.01, value=100.0, step=1.0
    )
with first_row[2]:
    maturity = st.number_input(
        "Selected maturity (years)",
        min_value=0.10,
        max_value=2.00,
        value=1.00,
        step=0.05,
    )
with first_row[3]:
    rate_percent = st.number_input(
        "Risk-free rate (%)", value=4.0, step=0.25
    )

second_row = st.columns(2)
with second_row[0]:
    dividend_percent = st.number_input(
        "Dividend yield (%)", value=0.0, step=0.25
    )
with second_row[1]:
   st.metric(
    "Automatic strike range",
    f"${0.70 * spot:.2f} to ${1.30 * spot:.2f}",
)

rate = rate_percent / 100.0
dividend = dividend_percent / 100.0

# The volatility surface uses strikes around the stock's spot price.
strikes = make_strike_grid(
    spot,
    range_percent=30.0,
    number_of_strikes=13,
)

maturities = np.array(
    [0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
)

st.caption(
    "The surface uses 13 strikes from 30% below to 30% above the spot price: "
    + ", ".join(f"${value:.2f}" for value in strikes)
)

cash_payout = 1.0
barrier_kind = None
barrier = None

if option_family == "Digital":
    cash_payout = st.number_input(
        "Digital cash payout ($)", min_value=0.01, value=10.0
    )
elif option_family == "Barrier":
    barrier_columns = st.columns(2)
    with barrier_columns[0]:
        barrier_label = st.selectbox(
            "Barrier type", ["Up and out", "Down and out"]
        )
        barrier_kind = barrier_label.lower().replace(" ", "_")
    with barrier_columns[1]:
        default_barrier = 1.4 * spot if barrier_kind == "up_and_out" else 0.7 * spot
        barrier = st.number_input(
            "Barrier level ($)", min_value=0.01, value=float(default_barrier)
        )

with st.expander("Formula for the selected option"):
    show_option_formula(option_family, option_side)

# The original notebook surface calculation.
market_prices, original_volatility = generate_notebook_prices(
    S0=spot,
    r=rate,
    q=dividend,
    strikes=strikes,
    maturities=maturities,
    noise_std=0.0,
    seed=42,
)
implied_surface = recover_implied_surface(
    market_prices=market_prices,
    S0=spot,
    r=rate,
    q=dividend,
)
spline = fit_surface_spline(implied_surface, spot, smoothing=0.0)
repriced = reprice_surface(implied_surface, spot, rate, dividend)

selected_volatility = float(spline.ev(strike / spot, maturity))
selected_volatility = float(np.clip(selected_volatility, 0.01, 3.0))
volatility_error = float((implied_surface - original_volatility).abs().max().max())
price_error = float((repriced - market_prices).abs().max().max())

# 4. Finite-difference inputs.
st.header("4. Finite-difference settings")
finite_columns = st.columns(3)
with finite_columns[0]:
    method = st.selectbox(
        "Finite-difference method",
        ["Crank-Nicolson", "Implicit", "Explicit"],
    )
with finite_columns[1]:
    stock_steps = st.slider("Stock-price steps (S steps)", 40, 320, 160, 20)
with finite_columns[2]:
    time_steps = st.slider("Time steps (T steps)", 50, 2000, 400, 50)

# 5. Monte Carlo inputs.
st.header("5. Monte Carlo settings")
monte_carlo_columns = st.columns(2)
with monte_carlo_columns[0]:
    paths = st.slider("Number of simulated paths", 10000, 150000, 30000, 10000)
with monte_carlo_columns[1]:
    seed = st.number_input("Random seed", min_value=0, value=7, step=1)

run_calculations = st.button(
    "Run finite differences and Monte Carlo",
    type="primary",
)

contract = None
finite_difference = None
monte_carlo = None
formula_price = None
pricing_completed = False

if run_calculations:
    barrier_problem = None
    if barrier_kind == "up_and_out" and barrier <= spot:
        barrier_problem = "The up-and-out barrier must be above the spot price."
    if barrier_kind == "down_and_out" and barrier >= spot:
        barrier_problem = "The down-and-out barrier must be below the spot price."

    if barrier_problem:
        st.error(barrier_problem)
    else:
        contract = make_contract(
            option_family=option_family,
            option_side=option_side,
            strike=strike,
            maturity=maturity,
            cash_payout=cash_payout,
            barrier_kind=barrier_kind,
            barrier=barrier,
        )

        with st.spinner("Running finite differences, then Monte Carlo..."):
            finite_difference, monte_carlo, formula_price = run_pricers(
                spot=spot,
                rate=rate,
                dividend=dividend,
                volatility=selected_volatility,
                contract=contract,
                option_family=option_family,
                method=method,
                stock_steps=stock_steps,
                time_steps=time_steps,
                paths=paths,
                seed=seed,
            )
        pricing_completed = True

# 6 and 7. Display every result and plot.
st.header("6. Results and plots")

if pricing_completed:
    confidence_low, confidence_high = monte_carlo.confidence_interval_95
    result_columns = st.columns(4)
    result_columns[0].metric(
        "Surface volatility", f"{100.0 * selected_volatility:.2f}%"
    )
    result_columns[1].metric(
        "Finite-difference price", f"${finite_difference.price:.5f}"
    )
    result_columns[2].metric("Monte Carlo price", f"${monte_carlo.price:.5f}")
    result_columns[3].metric(
        "Direct formula",
        "Not available" if formula_price is None else f"${formula_price:.5f}",
    )
    st.caption(
        f"Monte Carlo 95% confidence interval: "
        f"${confidence_low:.5f} to ${confidence_high:.5f}"
    )

    if confidence_low <= finite_difference.price <= confidence_high:
        st.success("The finite-difference price is inside the Monte Carlo interval.")
    else:
        st.warning(
            "The methods differ by more than the current Monte Carlo sampling range. "
            "Increase paths or grid steps for a tighter comparison."
        )

    if method == "Explicit" and not finite_difference.stable_explicit_weights:
        st.warning("The explicit grid may be unstable. Increase the time steps.")
else:
    st.info("Choose the inputs, then click **Run finite differences and Monte Carlo**.")

(
    volatility_tab,
    smile_tab,
    term_tab,
    finite_tab,
    data_tab,
    explanation_tab,
) = st.tabs(
    [
        "Volatility surface",
        "Volatility smiles",
        "Term structure",
        "Finite-difference plots",
        "Tables and checks",
        "Project explanation",
    ]
)

with volatility_tab:
    st.plotly_chart(
        make_surface_figure(implied_surface, spot, spline, resolution=200),
        use_container_width=True,
    )

with smile_tab:
    st.plotly_chart(
        make_smile_figure(implied_surface, spot),
        use_container_width=True,
    )

with term_tab:
    st.plotly_chart(
        make_term_figure(implied_surface, spot),
        use_container_width=True,
    )

with finite_tab:
    if pricing_completed:
        curve_tab, value_surface_tab, heatmap_tab = st.tabs(
            ["Price curve", "3D value surface", "Heatmap"]
        )
        with curve_tab:
            st.plotly_chart(
                make_price_curve(finite_difference, contract, spot),
                use_container_width=True,
            )
        with value_surface_tab:
            st.plotly_chart(
                make_value_surface(finite_difference, spot, strike),
                use_container_width=True,
            )
        with heatmap_tab:
            st.plotly_chart(
                make_value_heatmap(finite_difference, spot, strike),
                use_container_width=True,
            )
    else:
        st.info("Run the pricing calculation to create the finite-difference plots.")

with data_tab:
    data_columns = st.columns(2)
    with data_columns[0]:
        st.subheader("Notebook-generated call prices ($)")
        st.dataframe(market_prices.round(6), width="stretch")
    with data_columns[1]:
        st.subheader("Recovered implied volatility (%)")
        st.dataframe((implied_surface * 100.0).round(6), width="stretch")

    check_columns = st.columns(3)
    check_columns[0].metric("Surface points", market_prices.size)
    check_columns[1].metric(
        "Volatility recovery",
        "Exact match" if volatility_error < 0.000001 else "Check result",
    )
    check_columns[2].metric(
        "Price check",
        "Exact match" if price_error < 0.000001 else "Check result",
    )

    st.download_button(
        "Download generated prices CSV",
        market_prices.to_csv().encode("utf-8"),
        file_name="notebook_call_prices.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download implied-volatility CSV",
        implied_surface.to_csv().encode("utf-8"),
        file_name="implied_volatility_surface.csv",
        mime="text/csv",
    )

with explanation_tab:
    show_project_explanation()

notebook_path = Path(__file__).with_name("Generate_volatility_surface.ipynb")
st.download_button(
    "Download the unchanged original notebook",
    notebook_path.read_bytes(),
    file_name=notebook_path.name,
    mime="application/x-ipynb+json",
)
