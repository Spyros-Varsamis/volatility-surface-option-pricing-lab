# Volatility Surface and Option Pricing Site


## Project files

```text
volatility_surface_site/
├── Generate_volatility_surface.ipynb  # unchanged original notebook
├── dashboard.py                       # Streamlit website
├── volatility_surface.py              # notebook calculations and charts
├── contracts.py                       # option definitions and payoffs
├── model.py                           # Black–Scholes market model
├── analytics.py                       # vanilla and digital closed forms
├── finite_difference.py               # explicit/implicit/CN, Thomas, PSOR
├── monte_carlo.py                     # GBM, barriers, Longstaff–Schwartz
├── requirements.txt
└── tests/
```

## Install

Open this folder in VS Code and run:

```bash
python3 -m pip install -r requirements.txt
```

## Run the tests

```bash
python3 -m unittest discover -s tests -v
```

## Launch the website

```bash
python3 -m streamlit run dashboard.py
```

Open <http://localhost:8501> if the browser does not open automatically.

## What the single page does

1. Choose a European, American, digital, or barrier option.
2. Choose a call or put.
3. Enter spot, strike, maturity, interest rate, and dividend yield.
4. The site automatically creates 13 strikes from 30% below to 30% above the
   selected strike.
5. The notebook workflow creates prices, recovers implied volatility, and draws
   the 3D surface, smiles, and term structure.
6. Choose the finite-difference method, stock steps, and time steps.
7. Choose the Monte Carlo path count and random seed.
8. Run the calculation and compare finite differences, Monte Carlo, and the
   direct formula when one exists.

Finite-difference outputs include a price curve, 3D value surface, and heatmap.

## Surface settings in plain English

- **Dividend yield:** the percentage of the stock price paid as dividends each
  year. Enter `2` for 2%.
- **Price check:** recalculates every option price using its recovered
  volatility. An exact match means the method is working correctly.

The generated prices are a controlled teaching experiment, just like the
original notebook. They are not live market prices.

This project is educational and is not trading or investment advice.
