import unittest

import numpy as np
import pandas as pd

from volatility_surface import (
    black_scholes_call,
    fit_surface_spline,
    generate_notebook_prices,
    recover_implied_surface,
    reprice_surface,
)


class VolatilitySurfaceChecks(unittest.TestCase):
    def setUp(self):
        self.spot = 100.0
        self.rate = 0.04
        self.dividend = 0.00
        self.strikes = np.array(
            [70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 130],
            dtype=float,
        )
        self.maturities = np.array(
            [0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00],
            dtype=float,
        )

    def make_test_prices(self, volatility=0.25):
        """Create controlled inputs so the numerical code can be checked."""
        prices = pd.DataFrame(
            index=self.strikes,
            columns=self.maturities,
            dtype=float,
        )
        for strike in self.strikes:
            for maturity in self.maturities:
                prices.loc[strike, maturity] = black_scholes_call(
                    self.spot,
                    strike,
                    maturity,
                    self.rate,
                    volatility,
                    q=self.dividend,
                )
        return prices

    def test_implied_volatility_recovers_clean_surface(self):
        expected_volatility = 0.25
        prices = self.make_test_prices(expected_volatility)
        implied = recover_implied_surface(
            prices,
            self.spot,
            self.rate,
            self.dividend,
        )
        maximum_error = float((implied - expected_volatility).abs().max().max())
        self.assertLess(maximum_error, 1e-8)

    def test_notebook_experiment_recovers_original_surface(self):
        prices, original_volatility = generate_notebook_prices(
            self.spot,
            self.rate,
            self.dividend,
            self.strikes,
            self.maturities,
        )
        implied = recover_implied_surface(
            prices,
            self.spot,
            self.rate,
            self.dividend,
        )
        maximum_error = float(
            (implied - original_volatility).abs().max().max()
        )
        self.assertLess(maximum_error, 1e-8)

    def test_recovered_volatility_reprices_market(self):
        prices = self.make_test_prices()
        implied = recover_implied_surface(
            prices,
            self.spot,
            self.rate,
            self.dividend,
        )
        repriced = reprice_surface(
            implied,
            self.spot,
            self.rate,
            self.dividend,
        )
        maximum_error = float((repriced - prices).abs().max().max())
        self.assertLess(maximum_error, 1e-8)

    def test_spline_can_be_evaluated_between_grid_points(self):
        prices = self.make_test_prices()
        implied = recover_implied_surface(
            prices,
            self.spot,
            self.rate,
            self.dividend,
        )
        spline = fit_surface_spline(implied, self.spot, smoothing=0.0)
        interpolated = float(spline.ev(1.025, 0.60))
        self.assertTrue(0.05 <= interpolated <= 0.90)


if __name__ == "__main__":
    unittest.main()
