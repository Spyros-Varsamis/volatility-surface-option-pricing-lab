from math import exp
import unittest

import numpy as np

from analytics import black_scholes_price
from contracts import OptionContract
from finite_difference import solve_finite_difference, thomas_solve
from model import BlackScholesModel
from monte_carlo import least_squares_american, price_european_or_barrier


class PricingChecks(unittest.TestCase):
    model = BlackScholesModel(rate=0.04, dividend=0.01, volatility=0.25)

    def test_call_put_parity(self):
        call = OptionContract(strike=105.0, maturity=1.0, payoff="call")
        put = OptionContract(strike=105.0, maturity=1.0, payoff="put")
        difference = black_scholes_price(100.0, self.model, call) - black_scholes_price(
            100.0, self.model, put
        )
        expected = 100.0 * exp(-0.01) - 105.0 * exp(-0.04)
        self.assertAlmostEqual(difference, expected, places=12)

    def test_monte_carlo_contains_formula(self):
        option = OptionContract(strike=105.0, maturity=1.0, payoff="call")
        exact = black_scholes_price(100.0, self.model, option)
        mc = price_european_or_barrier(
            100.0, self.model, option, paths=180_000, seed=123
        )
        low, high = mc.confidence_interval_95
        self.assertTrue(low <= exact <= high)

    def test_crank_nicolson_matches_black_scholes(self):
        option = OptionContract(strike=105.0, maturity=1.0, payoff="call")
        exact = black_scholes_price(100.0, self.model, option)
        fd = solve_finite_difference(
            100.0, self.model, option, space_steps=240, time_steps=400
        )
        self.assertLess(abs(fd.price - exact), 0.02)

    def test_digital_matches_closed_form(self):
        option = OptionContract(
            strike=105.0,
            maturity=1.0,
            payoff="digital_call",
            cash_payout=10.0,
        )
        exact = black_scholes_price(100.0, self.model, option)
        fd = solve_finite_difference(
            100.0, self.model, option, space_steps=240, time_steps=400
        )
        self.assertLess(abs(fd.price - exact), 0.02)

    def test_barrier_is_cheaper_than_vanilla_with_same_random_numbers(self):
        vanilla = OptionContract(strike=100.0, maturity=1.0, payoff="call")
        barrier = OptionContract(
            strike=100.0,
            maturity=1.0,
            payoff="call",
            barrier_kind="up_and_out",
            barrier=140.0,
        )
        plain = price_european_or_barrier(
            100.0, self.model, vanilla, paths=40_000, steps=80, seed=8
        ).price
        knocked = price_european_or_barrier(
            100.0, self.model, barrier, paths=40_000, steps=80, seed=8
        ).price
        self.assertTrue(0.0 <= knocked <= plain)

    def test_american_put_is_at_least_european_put(self):
        european = OptionContract(strike=105.0, maturity=1.0, payoff="put")
        american = OptionContract(
            strike=105.0, maturity=1.0, payoff="put", exercise="american"
        )
        eu = solve_finite_difference(
            100.0, self.model, european, space_steps=140, time_steps=180
        ).price
        am = solve_finite_difference(
            100.0, self.model, american, space_steps=140, time_steps=180
        ).price
        self.assertGreaterEqual(am, eu)

    def test_american_pde_and_lsmc_are_reasonably_close(self):
        option = OptionContract(
            strike=105.0, maturity=1.0, payoff="put", exercise="american"
        )
        fd = solve_finite_difference(
            100.0, self.model, option, space_steps=140, time_steps=180
        )
        mc = least_squares_american(
            100.0, self.model, option, paths=60_000, steps=60, seed=16
        )
        self.assertLess(abs(fd.price - mc.price), 0.25)

    def test_thomas_algorithm(self):
        answer = thomas_solve(
            [-1.0, -1.0], [2.0, 2.0, 2.0], [-1.0, -1.0], [0.0, 2.0, 0.0]
        )
        np.testing.assert_allclose(answer, [1.0, 2.0, 1.0])


if __name__ == "__main__":
    unittest.main()
