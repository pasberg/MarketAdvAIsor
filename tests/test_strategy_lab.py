import unittest

import numpy as np
import pandas as pd

from pipeline import strategy_lab as lab


def frame(closes):
    c = pd.Series(closes, index=pd.bdate_range("2022-01-03", periods=len(closes)), dtype=float)
    return pd.DataFrame({"o": c, "h": c * 1.01, "l": c * 0.99, "c": c})


class Lab(unittest.TestCase):
    def test_no_look_ahead(self):
        # the position decided at day t earns the return of day t+1, never day t itself
        df = frame([100, 100, 110, 121, 121])
        rets, pos = lab.strategy_returns({"X": df}, lambda d: pd.Series([0, 1, 1, 0, 0], index=d.index, dtype=float), {}, cost_pct=0)
        self.assertEqual(list(rets["X"].round(4)), [0, 0, 0.1, 0.1, 0])

    def test_costs_on_position_changes(self):
        df = frame([100] * 6)
        rets, _ = lab.strategy_returns({"X": df}, lambda d: pd.Series([1, 1, 0, 1, 1, 1], index=d.index, dtype=float), {}, cost_pct=0.2)
        self.assertAlmostEqual(rets["X"].sum(), -0.001 * 3)  # enter, exit, re-enter: 0.1 % each side

    def test_hold_state_machine(self):
        idx = pd.RangeIndex(6)
        e = pd.Series([0, 1, 0, 0, 1, 0], index=idx).astype(bool)
        x = pd.Series([0, 0, 0, 1, 0, 0], index=idx).astype(bool)
        self.assertEqual(list(lab.hold(e, x)), [0, 1, 1, 0, 1, 1])

    def test_run_prefers_trend_following_in_a_crash(self):
        # up, then a long fall: a trend filter should avoid most of the fall, buy and hold should not
        rng = np.random.default_rng(1)
        up = np.cumprod(1 + rng.normal(0.001, 0.01, 400)) * 100
        down = up[-1] * np.cumprod(1 + rng.normal(-0.003, 0.01, 400))
        prices = {f"S{i}": frame(np.concatenate([up, down]) * (1 + i / 10)) for i in range(6)}
        res = lab.run(prices, cost_pct=0.1)
        fams = {f["id"]: f for f in res["families"]}
        self.assertLess(fams["buy_hold"]["best"]["test"]["cagr"], 0)
        self.assertGreater(fams["price_sma"]["best"]["test"]["cagr"], fams["buy_hold"]["best"]["test"]["cagr"])
        self.assertIn(res["champion"]["family"], lab.FAMILIES)
        self.assertEqual(len(res["champion"]["signals"]), 6)
        self.assertEqual(res["configs_tested"], sum(len(f["grid"]) for f in lab.FAMILIES.values()))

    def test_rotation_holds_the_strongest(self):
        n = 300
        strong = frame(100 * np.cumprod(np.full(n, 1.002)))
        weak = frame(100 * np.cumprod(np.full(n, 0.999)))
        prices = {"UP1": strong, "UP2": strong * 1.1, "DOWN": weak}
        pos = lab.xs_momentum(prices, lookback=126, top=2)
        self.assertEqual(list(pos.iloc[-1][["UP1", "UP2", "DOWN"]]), [1.0, 1.0, 0.0])  # the loser is never held
        rets, _ = lab.strategy_returns(prices, lab.xs_momentum, {"lookback": 126, "top": 2}, cost_pct=0, rotation=True)
        self.assertAlmostEqual(lab.portfolio(rets).iloc[-1], 0.002, places=6)  # half in each winner


if __name__ == "__main__":
    unittest.main()
