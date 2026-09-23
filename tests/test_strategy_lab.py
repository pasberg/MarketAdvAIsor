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

    def test_long_short_profits_from_a_fall_and_pays_short_cost(self):
        rng = np.random.default_rng(2)
        up = np.cumprod(1 + rng.normal(0.001, 0.01, 400)) * 100
        down = up[-1] * np.cumprod(1 + rng.normal(-0.003, 0.01, 400))
        prices = {f"S{i}": frame(np.concatenate([up, down]) * (1 + i / 10)) for i in range(4)}
        ls, pos = lab.strategy_returns(prices, lab.price_sma_ls, {"n": 100}, cost_pct=0.1)
        lo, _ = lab.strategy_returns(prices, lab.price_sma, {"n": 100}, cost_pct=0.1)
        fall = slice(pos.index[500], pos.index[-1])
        self.assertGreater(lab.portfolio(ls).loc[fall].sum(), lab.portfolio(lo).loc[fall].sum())
        self.assertIn(-1.0, set(pos.iloc[-1]))
        # a flat market: holding a short costs the yearly short fee
        flat = {"X": frame([100.0] * 300)}
        r, _ = lab.strategy_returns(flat, lambda d: pd.Series(-1.0, index=d.index), {}, cost_pct=0)
        self.assertAlmostEqual(r["X"].iloc[1:].sum(), -lab.SHORT_COST_PCT_YEAR / 100 / lab.YEAR * 299)

    def test_hold_ls(self):
        idx = pd.RangeIndex(6)
        b = lambda v: pd.Series(v, index=idx).astype(bool)
        pos = lab.hold_ls(b([1, 0, 0, 0, 0, 0]), b([0, 0, 1, 0, 0, 0]), b([0, 0, 0, 1, 0, 0]), b([0, 0, 0, 0, 0, 1]))
        self.assertEqual(list(pos), [1, 1, 0, -1, -1, 0])

    def test_dual_momentum_picks_the_best_asset_or_cash(self):
        n = 300
        up = lambda k: frame(100 * np.cumprod(np.full(n, 1 + k)))
        prices = {"OMXS30": up(0.001), "SPX": up(0.002), "NDX100": up(-0.001), "GOLD": up(0.0005), "VOLV-B": up(0.01)}
        pos = lab.dual_momentum(prices, lookback=126)
        self.assertNotIn("VOLV-B", pos.columns)  # only the index/gold universe
        self.assertEqual(pos.iloc[-1].idxmax(), "SPX")
        self.assertEqual(pos.iloc[-1].sum(), 1.0)
        # everything falling: absolute momentum keeps the strategy in cash
        down = {k: up(-0.002) for k in lab.DUAL_MOMENTUM_UNIVERSE}
        self.assertEqual(lab.dual_momentum(down, lookback=126).iloc[-1].sum(), 0.0)

    def test_dual_momentum_uses_no_future_data(self):
        # changing prices after a date must not change the positions up to that date
        rng = np.random.default_rng(3)
        base = {k: frame(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 400))) for k in lab.DUAL_MOMENTUM_UNIVERSE}
        cut = base["SPX"].index[300]
        changed = {k: v.copy() for k, v in base.items()}
        for v in changed.values():
            v.loc[v.index > cut] *= 3.0
        a = lab.dual_momentum(base, lookback=126).loc[:cut]
        b = lab.dual_momentum(changed, lookback=126).loc[:cut]
        pd.testing.assert_frame_equal(a, b)

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
