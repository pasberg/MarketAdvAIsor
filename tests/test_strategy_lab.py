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

    def test_weekly_run_uses_52_bars_per_year(self):
        idx = pd.date_range("2016-01-04", periods=520, freq="W-MON")
        mk = lambda g: pd.DataFrame({"o": g, "h": g * 1.01, "l": g * 0.99, "c": g}, index=idx)
        prices = {f"S{i}": mk(100 * np.cumprod(np.full(520, 1.002 + i / 10000))) for i in range(5)}
        bh = lab.metrics(lab.portfolio(lab.strategy_returns(prices, lab.buy_hold, {}, 0, ppy=lab.WEEKS)[0]).iloc[1:], ppy=lab.WEEKS)
        self.assertAlmostEqual(bh["cagr"], ((1.0022 ** 52) - 1) * 100, delta=0.2)
        res = lab.run(prices, cost_pct=0.1, families=lab.WEEKLY_FAMILIES, ppy=lab.WEEKS)
        self.assertEqual(res["bars"], "week")
        self.assertEqual(res["configs_tested"], sum(len(f["grid"]) for f in lab.WEEKLY_FAMILIES.values()))
        self.assertEqual({f["id"] for f in res["families"]}, set(lab.WEEKLY_FAMILIES))
        # the first re-selection needs two years of weekly history
        self.assertEqual(res["period"]["wf_start"], str(idx[104].date()))

    def test_load_prices_skips_currencies(self):
        import json, tempfile
        from pathlib import Path
        ser = {"t": [86400 * i for i in range(80)], "o": [1.0] * 80, "h": [1.0] * 80, "l": [1.0] * 80, "c": [1.0] * 80}
        d = {"symbols": {"VOLV-B": {"kind": "stock", "series": {"day": ser}}, "EURUSD": {"kind": "fx", "series": {"day": ser}}}}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "daily.json").write_text(json.dumps(d))
            self.assertEqual(set(lab.load_prices(Path(tmp))), {"VOLV-B"})

    def test_weekly_grids_stay_small(self):
        for key, fam in lab.WEEKLY_FAMILIES.items():
            self.assertLessEqual(len(fam["grid"]), 4, key)

    def test_portfolio_vol_target_scales_to_target_and_caps(self):
        idx = pd.bdate_range("2020-01-01", periods=400)
        rng = np.random.default_rng(4)
        invested = pd.DataFrame({"X": 1.0}, index=idx)
        wild = pd.Series(rng.normal(0.0005, 0.02, 400), index=idx)  # ~32 % yearly volatility
        calm = pd.Series(rng.normal(0.0005, 0.003, 400), index=idx)  # ~5 % yearly volatility
        _, s_wild = lab.portfolio_vol_target(wild, invested, cost_pct=0)
        _, s_calm = lab.portfolio_vol_target(calm, invested, cost_pct=0)
        self.assertAlmostEqual(s_wild.iloc[-1], 0.15 / 0.32, delta=0.12)
        self.assertEqual(s_calm.iloc[-1], lab.VOL_MAX_LEVERAGE)
        scaled, _ = lab.portfolio_vol_target(wild, invested, cost_pct=0)
        self.assertLess(scaled.iloc[100:].std() * np.sqrt(lab.YEAR), 0.20)

    def test_portfolio_vol_target_uses_no_future_data(self):
        idx = pd.bdate_range("2020-01-01", periods=400)
        rng = np.random.default_rng(5)
        r = pd.Series(rng.normal(0.0005, 0.01, 400), index=idx)
        later = r.copy()
        later.iloc[301:] *= 5  # a wild future must not change today's scale
        inv = pd.DataFrame({"X": 1.0}, index=idx)
        a, _ = lab.portfolio_vol_target(r, inv)
        b, _ = lab.portfolio_vol_target(later, inv)
        pd.testing.assert_series_equal(a.iloc[:301], b.iloc[:301])

    def test_leverage_is_charged(self):
        idx = pd.bdate_range("2020-01-01", periods=300)
        flat = pd.Series(0.0, index=idx)
        flat.iloc[::2] = 0.0001  # tiny volatility -> maximum scale
        out, scale = lab.portfolio_vol_target(flat, pd.DataFrame({"X": 1.0}, index=idx), cost_pct=0)
        extra = (scale * 1.0 - 1).clip(lower=0).iloc[1:]
        self.assertTrue((extra > 0).any())
        self.assertLess(out.sum(), (flat * scale).sum())  # financing is paid on the part above 100 %

    def test_combination_holds_an_equal_mix_of_the_best(self):
        idx = pd.bdate_range("2020-01-01", periods=600)
        rng = np.random.default_rng(6)
        mk = lambda mu: pd.Series(rng.normal(mu, 0.01, len(idx)), index=idx)
        per_family = {k: [({}, mk(mu), None)] for k, mu in (("a", .002), ("b", .0015), ("c", .001), ("d", -.001), ("e", -.002))}
        r, picks = lab.walk_forward_combo(per_family, idx, start=300, step=100, ppy=lab.YEAR, k=3)
        self.assertEqual({m["family"] for m in picks[-1]["members"]}, {"a", "b", "c"})
        expected = pd.concat([per_family[k][0][1] for k in "abc"], axis=1).mean(axis=1).iloc[500:600]
        pd.testing.assert_series_equal(r.iloc[-100:], expected, check_names=False)

    def test_run_reports_combination(self):
        rng = np.random.default_rng(7)
        prices = {f"S{i}": frame(100 * np.cumprod(1 + rng.normal(0.0006, 0.012, 700))) for i in range(5)}
        res = lab.run(prices, cost_pct=0.1)
        combo = res["combination"]
        self.assertEqual(len(combo["members_now"]), lab.COMBO_SIZE)
        self.assertEqual(len(combo["train_members"]), lab.COMBO_SIZE)
        self.assertIn("sharpe", combo["wf"])
        self.assertTrue(all(0 <= s["weight"] <= lab.VOL_MAX_LEVERAGE for s in combo["signals"]))


class Members(unittest.TestCase):
    """The universe without hindsight: current and former OMXS30 members."""

    def fake_yf(self, closes: dict[str, list[float]]):
        idx = pd.bdate_range("2020-01-01", periods=max(len(v) for v in closes.values()))

        class YF:
            calls = 0

            @staticmethod
            def download(tickers, **kw):
                YF.calls += 1
                frames = {}
                for t in tickers:
                    c = pd.Series(closes[t], index=idx[:len(closes[t])], dtype=float)
                    frames[t] = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c}).reindex(idx)
                return pd.concat(frames, axis=1)
        return YF

    def test_member_list_file(self):
        rows = lab.load_member_list()
        self.assertGreaterEqual(len(rows), 40)
        self.assertEqual({r["status"] for r in rows}, {"nuvarande", "tidigare", "avnoterad"})
        self.assertTrue(all(r["yahoo"] == "" for r in rows if r["status"] == "avnoterad"))

    def test_prices_are_cached_for_a_day(self):
        import tempfile
        from datetime import datetime, timedelta, timezone
        from pathlib import Path
        rows = [{"sym": "A", "name": "A", "yahoo": "A.ST", "status": "nuvarande"},
                {"sym": "B", "name": "B", "yahoo": "B.ST", "status": "tidigare"},
                {"sym": "C", "name": "C", "yahoo": "", "status": "avnoterad"}]
        yf = self.fake_yf({"A.ST": list(np.linspace(100, 150, 300)), "B.ST": list(np.linspace(100, 20, 150))})
        now = datetime(2026, 9, 28, 6, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            p1 = lab.member_prices(Path(tmp), rows, yf, now)
            p2 = lab.member_prices(Path(tmp), rows, yf, now + timedelta(hours=2))
            self.assertEqual(yf.calls, 1)  # the second call used the cache
            self.assertEqual(set(p1), {"A", "B"})
            self.assertEqual(len(p2["B"]), 150)  # a former member keeps its shorter history
            lab.member_prices(Path(tmp), rows, yf, now + timedelta(hours=25))
            self.assertEqual(yf.calls, 2)

    def test_a_member_that_disappears_is_counted_until_it_does(self):
        # the loser falls 80 % and then has no more prices: buy and hold must include the fall,
        # and the returns before it disappears must not depend on later data of the others
        rng = np.random.default_rng(8)
        up = frame(100 * np.cumprod(1 + rng.normal(0.001, 0.01, 400)))
        loser = frame(np.linspace(100, 20, 200))
        full = {"UP": up, "LOSER": loser}
        r_full = lab.portfolio(lab.strategy_returns(full, lab.buy_hold, {}, 0)[0])
        self.assertLess(r_full.iloc[:200].sum(), lab.portfolio(lab.strategy_returns({"UP": up}, lab.buy_hold, {}, 0)[0]).iloc[:200].sum())
        cut = {"UP": up.iloc[:250], "LOSER": loser}
        r_cut = lab.portfolio(lab.strategy_returns(cut, lab.buy_hold, {}, 0)[0])
        pd.testing.assert_series_equal(r_full.iloc[:250], r_cut.iloc[:250])

    def test_run_members_reports_who_is_missing(self):
        rng = np.random.default_rng(9)
        prices = {s: frame(100 * np.cumprod(1 + rng.normal(0.0005, 0.012, 700))) for s in ("A", "B", "C", "D", "E")}
        rows = [{"sym": s, "name": s, "yahoo": s, "status": "tidigare" if s == "E" else "nuvarande"} for s in prices] + \
               [{"sym": "X", "name": "Swedish Match", "yahoo": "", "status": "avnoterad"}]
        res = lab.run_members(prices, rows)
        self.assertEqual(res["members"]["former"], ["E"])
        self.assertEqual(res["members"]["missing"], [{"sym": "X", "name": "Swedish Match", "status": "avnoterad"}])
        self.assertIn("champion", res)


if __name__ == "__main__":
    unittest.main()
