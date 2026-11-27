"""Tests for the virtual portfolio engine (pipeline/portfolio.py) on synthetic prices."""
import json
import math
import random
import sys
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import portfolio as pf  # noqa: E402

CFG = json.loads(pf.CONFIG.read_text(encoding="utf-8"))


def weekdays(n, start=date(2024, 1, 2)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def walk(days, seed, drift=0.0, vol=0.015, p0=100.0):
    rnd = random.Random(seed)
    bars, c = {}, p0
    for d in days:
        o = c * (1 + rnd.gauss(0, vol / 3))
        c = o * (1 + drift + rnd.gauss(0, vol))
        h, l = max(o, c) * (1 + abs(rnd.gauss(0, vol / 2))), min(o, c) * (1 - abs(rnd.gauss(0, vol / 2)))
        bars[d] = [o, h, l, c, c]
    return bars


def market(n=400, n_stocks=30, seed=1):
    days = weekdays(n)
    universe, bars, fund = {}, {}, {}
    for k in range(n_stocks):
        s = f"S{k:02d}"
        universe[s] = {"name": s, "yahoo": s, "kind": "stock", "tags": "ndx", "avanza_name": ""}
        bars[s] = walk(days, seed * 100 + k, drift=(k - n_stocks / 2) * 0.0001)
        # quarterly reports after the close with a surprise of alternating sign
        evs = []
        for q in range(60, n, 63):
            ts = datetime.strptime(days[q], "%Y-%m-%d").replace(hour=21, tzinfo=timezone.utc)
            evs.append({"ts": ts.isoformat(), "est": 1.0, "eps": 1.1, "surprise": 10.0 if k % 2 else -10.0})
        for e, q in zip(evs, range(len(evs))):
            e["eps"] = 1.0 + 0.05 * q * (1 if k % 2 else -1)  # odd stocks grow, even shrink
        ana = [[days[q], "up" if k % 2 else "down", 110.0, 100.0] for q in range(100, n, 40)]
        fund[s] = {"events": evs, "info": {"marketCap": 5e11, "recommendationMean": 2.0}, "analysts": ana}
    universe["IDX"] = {"name": "Index", "yahoo": "^IDX", "kind": "index", "tags": "index", "avanza_name": "IDX"}
    bars["IDX"] = walk(days, seed + 7, drift=0.0008, vol=0.01)
    universe["FX1"] = {"name": "FX", "yahoo": "FX=X", "kind": "fx", "tags": "fx", "avanza_name": "FX"}
    bars["FX1"] = walk(days, seed + 8, drift=-0.0005, vol=0.005, p0=1.1)
    bars["QQQ"] = walk(days, seed + 9, drift=0.0005, vol=0.012)
    bars["DIA"] = walk(days, seed + 10, drift=0.0003, vol=0.01)
    bars["USDSEK"] = walk(days, seed + 11, vol=0.004, p0=10.5)
    return {"bars": bars, "errors": {}}, universe, fund, days


class PortfolioEngine(unittest.TestCase):
    def run_all(self, **kw):
        prices, universe, fund, days = market(**kw)
        m = pf.Market(prices, universe, fund, CFG)
        st = pf.new_state(CFG, days[250])
        st["nav"] = []
        pf.step(st, m, 250, CFG, first=True)
        for i in range(251, len(days)):
            pf.step(st, m, i, CFG)
            # the books balance: NAV = index fund + positions
            self.assertAlmostEqual(st["nav"][-1]["nav"], pf.nav_of(st, m, i), delta=1)
        return st, m

    def test_limits(self):
        st, m = self.run_all()
        lim = CFG["limits"]
        self.assertGreater(len(st["trades"]), 10)
        for day in st["log"]:
            self.assertLessEqual(len(day["fills"]), lim["max_orders_per_day"])
            self.assertLessEqual(len(day["decisions"]), lim["max_orders_per_day"])
        for x in st["nav"]:
            self.assertLessEqual(x["n"], lim["max_positions"])
        for t in st["trades"]:
            # stake at most 5 % of the portfolio when opened (the portfolio is ~1 MSEK here)
            self.assertLessEqual(t["N"], lim["max_position_pct"] / 100 * max(x["nav"] for x in st["nav"]) + 1)
            # held about two weeks at most (the time exit is decided at 10 days and filled the next open)
            self.assertLessEqual(t["days"], lim["hold_days"] + 2)
        self.assertGreater(st["sleeve"], 0)

    def test_short_profits_when_price_falls(self):
        p = {"side": -1, "lev": 2.0, "N": 1000.0, "p0": 100.0, "fx0": 1.0, "kind": "index"}
        self.assertAlmostEqual(pf.pos_value(p, 95.0, 1.0), 1100.0)
        self.assertEqual(pf.pos_value(p, 160.0, 1.0), 0.0)  # knocked out: never below zero
        s = {"side": 1, "lev": 1.0, "N": 1000.0, "p0": 100.0, "fx0": 10.0, "kind": "stock"}
        self.assertAlmostEqual(pf.pos_value(s, 110.0, 11.0), 1210.0)  # price and dollar both count

    def test_no_lookahead(self):
        """Scores on day i do not change when later days are removed from the data."""
        prices, universe, fund, days = market(n=320)
        full = pf.Market(prices, universe, fund, CFG).scores(300)
        cut = {"bars": {s: {d: v for d, v in b.items() if d <= days[300]} for s, b in prices["bars"].items()}, "errors": {}}
        part = pf.Market(cut, universe, fund, CFG).scores(300)
        self.assertEqual(full, part)

    def test_earnings_component(self):
        prices, universe, fund, days = market(n=320)
        m = pf.Market(prices, universe, fund, CFG)
        # report after the close on day 249 -> reacts on day 250; positive surprise for odd stocks
        e, ev = m.earnings_comp("S01", 252)
        self.assertIsNotNone(ev)
        self.assertEqual(ev["days_since"], 2)
        self.assertEqual(m.earnings_comp("S01", 270)[0], 0.0)  # outside the window
        self.assertEqual(m.next_earnings("S01", 300), 12)

    def test_fundamentals_are_dated(self):
        prices, universe, fund, days = market(n=700)
        m = pf.Market(prices, universe, fund, CFG)
        f = m.fa_comps("S01", 650)
        self.assertGreater(f["growth"], 0)
        self.assertGreater(m.fa_comps("S01", 650)["eps_g_pct"], 0)
        self.assertLess(m.fa_comps("S02", 650)["growth"], 0)
        # an analyst upgrade on day 660 is not known on day 650
        self.assertEqual(m.fa_comps("S01", 659)["ups"], 0)  # 620 is outside the 45-day window
        self.assertEqual(m.fa_comps("S01", 661)["ups"], 1)
        self.assertIn("value", m.scores(650)["S01"]["comp"])

    def test_risk_reward(self):
        prices, universe, fund, days = market(n=700)
        for f in fund.values():  # three analysts set a 300 target in the last 90 days
            f["analysts"] += [[days[640], "main", 300.0, 290.0], [days[645], "main", 300.0, 290.0], [days[648], "main", 300.0, 290.0]]
        m = pf.Market(prices, universe, fund, CFG)
        r = m.rr("S01", 650, 1)
        c, atr = m.s["S01"].c[650], m.s["S01"].atr[650]
        self.assertAlmostEqual(r["rr"], (r["ref"] - c) / (CFG["stops"]["atr_mult"] * atr), places=1)
        self.assertIsNone(m.rr("S01", 300, 1))  # too few targets then
        hi = pf.Market(prices, universe, fund, pf.merged(CFG, {"signals": {"rr_source": "high"}})).rr("S01", 650, 1)
        self.assertGreaterEqual(hi["rr"], 0)
        # a minimum R/R only lets trades through that meet it
        cfg = pf.merged(CFG, {"signals": {"min_rr": 1.5}})
        st = pf.new_state(cfg, days[600])
        st["nav"] = []
        m2 = pf.Market(prices, universe, fund, cfg)
        pf.step(st, m2, 600, cfg, first=True)
        for i in range(601, 700):
            pf.step(st, m2, i, cfg)
        for t in st["trades"]:
            self.assertGreaterEqual(t["rr"], 1.5)

    def test_profit_target_exit(self):
        cfg = pf.merged(CFG, {"stops": {"target_r": 1}})
        prices, universe, fund, days = market(n=500)
        m = pf.Market(prices, universe, fund, cfg)
        st = pf.new_state(cfg, days[250])
        st["nav"] = []
        pf.step(st, m, 250, cfg, first=True)
        for i in range(251, 500):
            pf.step(st, m, i, cfg)
        self.assertTrue(any(t["why_out"].startswith("Mål nått") for t in st["trades"]))

    def test_tune_moves_one_step_with_reason(self):
        prices, universe, fund, days = market(n=600)
        now = datetime.strptime(days[-1], "%Y-%m-%d").replace(hour=22, tzinfo=timezone.utc)
        st = pf.new_state(CFG, days[-1])
        msgs = pf.tune(st, CFG, universe, prices, fund, now)
        t = st["tuning"]
        A = CFG["auto_tune"]["entry_score"]
        self.assertTrue(A["min"] <= t["entry_score"] <= A["max"])
        self.assertLessEqual(abs(t["entry_score"] - CFG["signals"]["entry_score"]), A["step"] + 1e-9)
        self.assertTrue(msgs[0])
        self.assertEqual(pf.effective(CFG, st)["signals"]["entry_score"], t["entry_score"])
        self.assertEqual(set(t["history"][-1]["grid"]), {70.0, 72.5, 75.0, 77.5, 80.0})

    def test_dividend_is_credited(self):
        prices, universe, fund, days = market(n=300)
        bars = prices["bars"]["S05"]
        ex = days[280]
        for d in bars:  # a 2 % dividend: adjusted closes before the ex-date are 2 % lower
            if d < ex:
                bars[d][4] = bars[d][3] * 0.98
        m = pf.Market(prices, universe, fund, CFG)
        k = m.s["S05"].idx[280]
        self.assertAlmostEqual(m.s["S05"].own_dy[k], 1 / 0.98 - 1, places=6)
        st = pf.new_state(CFG, days[270])
        st["positions"].append({"id": 1, "sym": "S05", "kind": "stock", "cert": False, "side": 1, "lev": 1.0, "N": 50000.0,
                                "p0": m.s["S05"].c[270], "fx0": m.fx_at("S05", 270), "d": days[270], "atr0": 1.0, "stop": 0.0,
                                "best": 1e9, "held": 1, "cost_in": 0.0, "fin": 0.0, "why": "", "type": "momentum", "score": 80,
                                "last": m.s["S05"].c[279]})
        st["last_day"] = days[279]
        sleeve = st["sleeve"]
        pf.step(st, m, 280, CFG)
        p = next((x for x in st["positions"] if x["id"] == 1), None)
        self.assertIsNotNone(p)
        self.assertGreater(p["div"], 50000 * 0.0204 * 0.85 * 0.8)

    def test_spearman_and_ic(self):
        self.assertAlmostEqual(pf.spearman(list(range(20)), [x * x for x in range(20)]), 1.0)
        self.assertAlmostEqual(pf.spearman(list(range(20)), [-x for x in range(20)]), -1.0)
        prices, universe, fund, days = market(n=500)
        ic = pf.ic_study(pf.Market(prices, universe, fund, CFG), 260, 15)
        self.assertIn("momentum", ic)
        self.assertTrue(-1 <= ic["score"]["ic"] <= 1)

    def test_regime_filter_blocks_buys_in_a_downtrend(self):
        prices, universe, fund, days = market(n=400)
        prices["bars"]["QQQ"] = walk(days, 5, drift=-0.003, vol=0.005)  # a steadily falling market
        cfg = pf.merged(CFG, {"signals": {"regime": "close_200"}})
        m = pf.Market(prices, universe, fund, cfg)
        st = pf.new_state(cfg, days[300])
        st["nav"] = []
        pf.step(st, m, 300, cfg, first=True)
        for i in range(301, 400):
            pf.step(st, m, i, cfg)
        self.assertEqual(st["trades"], [])
        self.assertEqual(st["positions"], [])

    def test_trade_start_and_resume(self):
        prices, universe, fund, days = market(n=300)
        now = datetime.strptime(days[280], "%Y-%m-%d").replace(hour=22, tzinfo=timezone.utc)
        cut = {"bars": {s: {d: v for d, v in b.items() if d <= days[280]} for s, b in prices["bars"].items()}, "errors": {}}
        st, msgs = pf.trade(None, CFG, universe, cut, fund, now)
        self.assertEqual(st["last_day"], days[280])
        self.assertEqual(len(st["nav"]), 1)
        self.assertTrue(st["pending"])  # first orders wait for the next open
        later = datetime.strptime(days[290], "%Y-%m-%d").replace(hour=22, tzinfo=timezone.utc)
        cut = {"bars": {s: {d: v for d, v in b.items() if d <= days[290]} for s, b in prices["bars"].items()}, "errors": {}}
        st, msgs = pf.trade(st, CFG, universe, cut, fund, later)
        self.assertEqual(st["last_day"], days[290])
        self.assertEqual(len(st["nav"]), 11)
        self.assertTrue(st["positions"])
        out = pf.export(st, CFG, universe, later)
        self.assertEqual(out["status"], "aktiv")
        json.dumps(out)
        self.assertIn("backtest", out)

    def test_unfinished_session_is_dropped(self):
        days = ["2026-11-23", "2026-11-24"]
        during = datetime(2026, 11, 24, 18, 0, tzinfo=timezone.utc)  # 13:00 New York
        after = datetime(2026, 11, 24, 22, 0, tzinfo=timezone.utc)
        self.assertEqual(pf.completed_days(days, during), days[:1])
        self.assertEqual(pf.completed_days(days, after), days)


if __name__ == "__main__":
    unittest.main()
