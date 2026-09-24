import unittest

import numpy as np

from pipeline import intraday_lab as il


def session(closes, opens=None, highs=None, lows=None, prev=None):
    c = np.array(closes, dtype=float)
    o = np.array(opens if opens is not None else np.r_[c[0], c[:-1]], dtype=float)
    h = np.array(highs if highs is not None else np.maximum(o, c), dtype=float)
    l = np.array(lows if lows is not None else np.minimum(o, c), dtype=float)
    return {"o": o, "h": h, "l": l, "c": c, "v": np.ones(len(c)), "prev": prev, "day": 1}


class Rules(unittest.TestCase):
    def test_orb_enters_on_the_next_bar_after_a_close_above_the_range(self):
        # range from the first three bars (15 min) = 99–101; bar 4 closes at 102 -> long at bar 5's open
        d = session([100, 101, 99, 100, 102, 103, 104, 105], opens=[100, 100, 101, 99, 100, 102.5, 103, 104])
        s, entry, px = il.orb(d, 15)
        self.assertEqual((s, entry, px), (1, 102.5, 105))

    def test_orb_stop_at_the_other_side(self):
        d = session([100, 101, 99, 100, 102, 98, 97, 97], opens=[100, 100, 101, 99, 100, 102, 98, 97])
        s, entry, px = il.orb(d, 15)
        self.assertEqual(s, 1)
        self.assertEqual(px, 99)  # the range low; the bar opened at 102, above it, and fell through
        gapped = session([100, 101, 99, 100, 102, 97, 97, 97], opens=[100, 100, 101, 99, 100, 102, 97.5, 97])
        gapped["h"][6], gapped["l"][6] = 97.5, 96.5
        gapped["l"][5] = 99.5  # bar 5 stays above the stop; bar 6 opens below it
        self.assertEqual(il.orb(gapped, 15)[2], 97.5)  # filled at the open, below the stop

    def test_no_look_ahead_in_the_signal(self):
        d = session(list(np.linspace(100, 110, 40)))
        base = il.orb(d, 15)
        later = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
        later["c"][-5:] = 50  # a crash at the end must not change when or at what price the trade was entered
        later["l"][-5:] = 50
        self.assertEqual(il.orb(later, 15)[:2], base[:2])

    def test_momentum_trades_the_last_half_hour(self):
        closes = [100 + i * 0.1 for i in range(20)]
        d = session(closes)
        s, entry, px = il.momentum(d, 0.0)
        self.assertEqual(s, 1)
        self.assertEqual(entry, d["o"][14])
        self.assertEqual(px, d["c"][-1])
        self.assertIsNone(il.momentum(d, 5.0))  # the first half hour moved less than 5 %

    def test_gap_fade_targets_the_previous_close(self):
        # gap up 2 % from 100; short from bar 2's open 101.8, target 100
        d = session([101.8, 101.5, 100.5, 99.8, 100.2], opens=[102, 101.8, 101.5, 100.5, 99.8], prev=100)
        s, entry, px = il.gap(d, 1.0)
        self.assertEqual((s, entry, px), (-1, 101.8, 100))


class Run(unittest.TestCase):
    def test_run_reports_every_family(self):
        rng = np.random.default_rng(3)
        bars = {}
        for sym in ("VOLV-B", "NVDA", "EURUSD"):
            t, o, h, l, c = [], [], [], [], []
            px = 100.0
            for day in range(20):
                for i in range(60):
                    o.append(px); px *= 1 + rng.normal(0, 0.002); c.append(px)
                    h.append(max(o[-1], px) * 1.0005); l.append(min(o[-1], px) * 0.9995)
                    t.append((20000 + day) * 86400 + (540 + 5 * i) * 60)
            bars[sym] = {"t": t, "o": o, "h": h, "l": l, "c": c, "v": [1] * len(t)}
        res = il.run(bars, il.UNIVERSE)
        self.assertEqual({f["id"] for f in res["families"]}, set(il.FAMILIES))
        self.assertEqual(res["universe"], 3)
        for f in res["families"]:
            self.assertLessEqual(len(f["variants"]), 2)
            self.assertIn("0.15", f["best"]["by_cost"])


if __name__ == "__main__":
    unittest.main()
