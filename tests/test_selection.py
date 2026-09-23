import unittest

from etp import Direction, Product, ProductType, TradePlan, select_certificates, select_mini_futures
from etp.selection import mini_future_targets
from etp.sources import MorganStanleySource, SourceUnavailable, VontobelSource

L, S = Direction.LONG, Direction.SHORT


def bb(isin, lev, d=L, bid=10.0, ask=10.02, und="VOLVO B"):
    return Product(isin, "Vontobel", f"BULL {und} X{lev}", ProductType.BULL_BEAR, d, und, leverage=lev, bid=bid, ask=ask)


def mini(isin, fin, ko, d=L, und="VOLVO B", bid=5.0, ask=5.02):
    return Product(isin, "Morgan Stanley", "MINI", ProductType.MINI_FUTURE, d, und,
                   financing_level=fin, knockout_level=ko, bid=bid, ask=ask)


LONG_PLAN = TradePlan("Volvo B", L, price=284.3, entry=284.0, stop_loss=279.0)
SHORT_PLAN = TradePlan("VOLVO B", S, price=284.3, entry=284.0, stop_loss=289.0)


class Certificates(unittest.TestCase):
    def test_one_per_tier_tightest_spread(self):
        prods = [bb("A", 10, bid=10, ask=10.05), bb("B", 10, bid=10, ask=10.01), bb("C", 5), bb("D", 3)]
        picks = select_certificates(prods, LONG_PLAN)
        self.assertEqual([(p.tier, p.product.isin) for p in picks], [("x10", "B"), ("x5", "C")])

    def test_filters_direction_underlying_and_spread(self):
        prods = [bb("A", 10, d=S), bb("B", 10, und="SAAB B"), bb("C", 10, bid=10, ask=10.1)]
        self.assertEqual(select_certificates(prods, LONG_PLAN), [])


class MiniFutures(unittest.TestCase):
    def test_targets_keep_knockout_beyond_stop(self):
        fin, ko = mini_future_targets(LONG_PLAN, 0.12)
        self.assertAlmostEqual(fin, 284.0 - 284.0 * 0.12)
        self.assertEqual(ko, 279.0)
        fin_s, _ = mini_future_targets(SHORT_PLAN, 0.12)
        self.assertGreater(fin_s, SHORT_PLAN.entry)

    def test_long_tiers_pick_highest_qualifying_gearing(self):
        prods = [
            mini("TOO_CLOSE", fin=270, ko=275),   # financing too close to entry
            mini("KO_INSIDE", fin=240, ko=280),   # knock-out above the stop loss
            mini("AGG", fin=248, ko=252),         # ≥12 % away → aggressive
            mini("AGG2", fin=240, ko=244),
            mini("CONS", fin=225, ko=229),        # ≥20 % away → conservative
        ]
        picks = select_mini_futures(prods, LONG_PLAN)
        self.assertEqual([(p.tier, p.product.isin) for p in picks], [("aggressive", "AGG"), ("conservative", "CONS")])
        self.assertAlmostEqual(picks[0].gearing, 284.3 / (284.3 - 248))

    def test_short_mirrors_long(self):
        prods = [mini("S1", fin=320, ko=316, d=S), mini("S_KO_INSIDE", fin=330, ko=288, d=S)]
        picks = select_mini_futures(prods, SHORT_PLAN, tiers={"aggressive": 0.12})
        self.assertEqual([p.product.isin for p in picks], ["S1"])


class Sources(unittest.TestCase):
    def test_unimplemented_sources_fail_loudly(self):
        for src in (VontobelSource(), MorganStanleySource()):
            with self.assertRaises(SourceUnavailable):
                src.fetch()

    def test_morgan_stanley_product_url(self):
        self.assertEqual(MorganStanleySource.product_url("GB00BNTPCL16"),
                         "https://etp.morganstanley.com/se/sv/product-details/gb00bntpcl16")


if __name__ == "__main__":
    unittest.main()
