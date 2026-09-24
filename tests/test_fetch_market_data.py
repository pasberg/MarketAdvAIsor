import unittest

import pandas as pd

from pipeline.fetch_market_data import SESSION, SYMBOLS, UNIVERSE, build, frame_to_series, load_universe


def ohlcv(index):
    n = len(index)
    return pd.DataFrame({"Open": [10.0 + i for i in range(n)], "High": [11.0 + i for i in range(n)],
                         "Low": [9.0 + i for i in range(n)], "Close": [10.5 + i for i in range(n)],
                         "Volume": [1000] * n}, index=index)


class FrameToSeries(unittest.TestCase):
    def test_intraday_uses_stockholm_wall_clock(self):
        # 13:30 UTC in September = 15:30 in Stockholm (CEST)
        idx = pd.date_range("2026-09-23 13:30", periods=3, freq="1min", tz="UTC")
        s = frame_to_series(ohlcv(idx), "1m", 100)
        self.assertEqual(s["t"][0], int(pd.Timestamp("2026-09-23 15:30", tz="UTC").timestamp()))
        self.assertEqual(s["c"], [10.5, 11.5, 12.5])

    def test_daily_uses_bar_date_at_midnight(self):
        idx = pd.DatetimeIndex(["2026-09-22", "2026-09-23"]).tz_localize("America/New_York")
        s = frame_to_series(ohlcv(idx), "1d", 100)
        self.assertEqual(s["t"][-1], int(pd.Timestamp("2026-09-23", tz="UTC").timestamp()))

    def test_last_session_only_and_nan_rows(self):
        idx = pd.DatetimeIndex(["2026-09-22 09:00", "2026-09-23 09:00", "2026-09-23 09:05"]).tz_localize("Europe/Stockholm")
        df = ohlcv(idx)
        df.iloc[2, df.columns.get_loc("Close")] = float("nan")
        s = frame_to_series(df, "5m", 100, last_session_only=True)
        self.assertEqual(len(s["t"]), 1)

    def test_session_keeps_only_the_trading_window(self):
        # currencies trade around the clock; the intraday day for them ends at 22:00 Stockholm time
        idx = pd.date_range("2026-09-23 05:30", "2026-09-23 21:00", freq="30min", tz="UTC")  # 07:30–23:00 local
        s = frame_to_series(ohlcv(idx), "5m", 1000, session=SESSION["fx"])
        first, last = pd.Timestamp(s["t"][0], unit="s"), pd.Timestamp(s["t"][-1], unit="s")
        self.assertEqual((first.hour, first.minute), (8, 0))
        self.assertEqual((last.hour, last.minute), (21, 30))
        daily = frame_to_series(ohlcv(pd.date_range("2026-09-21", periods=3, freq="1D", tz="UTC")), "1d", 10, session=SESSION["fx"])
        self.assertEqual(len(daily["t"]), 3)  # daily bars are never trimmed

    def test_empty_frame(self):
        self.assertIsNone(frame_to_series(pd.DataFrame(), "1d", 10))


class Universe(unittest.TestCase):
    def test_universe_file(self):
        u = load_universe()
        self.assertGreater(len(u), 150)
        self.assertEqual(u["VOLV-B"]["yahoo"], "VOLV-B.ST")
        self.assertEqual(u["EURUSD"]["kind"], "fx")
        self.assertEqual({x["kind"] for x in u.values()}, {"stock", "index", "commodity", "fx"})
        self.assertEqual(len({x["yahoo"] for x in u.values()}), len(u))  # no Yahoo ticker twice
        self.assertTrue(SYMBOLS["NVDA"][2] and not SYMBOLS["VOLV-B"][2])  # US stocks use 1-minute bars


class FakeYF:
    """Stand-in for yfinance.download returning a grouped multi-ticker frame."""

    @staticmethod
    def download(tickers, interval, period, **kw):
        freq = {"1m": "1min", "5m": "5min", "60m": "60min", "1d": "1D", "1wk": "7D"}[interval]
        idx = pd.date_range("2026-09-23 07:00", periods=5, freq=freq, tz="UTC")
        return pd.concat({t: ohlcv(idx) for t in tickers}, axis=1)


class Build(unittest.TestCase):
    def test_build_all_parts_with_fake_source(self):
        data = build(FakeYF)
        self.assertEqual(set(data), {"intra", "hour", "daily"})
        intra = data["intra"]
        self.assertEqual(set(intra["symbols"]), set(SYMBOLS))
        volvo = intra["symbols"]["VOLV-B"]
        self.assertEqual(volvo["series"]["intra"]["interval"], "5m")
        self.assertEqual(intra["symbols"]["NVDA"]["series"]["intra"]["interval"], "1m")
        self.assertIn("last", volvo)
        self.assertIn("prevClose", volvo)
        self.assertIn("OMXS30", intra["indices"])
        self.assertEqual(set(data["daily"]["symbols"]["VOLV-B"]["series"]), {"day", "week"})
        self.assertEqual(set(data["hour"]["symbols"]["VOLV-B"]["series"]), {"hour", "intra5"})
        self.assertEqual(intra["meta"]["EURUSD"]["exchange"], "Valuta")
        self.assertEqual(intra["symbols"]["EURUSD"]["kind"], "fx")

    def test_retry_fills_tickers_a_large_request_dropped(self):
        calls = []

        class Flaky(FakeYF):
            @staticmethod
            def download(tickers, interval, period, **kw):
                calls.append(list(tickers))
                full = FakeYF.download(tickers, interval, period)
                return full.drop(columns=tickers[0], level=0) if len(calls) == 1 else full

        from pipeline.fetch_market_data import _download
        out = _download(Flaky, ["A", "B", "C"], "1d", "5d")
        self.assertTrue(all(df is not None for df in out.values()))
        self.assertEqual(calls[1], ["A"])

    def test_missing_parts(self):
        import json, tempfile
        from pathlib import Path
        from pipeline.fetch_market_data import missing_parts
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self.assertEqual(missing_parts(d), ["intra", "hour", "daily"])
            (d / "intra.json").write_text(json.dumps({"symbols": {s: {} for s in UNIVERSE}}))
            (d / "hour.json").write_text(json.dumps({"symbols": {"VOLV-B": {}}}))  # from before the universe grew
            self.assertEqual(missing_parts(d), ["hour", "daily"])

    def test_single_part(self):
        self.assertEqual(set(build(FakeYF, ["hour"])), {"hour"})


if __name__ == "__main__":
    unittest.main()
