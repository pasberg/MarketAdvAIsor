import unittest

import pandas as pd

from pipeline.fetch_market_data import SYMBOLS, build, frame_to_series


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

    def test_empty_frame(self):
        self.assertIsNone(frame_to_series(pd.DataFrame(), "1d", 10))


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
        self.assertEqual(set(data["hour"]["symbols"]["VOLV-B"]["series"]), {"hour"})

    def test_single_part(self):
        self.assertEqual(set(build(FakeYF, ["hour"])), {"hour"})


if __name__ == "__main__":
    unittest.main()
