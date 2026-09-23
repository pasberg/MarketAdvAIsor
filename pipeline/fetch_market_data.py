"""Fetch price data for the MarketAdvAIsor mockup and write it as one JSON file.

Source: Yahoo Finance via the yfinance package (unofficial, personal use only;
Nordic quotes are delayed ~15 minutes). Replace with a licensed provider before
publishing the site.

Output: one file per part in --out-dir, refreshed at different rates:
  intra.json  intraday bars, last price, previous close, index strip (every 5 min)
  hour.json   hourly bars and five sessions of 5-minute bars (every 30 min)
  daily.json  daily and weekly bars (twice a day)
Each file: {"part", "generated": "<ISO UTC>", "source", "tz",
            "symbols": {"VOLV-B": {"currency", "last"?, "prevClose"?, "asOf"?,
                                   "series": {"intra": {"interval", "t", "o", "h", "l", "c", "v"}}}},
            "indices"? (intra only), "errors": {"SYM/series": "message"}}
Timestamps are Stockholm wall-clock time encoded as UTC epoch seconds (what the
chart library displays as-is); daily and weekly bars use midnight of the bar's date.
"""
from __future__ import annotations

import argparse
import calendar
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

LOCAL_TZ = "Europe/Stockholm"

# page symbol -> (Yahoo symbol, currency, 1-minute intraday bars (US) instead of 5-minute)
SYMBOLS = {
    "NVDA": ("NVDA", "USD", True), "MSFT": ("MSFT", "USD", True), "AAPL": ("AAPL", "USD", True),
    "AMZN": ("AMZN", "USD", True), "META": ("META", "USD", True), "AVGO": ("AVGO", "USD", True),
    "AMD": ("AMD", "USD", True), "GOOGL": ("GOOGL", "USD", True), "TSLA": ("TSLA", "USD", True),
    "COST": ("COST", "USD", True), "NFLX": ("NFLX", "USD", True), "JPM": ("JPM", "USD", True),
    "LLY": ("LLY", "USD", True), "XOM": ("XOM", "USD", True), "CAT": ("CAT", "USD", True),
    "UNH": ("UNH", "USD", True), "V": ("V", "USD", True),
    "VOLV-B": ("VOLV-B.ST", "SEK", False), "ATCO-A": ("ATCO-A.ST", "SEK", False),
    "ERIC-B": ("ERIC-B.ST", "SEK", False), "SAAB-B": ("SAAB-B.ST", "SEK", False),
    "INVE-B": ("INVE-B.ST", "SEK", False), "SEB-A": ("SEB-A.ST", "SEK", False),
    "ASSA-B": ("ASSA-B.ST", "SEK", False), "SAND": ("SAND.ST", "SEK", False),
    "HM-B": ("HM-B.ST", "SEK", False), "EVO": ("EVO.ST", "SEK", False), "ABB": ("ABB.ST", "SEK", False),
    "NDA-SE": ("NDA-SE.ST", "SEK", False), "NOVO-B": ("NOVO-B.CO", "DKK", False),
    "DSV": ("DSV.CO", "DKK", False), "VWS": ("VWS.CO", "DKK", False),
    "EQNR": ("EQNR.OL", "NOK", False), "MOWI": ("MOWI.OL", "NOK", False),
    "NESTE": ("NESTE.HE", "EUR", False), "KNEBV": ("KNEBV.HE", "EUR", False),
    "NOKIA": ("NOKIA.HE", "EUR", False),
    # indices and commodities (front-month futures), tradable via certificates
    "OMXS30": ("^OMX", "SEK", False), "SPX": ("^GSPC", "USD", True), "NDX100": ("^NDX", "USD", True),
    "GOLD": ("GC=F", "USD", False), "SILVER": ("SI=F", "USD", False),
    "COPPER": ("HG=F", "USD", False), "BRENT": ("BZ=F", "USD", False),
}

# value: Yahoo symbol, or candidates tried in order (the first with data wins)
INDICES = {
    "S&P 500": "^GSPC", "Nasdaq 100": "^NDX", "OMXS30": "^OMX", "OMXC25": "^OMXC25",
    "OBX": ["OBX.OL", "^OBX", "OSEBX.OL", "^OSEBX"], "OMXH25": "^OMXH25", "VIX": "^VIX",
}

# series key -> yfinance (interval, period, max bars kept)
# Longer history than the charts show, so the backtest has enough past signals.
SERIES = {
    "hour": ("60m", "3mo", 450),
    "day": ("1d", "5y", 1000),
    "week": ("1wk", "10y", 520),
}
INTRA_HIST = ("5m", "5d", 600)  # last five sessions in 5-minute bars, for the intraday backtest
INTRA = {True: ("1m", "5d"), False: ("5m", "5d")}  # US 1-minute, Nordic 5-minute


def _r(x: float) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), 4 if abs(x) < 10 else 2)


def _local_epoch(ts, daily: bool) -> int:
    """Stockholm wall-clock time (or bar date for daily/weekly) as UTC epoch seconds."""
    if daily:
        d = ts.date()
        return calendar.timegm((d.year, d.month, d.day, 0, 0, 0))
    local = ts.tz_convert(LOCAL_TZ) if ts.tzinfo else ts.tz_localize("UTC").tz_convert(LOCAL_TZ)
    return calendar.timegm(local.replace(tzinfo=None).timetuple())


def frame_to_series(df, interval: str, max_bars: int, last_session_only: bool = False) -> dict | None:
    """Convert one ticker's OHLCV DataFrame to the compact column format."""
    if df is None or df.empty:
        return None
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.empty:
        return None
    daily = interval in ("1d", "1wk")
    if last_session_only:
        idx = df.index.tz_convert(LOCAL_TZ) if df.index.tz is not None else df.index
        last_day = idx[-1].date()
        df = df[[d.date() == last_day for d in idx]]
    df = df.tail(max_bars)
    return {
        "interval": interval,
        "t": [_local_epoch(ts, daily) for ts in df.index],
        "o": [_r(v) for v in df["Open"]],
        "h": [_r(v) for v in df["High"]],
        "l": [_r(v) for v in df["Low"]],
        "c": [_r(v) for v in df["Close"]],
        "v": [int(v) if v == v else 0 for v in df.get("Volume", [0] * len(df))],
    }


def _download(yf, tickers: list[str], interval: str, period: str) -> dict:
    """Download several tickers; returns {yahoo_ticker: DataFrame}."""
    data = yf.download(tickers, interval=interval, period=period, group_by="ticker",
                       auto_adjust=False, prepost=False, threads=True, progress=False)
    out = {}
    for t in tickers:
        try:
            out[t] = data[t] if len(tickers) > 1 else data
        except KeyError:
            out[t] = None
    return out


PARTS = ("intra", "hour", "daily")


def _new_part(name: str) -> dict:
    return {"part": name, "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "Yahoo Finance (yfinance) — endast för prototyp", "tz": LOCAL_TZ,
            "symbols": {sym: {"currency": cur, "series": {}} for sym, (_, cur, _) in SYMBOLS.items()},
            "errors": {}}


def _put(part: dict, key: str, frames: dict, interval: str, max_bars: int, last_session: bool = False):
    for sym, (ysym, _, _) in SYMBOLS.items():
        if ysym not in frames:
            continue
        try:
            s = frame_to_series(frames[ysym], interval, max_bars, last_session)
            if s:
                part["symbols"][sym]["series"][key] = s
            else:
                part["errors"][f"{sym}/{key}"] = "ingen data"
        except Exception as e:  # keep going; one bad ticker must not stop the run
            part["errors"][f"{sym}/{key}"] = str(e)[:200]


def build_intra(yf) -> dict:
    """Intraday bars, last price / previous close and the index strip — refreshed every few minutes."""
    part = _new_part("intra")
    for is_us, (interval, period) in INTRA.items():
        tick = [y for y, _, us in SYMBOLS.values() if us == is_us]
        _put(part, "intra", _download(yf, tick, interval, period), interval, 420, last_session=True)
    quotes = _download(yf, [y for y, _, _ in SYMBOLS.values()], "1d", "5d")
    for sym, (ysym, _, _) in SYMBOLS.items():
        info = part["symbols"][sym]
        q = frame_to_series(quotes.get(ysym), "1d", 5)
        if q and len(q["c"]) >= 2:
            info["prevClose"], info["last"] = q["c"][-2], q["c"][-1]
        intra = info["series"].get("intra")
        if intra and intra["c"]:
            info["last"] = intra["c"][-1]
            info["asOf"] = intra["t"][-1]
    part["indices"] = {}
    cands = {name: ([y] if isinstance(y, str) else y) for name, y in INDICES.items()}
    idx_frames = _download(yf, [y for ys in cands.values() for y in ys], "1d", "3mo")
    for name, ysyms in cands.items():
        for ysym in ysyms:
            s = frame_to_series(idx_frames.get(ysym), "1d", 40)
            if s and len(s["c"]) >= 2:
                part["indices"][name] = {"last": s["c"][-1], "prevClose": s["c"][-2], "spark": s["c"], "yahoo": ysym}
                break
        else:
            part["errors"][f"index/{name}"] = "ingen data"
    return part


def build_hour(yf) -> dict:
    part = _new_part("hour")
    interval, period, n = SERIES["hour"]
    tick = [y for y, _, _ in SYMBOLS.values()]
    _put(part, "hour", _download(yf, tick, interval, period), interval, n)
    interval, period, n = INTRA_HIST
    _put(part, "intra5", _download(yf, tick, interval, period), interval, n)
    return part


def build_daily(yf) -> dict:
    part = _new_part("daily")
    tick = [y for y, _, _ in SYMBOLS.values()]
    for key in ("day", "week"):
        interval, period, n = SERIES[key]
        _put(part, key, _download(yf, tick, interval, period), interval, n)
    return part


BUILDERS = {"intra": build_intra, "hour": build_hour, "daily": build_daily}


def build(yf, parts=PARTS) -> dict[str, dict]:
    return {p: BUILDERS[p](yf) for p in parts}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--parts", default=",".join(PARTS),
                    help="kommaseparerat: intra (var 5:e min), hour (var 30:e min), daily (2 ggr/dag)")
    args = ap.parse_args(argv)
    parts = [p for p in args.parts.split(",") if p]
    unknown = set(parts) - set(PARTS)
    if unknown:
        ap.error(f"okända delar: {', '.join(sorted(unknown))}")
    import yfinance as yf

    ok_all = True
    for name, data in build(yf, parts).items():
        ok = sum(1 for s in data["symbols"].values() if s["series"])
        out = Path(args.out_dir) / f"{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"{name}: {ok}/{len(SYMBOLS)} symboler med data, {len(data['errors'])} fel -> {out} "
              f"({out.stat().st_size // 1024} kB)")
        for k, v in sorted(data["errors"].items())[:20]:
            print(f"  {k}: {v}")
        ok_all = ok_all and ok > 0
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
