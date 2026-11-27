"""Virtual portfolio: 1 000 000 SEK traded by rules on Nasdaq-100/Dow Jones stocks, indices and currencies.

Each completed US session d (bars up to the 16:00 ET close):
  1. orders decided at the close of the previous session are filled at d's open
  2. open positions are charged financing (certificates, short positions) and marked at d's close
  3. scores are computed from data up to d's close only, and the orders for the next open decided:
     exits first (stop on the close, signal reversed, held two weeks), then the best new entries
Every order counts against the daily limit (5). Capital not in positions sits in an index fund
(the benchmark: 50 % Nasdaq-100 + 50 % Dow Jones in SEK), so the trades alone decide whether the
portfolio beats the index. Rules: config/portfolio.json (changed by PR, applied forward only).

State (positions, trades, the analysis log) lives encrypted on the `portfolio` branch with the same
key scheme as the intraday archive; data/portfolio.json is what the page shows.

  python pipeline/portfolio.py --data data --mode trade    # after the US close (portfolio.yml)
  python pipeline/portfolio.py --data data --mode export   # page data from the stored state (publish.sh)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "portfolio.json"
UNIVERSE_FILE = ROOT / "config" / "portfolio_universe.csv"
NY = ZoneInfo("America/New_York")
STATE_FILE = "state.enc.json"
BACKTEST_MAX_AGE_DAYS = 6
FUND_VERSION = 3  # bump when fetch_fundamentals stores new fields (forces a refetch)
CERT_KINDS = {"index", "fx"}
FA_KEYS = ("earnings", "beats", "growth", "revisions", "value", "quality")


# ---------------------------------------------------------------- universe and data
def load_universe(path: Path = UNIVERSE_FILE) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out[r["sym"]] = {"name": r["name"], "yahoo": r["yahoo"], "kind": r["kind"], "tags": r["tags"], "currency": r.get("currency") or "USD",
                             "avanza_name": r.get("avanza_name") or ""}
    return out


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else x


def fetch_prices(universe: dict, bench: list[str], period: str = "3y") -> dict:
    """Daily OHLC per symbol (raw prices) plus adjusted closes for the benchmark funds and USD/SEK."""
    import yfinance as yf
    wanted = {u["yahoo"]: s for s, u in universe.items()}
    for b in bench:
        wanted.setdefault(b, b)
    wanted.setdefault("USDSEK=X", "USDSEK")
    out, errors = {}, {}
    tickers = list(wanted)
    for i in range(0, len(tickers), 40):
        chunk = tickers[i:i + 40]
        try:
            df = yf.download(chunk, period=period, interval="1d", auto_adjust=False, group_by="ticker",
                             progress=False, threads=True)
        except Exception as e:  # one bad chunk must not stop the rest
            errors[",".join(chunk)] = str(e)
            continue
        for y in chunk:
            try:
                sub = df[y] if y in df.columns.get_level_values(0) else None
                if sub is None:
                    raise KeyError("inga data")
                sub = sub.dropna(subset=["Open", "High", "Low", "Close"])
                if sub.empty:
                    raise KeyError("inga data")
                rows = {}
                for ts, r in sub.iterrows():
                    rows[ts.strftime("%Y-%m-%d")] = [_num(r["Open"]), _num(r["High"]), _num(r["Low"]), _num(r["Close"]),
                                                     _num(r.get("Adj Close", r["Close"]))]
                out[wanted[y]] = rows
            except Exception as e:
                errors[y] = str(e)
    return {"bars": out, "errors": errors}


def fetch_fundamentals(universe: dict, cache: dict, now: datetime) -> dict:
    """Earnings dates with EPS surprise, and a few key figures, per stock (Yahoo; refreshed daily)."""
    import yfinance as yf
    today = now.strftime("%Y-%m-%d")
    out = dict(cache)
    for sym, u in universe.items():
        old = out.get(sym) or {}
        if u["kind"] != "stock" or (old.get("fetched") == today and old.get("fv") == FUND_VERSION):
            continue
        rec = {"fetched": today, "fv": FUND_VERSION, "events": [], "info": {}, "analysts": []}
        try:
            t = yf.Ticker(u["yahoo"])
            ed = t.get_earnings_dates(limit=16)
            if ed is not None:
                for ts, r in ed.iterrows():
                    rec["events"].append({"ts": ts.isoformat(), "est": _num(r.get("EPS Estimate")),
                                          "eps": _num(r.get("Reported EPS")), "surprise": _num(r.get("Surprise(%)"))})
            ud = t.upgrades_downgrades
            if ud is not None and len(ud):
                cut = (now - timedelta(days=1100)).strftime("%Y-%m-%d")
                for ts, r in ud.iterrows():
                    d = ts.strftime("%Y-%m-%d")
                    if d >= cut:
                        rec["analysts"].append([d, str(r.get("Action") or ""), _num(r.get("currentPriceTarget")),
                                                _num(r.get("priorPriceTarget"))])
                rec["analysts"].sort()
            info = t.info or {}
            rec["info"] = {k: _num(info.get(k)) for k in ("marketCap", "recommendationMean", "forwardPE", "trailingPE",
                                                          "earningsGrowth", "revenueGrowth", "profitMargins")}
            rec["info"]["sector"] = info.get("sector") or None
        except Exception as e:
            if "Rate" in type(e).__name__ or "Too Many" in str(e):
                print(f"::warning::Yahoo begränsar anropen — nyckeltal för {sym} och resten används från förra hämtningen.")
                break
            rec["error"] = str(e)[:200]
            if sym in out:  # keep the old figures rather than lose them on one failed call
                rec = dict(out[sym], error=rec["error"])
        out[sym] = rec
        time.sleep(0.2)
    return out


# ---------------------------------------------------------------- indicators
def _sma(a, n, i):
    if i + 1 < n or any(x is None for x in a[i + 1 - n:i + 1]):
        return None
    return sum(a[i + 1 - n:i + 1]) / n


class Series:
    """One instrument aligned to the US trading calendar. traded[i]: a bar exists on day i (else the
    previous bar is carried, and orders for that day wait)."""

    def __init__(self, days: list[str], bars: dict[str, list]):
        self.n = len(days)
        self.o, self.h, self.l, self.c, self.traded = [None] * self.n, [None] * self.n, [None] * self.n, [None] * self.n, [False] * self.n
        own = sorted(bars)
        # indicators run on the instrument's own bars, then are mapped to the US days
        oc = [bars[d][3] for d in own]
        oh = [bars[d][1] for d in own]
        ol = [bars[d][2] for d in own]
        tr = [None] * len(own)
        for k in range(len(own)):
            if k == 0:
                tr[k] = oh[k] - ol[k]
            else:
                tr[k] = max(oh[k] - ol[k], abs(oh[k] - oc[k - 1]), abs(ol[k] - oc[k - 1]))
        atr, rsi = [None] * len(own), [None] * len(own)
        a, g, lo = None, None, None
        for k in range(len(own)):
            a = tr[k] if a is None else (a * 13 + tr[k]) / 14 if k >= 1 else tr[k]
            atr[k] = a if k >= 13 else None
            if k >= 1:
                ch = oc[k] - oc[k - 1]
                up, dn = max(ch, 0), max(-ch, 0)
                g = up if g is None else (g * 13 + up) / 14
                lo = dn if lo is None else (lo * 13 + dn) / 14
                if k >= 14:
                    rsi[k] = 100.0 if lo == 0 else 100 - 100 / (1 + g / lo)
        self.atr, self.rsi, self.sma50, self.sma200, self.idx = ([None] * self.n for _ in range(5))
        self.own_c, self.own_h = oc, oh
        # dividend yield on the ex-date: the adjusted close moves more than the raw close
        oa = [bars[d][4] if len(bars[d]) > 4 and bars[d][4] else bars[d][3] for d in own]
        self.own_dy = [0.0] * len(own)
        for k in range(1, len(own)):
            if oc[k - 1] and oc[k] and oa[k - 1] and oa[k]:
                y = (oa[k] / oa[k - 1]) / (oc[k] / oc[k - 1]) - 1
                self.own_dy[k] = y if 1e-4 < y < 0.2 else 0.0
        pos = {d: k for k, d in enumerate(own)}
        ptr = -1
        for i, d in enumerate(days):
            while ptr + 1 < len(own) and own[ptr + 1] <= d:
                ptr += 1
            if d in pos:
                self.traded[i] = True
                self.o[i], self.h[i], self.l[i] = bars[d][0], bars[d][1], bars[d][2]
            if ptr < 0:
                continue
            self.c[i] = oc[ptr]
            self.idx[i] = ptr
            self.atr[i], self.rsi[i] = atr[ptr], rsi[ptr]
            self.sma50[i], self.sma200[i] = _sma(oc, 50, ptr), _sma(oc, 200, ptr)

    def ret(self, i: int, days: int, skip: int) -> float | None:
        k = self.idx[i]
        if k is None or k - days < 0:
            return None
        a, b = self.own_c[k - days], self.own_c[k - skip]
        return b / a - 1 if a else None

    def vol(self, i: int, n: int = 63) -> float | None:
        k = self.idx[i]
        if k is None or k < n:
            return None
        r = [self.own_c[j] / self.own_c[j - 1] - 1 for j in range(k - n + 1, k + 1)]
        m = sum(r) / n
        return math.sqrt(sum((x - m) ** 2 for x in r) / (n - 1))


def earnings_events(fund: dict, days: list[str]) -> list[dict]:
    """Reported quarters with the US trading day the price reacted on (after 12:00 ET -> the next day)."""
    evs = []
    for e in fund.get("events") or []:
        try:
            ts = datetime.fromisoformat(e["ts"]).astimezone(NY)
        except Exception:
            continue
        d = ts.strftime("%Y-%m-%d")
        after = ts.hour >= 12
        react = next((x for x in days if (x > d if after else x >= d)), None)
        evs.append({"date": d, "react": react, "eps": e.get("eps"), "est": e.get("est"), "surprise": e.get("surprise"),
                    "reported": e.get("eps") is not None})
    return sorted(evs, key=lambda x: x["date"])


def clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def quality(info: dict) -> float | None:
    parts = []
    rm = (info or {}).get("recommendationMean")
    if rm:
        parts.append(clip((3 - rm) / 1.5))
    for k in ("earningsGrowth", "revenueGrowth"):
        g = (info or {}).get(k)
        if g is not None:
            parts.append(clip(g / 0.2))
    fpe, tpe = (info or {}).get("forwardPE"), (info or {}).get("trailingPE")
    if fpe and tpe and fpe > 0 and tpe > 0:
        parts.append(clip((tpe / fpe - 1) / 0.25))
    return sum(parts) / len(parts) if parts else None


class Market:
    """Everything the rules may look at, indexed by US trading day."""

    def __init__(self, prices: dict, universe: dict, fund: dict, cfg: dict, use_quality: bool = True):
        bars = prices["bars"]
        parts = cfg["benchmark"]["parts"]
        base = [b for b in parts if b in bars]
        if len(base) != len(parts) or "USDSEK" not in bars:
            raise RuntimeError("Jämförelseindex eller USD/SEK saknas i kursdatan")
        self.days = sorted(set.intersection(*(set(bars[b]) for b in base)))
        self.universe = {s: u for s, u in universe.items() if s in bars}
        self.cfg, self.use_quality = cfg, use_quality
        self.s = {s: Series(self.days, bars[s]) for s in self.universe}
        self.fx = Series(self.days, bars["USDSEK"])
        # benchmark: daily total return in SEK (adjusted closes, so dividends count)
        self.bench_r = [0.0] * len(self.days)
        for i in range(1, len(self.days)):
            r = sum(w * (bars[b][self.days[i]][4] / bars[b][self.days[i - 1]][4] - 1) for b, w in parts.items())
            f0, f1 = self.fx.c[i - 1], self.fx.c[i]
            if cfg["benchmark"].get("currency", "USD") != "USD":
                f0 = f1 = None  # a Swedish index fund: no currency effect
            self.bench_r[i] = (1 + r) * (f1 / f0 if f0 and f1 else 1) - 1
        self.qqq = Series(self.days, bars[base[0]])
        self.events = {s: earnings_events(fund.get(s) or {}, self.days) for s, u in self.universe.items() if u["kind"] == "stock"}
        self.info = {s: (fund.get(s) or {}).get("info") or {} for s in self.universe}
        # analyst actions as dated scores: up/down +-1, raised/cut price target +-0.5
        self.ana = {}
        for s in self.events:
            rows = []
            for d, act, cur, prior in (fund.get(s) or {}).get("analysts") or []:
                v = {"up": 1.0, "down": -1.0}.get(act, 0.0)
                if cur and prior and cur != prior:
                    v += 0.5 if cur > prior else -0.5
                if v:
                    rows.append((d, v))
            self.ana[s] = (sorted(rows), [r[0] for r in sorted(rows)])
        # analysts' price targets as set (date, target), for the reward side of risk/reward
        self.targets = {}
        for s in self.events:
            t = sorted((d, cur) for d, _, cur, _ in (fund.get(s) or {}).get("analysts") or [] if cur)
            self.targets[s] = (t, [x[0] for x in t])
        self.pos = {d: i for i, d in enumerate(self.days)}

    def fx_at(self, sym: str, i: int) -> float:
        u = self.universe[sym]
        return (self.fx.c[i] or 1.0) if u["kind"] == "stock" and u.get("currency", "USD") == "USD" else 1.0

    def scores(self, i: int) -> dict[str, dict]:
        """Long score 0-100 per instrument at the close of day i (short score = 100 - score)."""
        sig = self.cfg["signals"]
        out, mom = {}, {}
        for s, ser in self.s.items():
            if ser.c[i] is None or ser.sma200[i] is None or ser.atr[i] is None:
                continue
            r = ser.ret(i, sig["momentum_days"], sig["momentum_skip"])
            if r is None:
                continue
            mom[s] = r
        stocks = sorted((s for s in mom if self.universe[s]["kind"] == "stock"), key=lambda s: mom[s])
        rank = {s: (k + 0.5) / len(stocks) for k, s in enumerate(stocks)}
        fa = {s: self.fa_comps(s, i) for s in stocks}
        ey = sorted((s for s in stocks if fa[s].get("ey") is not None), key=lambda s: fa[s]["ey"])
        ey_rank = {s: (k + 0.5) / len(ey) for k, s in enumerate(ey)}
        if sig.get("value_by_sector"):
            groups = {}
            for s_ in ey:
                groups.setdefault(self.info.get(s_, {}).get("sector"), []).append(s_)
            for sec, mem in groups.items():
                if sec and len(mem) >= 4:  # rank within the sector; small or unknown sectors keep the overall rank
                    for k, s_ in enumerate(mem):  # ey is sorted, so each group is too
                        ey_rank[s_] = (k + 0.5) / len(mem)
        for s in mom:
            ser, u = self.s[s], self.universe[s]
            c, a, b = ser.c[i], ser.sma50[i], ser.sma200[i]
            comp = {"trend": (math.copysign(1, c - a) + math.copysign(1, a - b) + math.copysign(1, c - b)) / 3}
            if u["kind"] == "stock":
                comp["momentum"] = 2 * rank[s] - 1
                comp["earnings"], ev = self.earnings_comp(s, i)
                f = fa[s]
                for k in ("beats", "growth", "revisions"):
                    if f.get(k) is not None:
                        comp[k] = f[k]
                if s in ey_rank:
                    comp["value"] = 2 * ey_rank[s] - 1
                if self.use_quality and "quality" in sig["weights_stock"]:
                    q = quality(self.info.get(s))
                    if q is not None:
                        comp["quality"] = q
                w = sig["weights_stock"]
            else:
                v = ser.vol(i)
                z = mom[s] / (v * math.sqrt(sig["momentum_days"] - sig["momentum_skip"])) if v else 0
                comp["momentum"] = clip(z / 2)
                ev = None
                w = sig["weights_cert"]
            tw = sum(w[k] for k in comp if k in w)
            raw = sum(w[k] * comp[k] for k in comp if k in w) / tw if tw else 0
            out[s] = {"score": round(50 + 50 * raw, 1), "comp": {k: round(v, 2) for k, v in comp.items()},
                      "mom_pct": round(mom[s] * 100, 1), "atr_pct": round(ser.atr[i] / c * 100, 2),
                      "rsi": round(ser.rsi[i], 0) if ser.rsi[i] is not None else None, "close": c, "event": ev,
                      "fa": {k: v for k, v in (fa.get(s) or {}).items() if k in ("eps_g_pct", "pe", "ups", "downs", "avg_surprise")} or None}
        return out

    def fa_comps(self, s: str, i: int) -> dict:
        """Fundamentals known at the close of day i: reported quarters and dated analyst actions."""
        sig = self.cfg["signals"]
        d = self.days[i]
        evs = [e for e in self.events.get(s, []) if e["reported"] and e["react"] and e["react"] <= d]
        out = {}
        sur = [e["surprise"] for e in evs[-4:] if e["surprise"] is not None]
        if len(sur) >= 2:
            out["avg_surprise"] = round(sum(sur) / len(sur), 1)
            out["beats"] = clip(out["avg_surprise"] / sig["surprise_full_pct"])
        eps = [e["eps"] for e in evs]
        if len(eps) >= 8:
            ttm, prev = sum(eps[-4:]), sum(eps[-8:-4])
            if prev > 0:
                g = ttm / prev - 1
                out["eps_g_pct"] = round(g * 100, 1)
                out["growth"] = clip(g * 100 / sig["growth_full_pct"])
            else:
                out["growth"] = 0.5 if ttm > prev else -0.5
        if len(eps) >= 4:
            ttm = sum(eps[-4:])
            c = self.s[s].c[i]
            if c:
                out["ey"] = ttm / c
                out["pe"] = round(c / ttm, 1) if ttm > 0 else None
        rows, dates = self.ana.get(s, ([], []))
        if dates:
            import bisect
            lo = (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=sig["revision_days"])).strftime("%Y-%m-%d")
            win = rows[bisect.bisect_right(dates, lo):bisect.bisect_right(dates, d)]
            out["ups"] = sum(1 for _, v in win if v > 0)
            out["downs"] = sum(1 for _, v in win if v < 0)
            out["revisions"] = clip(sum(v for _, v in win) / sig["revision_full"])
        return out

    def regime_ok(self, i: int, kind: str) -> bool:
        """Market regime from the benchmark's first fund (QQQ): new buys only in an up-trend."""
        q = self.qqq
        if q.sma200[i] is None:
            return True
        if kind == "close_200":
            return q.c[i] > q.sma200[i]
        if kind == "50_200":
            return q.sma50[i] > q.sma200[i]
        return True

    def rr(self, s: str, i: int, side: int) -> dict | None:
        """Reward / risk at the close of day i. Risk: the stop distance (atr_mult x ATR). Reward, by
        signals.rr_source: 'analyst' = mean of the price targets set in the last rr_target_days, or
        'high' = distance to the highest price of the last year."""
        sig, ser = self.cfg["signals"], self.s[s]
        c, atr = ser.c[i], ser.atr[i]
        if not c or not atr:
            return None
        risk = self.cfg["stops"]["atr_mult"] * atr
        src = sig.get("rr_source", "analyst")
        if src == "analyst":
            rows, dates = self.targets.get(s, ([], []))
            if not dates:
                return None
            import bisect
            d = self.days[i]
            lo = (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=sig.get("rr_target_days", 90))).strftime("%Y-%m-%d")
            win = [t for _, t in rows[bisect.bisect_right(dates, lo):bisect.bisect_right(dates, d)]]
            if len(win) < 3:
                return None
            ref = sum(win) / len(win)
        else:
            k = ser.idx[i]
            ref = max(ser.own_h[max(0, k - 251):k + 1]) if side > 0 else min(ser.own_c[max(0, k - 251):k + 1])
        reward = (ref - c) * side
        return {"rr": round(reward / risk, 2), "ref": round(ref, 2), "src": src}

    def earnings_comp(self, s: str, i: int):
        sig = self.cfg["signals"]
        d = self.days[i]
        last = None
        for e in self.events.get(s, []):
            if e["reported"] and e["react"] and e["react"] <= d:
                last = e
        if not last:
            return 0.0, None
        k = i - self.pos[last["react"]] if last["react"] in self.pos else 99
        lo, hi = sig["earnings_window"]
        if not lo <= k <= hi or last["surprise"] is None:
            return 0.0, None
        ri = self.pos[last["react"]]
        ser = self.s[s]
        if ri < 1 or ser.c[ri - 1] is None or not ser.traded[ri]:
            return 0.0, None
        react = ser.c[ri] / ser.c[ri - 1] - 1 - (self.qqq.c[ri] / self.qqq.c[ri - 1] - 1)
        e = clip(0.5 * clip(last["surprise"] / sig["surprise_full_pct"]) + 0.5 * clip(react * 100 / sig["reaction_full_pct"]))
        return e, {"date": last["date"], "surprise": round(last["surprise"], 1), "reaction": round(react * 100, 1), "days_since": k}

    def next_earnings(self, s: str, i: int) -> int | None:
        """US trading days until the next report (None: not known)."""
        d = self.days[i]
        for e in self.events.get(s, []):
            if e["date"] > d:
                nd = e["date"]
                later = [x for x in self.days if x > d and x <= nd]
                if self.days[-1] >= nd:
                    return len(later)
                # beyond the data: count weekdays
                dt, n = datetime.strptime(self.days[-1], "%Y-%m-%d"), len(later)
                end = datetime.strptime(nd, "%Y-%m-%d")
                while dt < end:
                    dt += timedelta(days=1)
                    n += dt.weekday() < 5
                return n
        return None


# ---------------------------------------------------------------- the book
def new_state(cfg: dict, day: str) -> dict:
    cap = float(cfg["start_capital"])
    return {"v": 1, "start": day, "capital0": cap, "last_day": day, "sleeve": cap, "bench": cap,
            "positions": [], "pending": [], "trades": [], "nav": [{"d": day, "nav": cap, "bench": cap, "n": 0, "long": 0, "short": 0}],
            "log": [], "seq": 0, "rules_version": cfg["version"]}


def pos_value(p: dict, price: float, fx: float) -> float:
    """SEK value of a position (stake plus result; a certificate cannot lose more than its stake)."""
    move = p["side"] * p["lev"] * (price / p["p0"] - 1)
    v = p["N"] * (1 + move) * (fx / p["fx0"] if p["kind"] == "stock" else 1.0)
    return max(v, 0.0)


def cost_rate(cfg: dict, p: dict) -> float:
    c = cfg["costs"]
    return (c["stock_pct_per_order"] if p["kind"] == "stock" and not p["cert"] else c["cert_pct_per_order"]) / 100


def nav_of(state: dict, m: Market, i: int) -> float:
    return state["sleeve"] + sum(pos_value(p, m.s[p["sym"]].c[i], m.fx_at(p["sym"], i)) for p in state["positions"])


def _sv(x, d=0):
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def step(state: dict, m: Market, i: int, cfg: dict, keep_log: bool = True, first: bool = False) -> None:
    """Process US trading day i: fill yesterday's orders at the open, mark at the close, decide tomorrow.
    first: the start day (no orders, no return; only the first decisions)."""
    d = m.days[i]
    lim, sig = cfg["limits"], cfg["signals"]
    day_log = {"d": d, "fills": [], "decisions": [], "skipped": []}
    # 1. orders at the open (the sleeve's return for the day applies to what stays in it)
    proceeds, keep = 0.0, []
    for o in state["pending"]:
        ser = m.s.get(o["sym"])
        if ser is None or not ser.traded[i] or ser.o[i] is None:
            if o["action"] == "close":
                keep.append(o)  # the market is closed for this instrument today: try at the next open
            else:
                day_log["skipped"].append({"sym": o["sym"], "why": "marknaden stängd"})
            continue
        px, fx = ser.o[i], m.fx_at(o["sym"], i)
        if o["action"] == "close":
            p = next((x for x in state["positions"] if x["id"] == o["id"]), None)
            if not p:
                continue
            val = pos_value(p, px, fx)
            cost = p["N"] * p["lev"] * cost_rate(cfg, p)
            proceeds += val - cost
            state["positions"].remove(p)
            pnl = val - cost - p["N"] - p["cost_in"] - p["fin"] + p.get("div", 0.0)
            t = {"id": p["id"], "sym": p["sym"], "side": p["side"], "kind": p["kind"], "cert": p["cert"], "lev": p["lev"],
                 "N": round(p["N"]), "open_d": p["d"], "p0": p["p0"], "close_d": d, "p1": px, "days": p["held"],
                 "pnl": round(pnl), "ret_pct": round(pnl / p["N"] * 100, 2), "cost": round(cost + p["cost_in"] + p["fin"]),
                 "why_in": p["why"], "why_out": o["why"], "type": p["type"], "score_in": p["score"],
                 "comp": p.get("comp"), "rr": p.get("rr")}
            state["trades"].append(t)
            day_log["fills"].append({"sym": p["sym"], "action": "stäng", "side": p["side"], "price": px, "pnl": round(pnl), "why": o["why"]})
        else:
            cost = o["N"] * o["lev"] * cost_rate(cfg, o)
            if state["sleeve"] < o["N"] + cost:
                day_log["skipped"].append({"sym": o["sym"], "why": "inte nog med fritt kapital"})
                continue
            state["sleeve"] -= o["N"] + cost
            state["seq"] += 1
            atr = o["atr"]
            k = cfg["stops"]["atr_mult"]
            p = {"id": state["seq"], "sym": o["sym"], "kind": o["kind"], "cert": o["cert"], "side": o["side"], "lev": o["lev"],
                 "N": o["N"], "p0": px, "fx0": fx, "d": d, "atr0": atr, "stop": px - o["side"] * k * atr,
                 "best": px, "held": 0, "cost_in": cost, "fin": 0.0, "why": o["why"], "type": o["type"], "score": o["score"],
                 "comp": o.get("comp"), "rr": o.get("rr")}
            state["positions"].append(p)
            day_log["fills"].append({"sym": o["sym"], "action": "köp" if o["side"] > 0 else "blanka", "side": o["side"],
                                     "price": px, "N": round(o["N"]), "lev": o["lev"], "why": o["why"]})
    state["pending"] = keep
    br = 0.0 if first else m.bench_r[i]
    state["sleeve"] = state["sleeve"] * (1 + br) + proceeds
    state["bench"] *= 1 + br
    # 2. financing and marks at the close
    fin_rate = cfg["costs"]["financing_pct_year"] / 100 / 252
    for p in state["positions"]:
        if p["cert"] and not first:
            f = p["N"] * p["lev"] * fin_rate
            p["fin"] += f
            state["sleeve"] -= f
        ser = m.s[p["sym"]]
        if p["kind"] == "stock" and p["d"] != d and ser.traded[i] and ser.idx[i] is not None and not first:
            dy = ser.own_dy[ser.idx[i]]
            if dy:
                # a long holder receives the dividend less 15 % US withholding tax; a short position pays it
                amt = pos_value(p, p.get("last", p["p0"]), m.fx_at(p["sym"], i)) * dy * (cfg["costs"].get("dividend_net", 0.85) if p["side"] > 0 else -1.0)
                p["div"] = p.get("div", 0.0) + amt
                state["sleeve"] += amt
        c = m.s[p["sym"]].c[i]
        if m.s[p["sym"]].traded[i]:
            p["held"] += 1
        p["last"] = c
        if cfg["stops"]["trailing"] and (c - p["best"]) * p["side"] > 0:
            p["best"] = c
            p["stop"] = max(p["stop"], c - k_atr(cfg, p)) if p["side"] > 0 else min(p["stop"], c + k_atr(cfg, p))
    nav = nav_of(state, m, i)
    long_x = sum(p["N"] * p["lev"] for p in state["positions"] if p["side"] > 0)
    short_x = sum(p["N"] * p["lev"] for p in state["positions"] if p["side"] < 0)
    state["nav"].append({"d": d, "nav": round(nav), "bench": round(state["bench"]), "n": len(state["positions"]),
                         "long": round(long_x), "short": round(short_x)})
    # 3. decide the orders for the next open
    sc = m.scores(i)
    orders = []
    exits = []
    carried = {o["id"]: o["why"] for o in keep if o["action"] == "close"}
    for p in state["positions"]:
        if p["id"] in carried:
            exits.append((-1, p, carried[p["id"]]))
            continue
        c = m.s[p["sym"]].c[i]
        s = sc.get(p["sym"])
        side_score = None if s is None else (s["score"] if p["side"] > 0 else 100 - s["score"])
        if (c - p["stop"]) * p["side"] <= 0:
            exits.append((0, p, f"Stopp: stängde {fmt_px(c)} {'under' if p['side'] > 0 else 'över'} stoppen {fmt_px(p['stop'])}"))
        elif cfg["stops"].get("target_r") and (c - p["p0"]) * p["side"] >= cfg["stops"]["target_r"] * k_atr(cfg, p):
            exits.append((0.5, p, f"Mål nått: {cfg['stops']['target_r']:g} R ({fmt_px(c)})"))
        elif side_score is not None and side_score < sig["exit_score"] and p["held"] >= lim["min_hold_days"]:
            exits.append((1, p, f"Signalen har vänt: poäng {side_score:.0f} (under {sig['exit_score']})"))
        elif p["held"] >= lim["hold_days"]:
            exits.append((2, p, f"Tidsgräns: hållen {p['held']} handelsdagar"))
    exits.sort(key=lambda x: x[0])
    for _, p, why in exits[:lim["max_orders_per_day"]]:
        orders.append({"action": "close", "id": p["id"], "sym": p["sym"], "why": why})
    for _, p, why in exits[lim["max_orders_per_day"]:]:
        day_log["skipped"].append({"sym": p["sym"], "why": "stängs nästa dag (taket på 5 order)"})
    slots_orders = lim["max_orders_per_day"] - len(orders)
    n_after = len(state["positions"]) - len(orders)
    held = {p["sym"] for p in state["positions"]}
    kinds = {"fx": sum(p["kind"] == "fx" for p in state["positions"]), "index": sum(p["kind"] == "index" for p in state["positions"])}
    cands = []
    for s, x in sc.items():
        u = m.universe[s]
        for side in (1, -1):
            ss = x["score"] if side > 0 else 100 - x["score"]
            if ss < sig["entry_score"]:
                continue
            why_not = None
            if s in held:
                why_not = "redan i portföljen"
            elif side > 0 and sig.get("regime") and not m.regime_ok(i, sig["regime"]):
                why_not = {"close_200": "Nasdaq-100 under sitt 200-dagars snitt", "50_200": "Nasdaq-100:s 50-dagars snitt under 200-dagars"}.get(sig["regime"], "marknadsfilter")
            elif side > 0 and x["rsi"] is not None and x["rsi"] > sig["rsi_max_long"]:
                why_not = f"överköpt (RSI {x['rsi']:.0f})"
            elif side < 0 and x["rsi"] is not None and x["rsi"] < sig["rsi_min_short"]:
                why_not = f"översåld (RSI {x['rsi']:.0f})"
            elif side < 0 and not sig.get("allow_short", True):
                why_not = "blankning avstängd i reglerna"
            elif side < 0 and u["kind"] == "stock" and (m.info.get(s, {}).get("marketCap") or 0) < sig["short_min_market_cap_usd"]:
                why_not = "för litet bolag för mini short/bear"
            elif u["kind"] != "stock" and lim.get(f"max_{u['kind']}_positions", 1) == 0:
                why_not = "valutor och index avstängda i reglerna"
            elif u["kind"] == "stock":
                ne = m.next_earnings(s, i)
                if ne is not None and ne <= sig["no_entry_days_before_earnings"]:
                    why_not = f"rapport om {ne} handelsdagar"
            rr = m.rr(s, i, side) if u["kind"] == "stock" else None
            if not why_not and sig.get("min_rr"):
                if rr is None:
                    if sig.get("rr_require", True):
                        why_not = "risk/reward okänd (för få riktkurser)" if sig.get("rr_source", "analyst") == "analyst" else "risk/reward okänd"
                elif rr["rr"] < sig["min_rr"]:
                    why_not = f"risk/reward {rr['rr']:.1f} (under {sig['min_rr']:g})"
            cands.append({"sym": s, "side": side, "score": ss, "x": dict(x, rr=rr), "why_not": why_not})
    cands.sort(key=lambda c: -c["score"])
    nav_now = nav
    for c in cands:
        if c["why_not"]:
            continue
        if slots_orders <= 0:
            c["why_not"] = "taket på 5 order per dag nått"
            continue
        if n_after >= lim["max_positions"]:
            c["why_not"] = f"redan {lim['max_positions']} innehav"
            continue
        k = m.universe[c["sym"]]["kind"]
        if k in kinds and kinds[k] >= lim[f"max_{k}_positions"]:
            c["why_not"] = f"redan {kinds[k]} positioner i {'valutor' if k == 'fx' else 'index'}"
            continue
        o = size_order(cfg, m, c, nav_now)
        orders.append(o)
        slots_orders -= 1
        n_after += 1
        held.add(c["sym"])
        if k in kinds:
            kinds[k] += 1
    state["pending"] = orders
    for o in orders:
        day_log["decisions"].append({"sym": o["sym"], "action": "stäng" if o["action"] == "close" else ("köp" if o["side"] > 0 else "blanka"),
                                     "why": o["why"], **({"N": round(o["N"]), "lev": o["lev"]} if o["action"] == "open" else {})})
    if keep_log:
        top = lambda side: [{"sym": c["sym"], "score": c["score"], "comp": c["x"]["comp"], "mom": c["x"]["mom_pct"], "rsi": c["x"]["rsi"],
                             "atr": c["x"]["atr_pct"], "event": c["x"]["event"], "why_not": c["why_not"],
                             "rr": (c["x"].get("rr") or {}).get("rr")}
                            for c in cands if c["side"] == side][:8]
        day_log.update({"nav": round(nav), "bench": round(state["bench"]), "bench_r": round(m.bench_r[i] * 100, 2),
                        "n_scored": len(sc), "n_long": sum(c["side"] > 0 for c in cands), "n_short": sum(c["side"] < 0 for c in cands),
                        "top_long": top(1), "top_short": top(-1)})
        state["log"].append(day_log)
    state["last_day"] = d


def k_atr(cfg, p):
    return cfg["stops"]["atr_mult"] * p["atr0"]


def fmt_px(x):
    return f"{x:.4f}" if x < 10 else f"{x:.2f}"


def size_order(cfg: dict, m: Market, c: dict, nav: float) -> dict:
    s, side, x = c["sym"], c["side"], c["x"]
    kind = m.universe[s]["kind"]
    tgt, mx = cfg["sizing"]["target_atr_pct"], cfg["sizing"]["max_leverage"]
    cert = kind in CERT_KINDS or side < 0
    atr_pct = max(x["atr_pct"], 0.05)
    if cert:
        lev = round(clip(tgt / atr_pct, 1.0, mx), 1)
        shrink = min(1.0, tgt / (atr_pct * lev))
    else:
        lev, shrink = 1.0, min(1.0, tgt / atr_pct)
    N = nav * cfg["limits"]["max_position_pct"] / 100 * shrink
    comp = x["comp"]
    w = cfg["signals"]["weights_stock"]
    fa_part = sum(w.get(k, 0) * comp.get(k, 0) for k in FA_KEYS) * side
    ta_part = sum(w.get(k, 0) * comp.get(k, 0) for k in ("momentum", "trend")) * side
    if kind == "stock" and "earnings" in w and abs(comp.get("earnings", 0)) >= 0.25:
        typ = "rapport"
    elif kind == "stock" and fa_part > ta_part:
        typ = "fundamenta"
    elif kind == "stock":
        typ = "momentum"
    else:
        typ = "index" if kind == "index" else "valuta"
    parts = [f"poäng {c['score']:.0f}"]
    if x["event"]:
        e = x["event"]
        parts.append(f"rapport {e['date']}: överraskning {e['surprise']:+.1f} %, kursreaktion {e['reaction']:+.1f} % mot Nasdaq")
    fa = x.get("fa") or {}
    if fa.get("eps_g_pct") is not None:
        parts.append(f"vinst/aktie 12 mån {fa['eps_g_pct']:+.0f} % mot året innan")
    if fa.get("avg_surprise") is not None:
        parts.append(f"snittöverraskning 4 kv {fa['avg_surprise']:+.1f} %")
    if fa.get("ups") or fa.get("downs"):
        parts.append(f"analytiker {cfg['signals']['revision_days']} d: {fa.get('ups', 0)} upp, {fa.get('downs', 0)} ned")
    if fa.get("pe"):
        parts.append(f"P/E {fa['pe']:.0f}")
    if x.get("rr"):
        r = x["rr"]
        parts.append(f"risk/reward {r['rr']:.1f} ({'riktkurs' if r['src'] == 'analyst' else 'årshögsta'} {fmt_px(r['ref'])})")
    parts.append(f"momentum 3 mån {x['mom_pct']:+.1f} %")
    parts.append({1: "över 50 och 200 dagars snitt", -1: "under 50 och 200 dagars snitt"}.get(round(comp["trend"]), f"trend {comp['trend']:+.2f}"))
    if "quality" in comp:
        parts.append(f"nyckeltal {comp['quality']:+.2f}")
    vehicle = "aktien" if not cert else (f"mini short ×{lev:g}" if kind == "stock" else f"{'bull' if side > 0 else 'bear'}/mini ×{lev:g}")
    return {"action": "open", "sym": s, "side": side, "kind": kind, "cert": cert, "lev": lev, "N": N, "atr": x["atr_pct"] / 100 * x["close"],
            "score": c["score"], "type": typ, "comp": comp, "rr": (x.get("rr") or {}).get("rr"), "why": f"{'Köp' if side > 0 else 'Blanka'} via {vehicle}: " + "; ".join(parts)}


# ---------------------------------------------------------------- runs
def run_backtest(m: Market, cfg: dict, start_i: int) -> dict:
    st = new_state(cfg, m.days[start_i])
    st["nav"] = []
    step(st, m, start_i, cfg, keep_log=False, first=True)
    for i in range(start_i + 1, len(m.days)):
        step(st, m, i, cfg, keep_log=False)
    return summarize(st, m.days[start_i], m.days[-1], backtest=True)


def summarize(st: dict, start: str, end: str, backtest: bool = False) -> dict:
    nav = st["nav"]
    rets = [nav[k]["nav"] / nav[k - 1]["nav"] - 1 for k in range(1, len(nav))]
    brets = [nav[k]["bench"] / nav[k - 1]["bench"] - 1 for k in range(1, len(nav))]
    n = len(rets)
    years = max(n / 252, 1e-9)

    def stats(r, series):
        tot = series[-1] / series[0] - 1
        sd = (sum((x - sum(r) / len(r)) ** 2 for x in r) / max(len(r) - 1, 1)) ** 0.5 if r else 0
        peak, mdd = series[0], 0.0
        for v in series:
            peak = max(peak, v)
            mdd = min(mdd, v / peak - 1)
        return {"total_pct": round(tot * 100, 2), "cagr_pct": round(((1 + tot) ** (1 / years) - 1) * 100, 2) if n > 20 else None,
                "vol_pct": round(sd * math.sqrt(252) * 100, 1), "mdd_pct": round(mdd * 100, 1),
                "sharpe": round(sum(r) / len(r) / sd * math.sqrt(252), 2) if r and sd else None}
    tr = st["trades"]
    by_type = {}
    for t in tr:
        b = by_type.setdefault(t["type"], {"n": 0, "win": 0, "pnl": 0, "ret": 0.0})
        b["n"] += 1
        b["win"] += t["pnl"] > 0
        b["pnl"] += t["pnl"]
        b["ret"] += t["ret_pct"]
    for b in by_type.values():
        b["avg_ret_pct"] = round(b.pop("ret") / b["n"], 2)
        b["win_pct"] = round(b["win"] / b["n"] * 100)
    act = [a - b for a, b in zip(rets, brets)]
    sd = (sum((x - sum(act) / len(act)) ** 2 for x in act) / max(len(act) - 1, 1)) ** 0.5 if act else 0
    out = {"start": start, "end": end, "days": n, "portfolio": stats(rets, [x["nav"] for x in nav]),
           "benchmark": stats(brets, [x["bench"] for x in nav]),
           "excess_pct": round((nav[-1]["nav"] / nav[0]["nav"] - nav[-1]["bench"] / nav[0]["bench"]) * 100, 2),
           "tracking_t": round(sum(act) / len(act) / sd * math.sqrt(len(act)), 2) if act and sd else None,
           "trades": len(tr), "win_pct": round(sum(t["pnl"] > 0 for t in tr) / len(tr) * 100) if tr else None,
           "avg_ret_pct": round(sum(t["ret_pct"] for t in tr) / len(tr), 2) if tr else None,
           "pnl_sek": round(sum(t["pnl"] for t in tr)), "costs_sek": round(sum(t["cost"] for t in tr)), "by_type": by_type,
           "avg_positions": round(sum(x["n"] for x in nav) / len(nav), 1)}
    if backtest:
        step_n = max(1, len(nav) // 300)
        out["curve"] = [{"d": x["d"], "nav": x["nav"], "bench": x["bench"]} for x in nav[::step_n]] + ([nav[-1]] if (len(nav) - 1) % step_n else [])
        out["curve"] = [{"d": x["d"], "nav": x["nav"], "bench": x["bench"]} for x in out["curve"]]
    return out


def completed_days(m_days: list[str], now: datetime) -> list[str]:
    """Drop today's bar while the US session is still open (it is not final)."""
    ny = now.astimezone(NY)
    today = ny.strftime("%Y-%m-%d")
    if m_days and m_days[-1] == today and (ny.hour, ny.minute) < (16, 20):
        return m_days[:-1]
    return m_days


def export(state: dict | None, cfg: dict, universe: dict, now: datetime) -> dict:
    gen = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    if not state:
        return {"part": "portfolio", "generated": gen, "status": "ej startad", "rules": cfg}
    nav = state["nav"]
    last = nav[-1]
    pos = []
    for p in state["positions"]:
        price = p.get("last", p["p0"])
        pos.append({"id": p["id"], "sym": p["sym"], "name": universe.get(p["sym"], {}).get("name", p["sym"]), "kind": p["kind"],
                    "avanza_name": universe.get(p["sym"], {}).get("avanza_name", ""),
                    "side": p["side"], "cert": p["cert"], "lev": p["lev"], "N": round(p["N"]), "p0": p["p0"], "last": price,
                    "stop": p["stop"], "d": p["d"], "held": p["held"], "why": p["why"], "type": p["type"],
                    "ret_pct": round((p["side"] * p["lev"] * (price / p["p0"] - 1)) * 100, 2)})
    day = state["log"][-1] if state["log"] else None
    today_orders = len(day["fills"]) if day else 0
    return {"part": "portfolio", "generated": gen, "status": "aktiv", "start": state["start"], "last_day": state["last_day"],
            "capital0": state["capital0"], "nav": last["nav"], "bench": last["bench"], "sleeve": round(state["sleeve"]),
            "positions": pos, "pending": [{k: (round(v) if k == "N" else v) for k, v in o.items() if k in ("action", "sym", "side", "lev", "N", "why", "type", "id")}
                                          for o in state["pending"]],
            "orders_last_day": today_orders, "curve": [{"d": x["d"], "nav": x["nav"], "bench": x["bench"]} for x in nav],
            "trades": state["trades"][-300:][::-1], "log": state["log"][-90:][::-1],
            "stats": summarize(state, state["start"], state["last_day"]) if len(nav) > 1 else None,
            "backtest": state.get("backtest"), "rules": cfg, "rules_version": state.get("rules_version"),
            "changes": state.get("changes", []), "data_errors": state.get("data_errors", {}),
            "tuning": {"entry_score": (state.get("tuning") or {}).get("entry_score"),
                       "history": [{k: v for k, v in h.items() if k != "grid"} | {"grid": h.get("grid")} for h in ((state.get("tuning") or {}).get("history") or [])[-12:][::-1]]}}


# ---------------------------------------------------------------- storage
def open_store(token: str | None, repo: str | None, tmp: Path, local: Path):
    """The portfolio branch (encrypted) in Actions; a plain local file otherwise (development)."""
    if token and repo:
        import intraday_archive as ia
        b = ia.Branch(f"https://x-access-token:{token}@github.com/{repo}.git", tmp, branch="portfolio")
        r = b.git("ls-remote", "origin", "refs/heads/portfolio", check=False)
        if r.stdout.strip() and not b.head:
            raise RuntimeError("Portföljgrenen finns men kunde inte hämtas — avbryter hellre än börjar om.")
        return b
    return local


def load_state(store, users: dict) -> dict | None:
    if isinstance(store, Path):
        return json.loads(store.read_text(encoding="utf-8")) if store.exists() else None
    import intraday_archive as ia
    raw = store.read(STATE_FILE)
    if not raw:
        return None
    return ia.unpack(ia.archive_key(store, users), raw)


def save_state(store, users: dict, state: dict, message: str) -> None:
    if isinstance(store, Path):
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(json.dumps(state), encoding="utf-8")
        return
    import intraday_archive as ia
    key = ia.archive_key(store, users)
    store.write(STATE_FILE, ia.pack(key, state))
    store.write("auth.json", ia.wrap_key(key, users))
    store.write("README.md", b"# Virtuell portfolj\n\nKrypterat tillstand for den virtuella portfoljen (pipeline/portfolio.py). Skrivs bara av GitHub Actions.\n")
    store.commit_and_push(message)


def effective(cfg: dict, state: dict | None) -> dict:
    """The rules in force: config/portfolio.json plus the automatically tuned entry score (if any)."""
    t = (state or {}).get("tuning") or {}
    if t.get("entry_score") is None:
        return cfg
    return merged(cfg, {"signals": {"entry_score": t["entry_score"]}})


def tune(state: dict, cfg: dict, universe: dict, prices: dict, fund: dict, now: datetime) -> list[str]:
    """Weekly, mechanical adjustment of the entry score within the bounds the owner approved
    (config auto_tune.entry_score). Every scored grid point is backtested on two years; the score moves
    at most one step toward the best point, only when that point beats the current one by min_gain_pp,
    and a raise is blocked when the live trades it would have excluded did better than the rest."""
    A = cfg["auto_tune"]["entry_score"]
    cur = effective(cfg, state)["signals"]["entry_score"]
    grid, x = [], A["min"]
    while x <= A["max"] + 1e-9:
        grid.append(round(x, 1))
        x += A["step"]
    res = {}
    for g in grid:
        c = merged(cfg, {"signals": {"entry_score": g}})
        m = Market(prices, universe, fund, c, use_quality=False)
        end = completed_days(m.days, now)[-1]
        start_i = next(i for i in range(len(m.days)) if i >= 210 and m.days[i] >= (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=730)).strftime("%Y-%m-%d"))
        r = run_backtest(m, c, start_i)
        res[g] = {"excess_pct": r["excess_pct"], "total_pct": r["portfolio"]["total_pct"], "trades": r["trades"],
                  "avg_ret_pct": r["avg_ret_pct"], "tracking_t": r["tracking_t"]}
    best = max(grid, key=lambda g: res[g]["excess_pct"])
    if cur not in res:  # the current value is off the grid (bounds changed): start from the nearest point
        cur = min(grid, key=lambda g: abs(g - cur))
    gain = res[best]["excess_pct"] - res[cur]["excess_pct"]
    new, why = cur, ""
    fmt = lambda g: f"{g:g} ({res[g]['excess_pct']:+.1f} %-enh. mot index, {res[g]['trades']} affärer)"
    if best == cur:
        why = f"Ingen ändring: {fmt(cur)} är bäst i backtestet."
    elif gain < A["min_gain_pp"]:
        why = f"Ingen ändring: bästa gränsen {fmt(best)} är bara {gain:.1f} %-enh. bättre än nuvarande {fmt(cur)} (kravet är {A['min_gain_pp']:g})."
    else:
        target = round(cur + A["step"] * (1 if best > cur else -1), 1)
        live = [t for t in state["trades"] if t.get("score_in") is not None]
        block = None
        if target > cur and len(live) >= A["min_live_trades"]:
            cut = [t["ret_pct"] for t in live if t["score_in"] < target]
            kept = [t["ret_pct"] for t in live if t["score_in"] >= target]
            if len(cut) >= 5 and kept and sum(cut) / len(cut) > sum(kept) / len(kept):
                block = (f"de {len(cut)} riktiga affärerna med poäng under {target:g} gick i snitt {sum(cut) / len(cut):+.2f} %, "
                         f"bättre än övriga ({sum(kept) / len(kept):+.2f} %)")
        if block:
            why = f"Ingen ändring: backtestet talar för {fmt(best)}, men {block}."
        else:
            new = target
            why = (f"{'Höjd' if new > cur else 'Sänkt'} från {cur:g} till {new:g}: i backtestet är {fmt(best)} bäst, "
                   f"{gain:.1f} %-enh. bättre än {fmt(cur)}. Flyttas ett steg i taget.")
    t = state.setdefault("tuning", {"history": []})
    t["entry_score"] = new
    t["history"].append({"d": now.strftime("%Y-%m-%d"), "from": cur, "to": new, "why": why, "grid": res})
    if new != cur:
        state.setdefault("changes", []).append({"d": state["last_day"], "from": state.get("rules_version"), "to": state.get("rules_version"),
                                                "auto": f"köpgräns {cur:g} → {new:g}"})
    return [why]


def trade(state: dict | None, cfg: dict, universe: dict, prices: dict, fund: dict, now: datetime) -> tuple[dict, list[str]]:
    m = Market(prices, universe, fund, cfg)
    days = completed_days(m.days, now)
    msgs = []
    if not days:
        return state, ["Ingen avslutad handelsdag i datan."]
    if state is None:
        # starts at the latest completed close: all capital in the index fund, first orders at the next open
        state = new_state(cfg, days[-1])
        state["nav"] = []
        step(state, m, m.pos[days[-1]], cfg, first=True)
        msgs.append(f"Portföljen startad {days[-1]} med {_sv(state['capital0'])} kr.")
    if state.get("rules_version") != cfg["version"]:
        # unfilled entries the new rules no longer allow are cancelled (exits always stay)
        def allowed(o):
            if o["action"] != "open":
                return True
            if o["side"] < 0 and not cfg["signals"].get("allow_short", True):
                return False
            return o["kind"] == "stock" or cfg["limits"].get(f"max_{o['kind']}_positions", 1) > 0
        cancelled = [o["sym"] for o in state["pending"] if not allowed(o)]
        state["pending"] = [o for o in state["pending"] if allowed(o)]
        state.setdefault("changes", []).append({"d": state["last_day"], "from": state.get("rules_version"), "to": cfg["version"],
                                                "cancelled": cancelled})
        state["rules_version"] = cfg["version"]
    todo = [d for d in days if d > state["last_day"]]
    for d in todo:
        step(state, m, m.pos[d], cfg)
        msgs.append(f"{d}: {len(state['log'][-1]['fills'])} order genomförda, {len(state['pending'])} lagda till nästa öppning.")
    bt = state.get("backtest")
    stale = not bt or bt.get("rules_version") != cfg["version"] or bt.get("run") is None or \
        bt.get("entry_score") != cfg["signals"]["entry_score"] or \
        (now - datetime.fromisoformat(bt["run"])).days >= BACKTEST_MAX_AGE_DAYS
    if stale:
        try:
            mb = Market(prices, universe, fund, cfg, use_quality=False)
            start_i = next(i for i in range(len(mb.days)) if i >= 210 and mb.days[i] >= (datetime.strptime(days[-1], "%Y-%m-%d") - timedelta(days=730)).strftime("%Y-%m-%d"))
            res = run_backtest(mb, cfg, start_i)
            res.update({"rules_version": cfg["version"], "run": now.isoformat(), "entry_score": cfg["signals"]["entry_score"]})
            state["backtest"] = res
            msgs.append(f"Backtest {res['start']} – {res['end']}: portfölj {res['portfolio']['total_pct']} %, index {res['benchmark']['total_pct']} %.")
        except Exception as e:
            msgs.append(f"::warning::Backtestet misslyckades: {e}")
    state["data_errors"] = prices.get("errors", {})
    return state, msgs


VARIANTS = [
    ("v5: utan marknadsfilter", {"signals": {"regime": None}}),
    ("v6 (nu): köp bara när Nasdaq-100 > 200-dagars snitt", {}),
]


SE_BENCH = ["XACT-OMXS30.ST", "^OMX"]  # the OMXS30 index fund with dividends, else the price index


def load_se_universe() -> dict:
    """Stockholm-listed stocks from the site's universe (config/universe.csv)."""
    out = {}
    with (ROOT / "config" / "universe.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["kind"] == "stock" and r["currency"] == "SEK" and r["exchange"] == "Stockholm":
                out[r["sym"]] = {"name": r["name"], "yahoo": r["yahoo"], "kind": "stock", "tags": "se", "currency": "SEK",
                                 "avanza_name": r.get("avanza_name") or ""}
    return out


SE_VARIANTS = [
    ("Sverige: samma regler", {}),
    ("Sverige: köpgräns 75", {"signals": {"entry_score": 75}}),
    ("Sverige: bara teknik (momentum + trend)", {"signals": {"weights_stock": {"momentum": 0.6, "trend": 0.4}}}),
]


def merged(cfg: dict, over: dict) -> dict:
    out = json.loads(json.dumps(cfg))
    for k, v in over.items():
        if isinstance(v, dict) and k != "weights_stock":
            out[k] = merged(out.get(k, {}), v)
        else:
            out[k] = v
    return out


def _ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda k: v[k])
    r = [0.0] * len(v)
    k = 0
    while k < len(order):
        j = k
        while j + 1 < len(order) and v[order[j + 1]] == v[order[k]]:
            j += 1
        for t in range(k, j + 1):
            r[order[t]] = (k + j) / 2
        k = j + 1
    return r


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 10:
        return None
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va, vb = sum((x - ma) ** 2 for x in ra), sum((y - mb) ** 2 for y in rb)
    return cov / math.sqrt(va * vb) if va and vb else None


def ic_study(m: Market, start_i: int, horizon: int) -> dict:
    """How well each score part ranks the stocks' next `horizon`-day return (rank IC, as in Hikyuu's
    MF_ICIRWeight). Measured every `horizon` days so the periods do not overlap; t = mean / sd x sqrt(n)."""
    series = {}
    for i in range(start_i, len(m.days) - horizon, horizon):
        sc = m.scores(i)
        rows = []
        for s, x in sc.items():
            if m.universe[s]["kind"] != "stock":
                continue
            ser = m.s[s]
            k = ser.idx[i]
            if k is None or k + horizon >= len(ser.own_c):
                continue
            rows.append((x, ser.own_c[k + horizon] / ser.own_c[k] - 1))
        keys = {"score"} | {k for x, _ in rows for k in x["comp"]}
        for key in keys:
            pts = [((x["score"] if key == "score" else x["comp"].get(key)), r) for x, r in rows]
            pts = [(a, r) for a, r in pts if a is not None]
            ic = spearman([a for a, _ in pts], [r for _, r in pts])
            if ic is not None:
                series.setdefault(key, []).append((m.days[i], ic))
    out = {}
    for key, v in series.items():
        ics = [x for _, x in v]
        n = len(ics)
        mu = sum(ics) / n
        sd = (sum((x - mu) ** 2 for x in ics) / max(n - 1, 1)) ** 0.5
        h = n // 2
        out[key] = {"ic": round(mu, 3), "icir": round(mu / sd, 2) if sd else None, "t": round(mu / sd * math.sqrt(n), 2) if sd else None,
                    "n": n, "pos_pct": round(sum(x > 0 for x in ics) / n * 100), "first_half": round(sum(ics[:h]) / max(h, 1), 3),
                    "second_half": round(sum(ics[h:]) / max(n - h, 1), 3)}
    return out


def research(cfg, universe, prices, fund, now, se=None) -> dict:
    """The same backtest for rule variants, so a change can be judged before it goes live."""
    res = []
    for name, over in VARIANTS:
        over = over() if callable(over) else over
        c = merged(cfg, over)
        try:
            m = Market(prices, universe, fund, c, use_quality=False)
            end = completed_days(m.days, now)[-1]
            start_i = next(i for i in range(len(m.days)) if i >= 210 and m.days[i] >= (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=730)).strftime("%Y-%m-%d"))
            r = run_backtest(m, c, start_i)
            r.pop("curve", None)
            res.append({"name": name, "over": over, **r})
            print(f"{name}: {r['portfolio']['total_pct']} % (index {r['benchmark']['total_pct']} %), t {r['tracking_t']}, "
                  f"{r['trades']} affärer, kostnad {r['costs_sek']} kr")
        except Exception as e:
            res.append({"name": name, "error": str(e)})
            print(f"{name}: fel {e}")
    if se:
        se_u, se_prices, se_fund = se
        bars = se_prices["bars"]
        bench = next((b for b in SE_BENCH if b in bars), None)
        for name, over in SE_VARIANTS:
            c = merged(cfg, over)
            c["benchmark"] = {"name": f"OMXS30 ({bench})", "parts": {bench: 1.0}, "currency": "SEK"}
            c["costs"] = dict(c["costs"], stock_pct_per_order=0.15)  # Avanza courtage, no currency exchange
            try:
                m = Market(se_prices, se_u, se_fund, c, use_quality=False)
                end = m.days[-1]
                start_i = next(i for i in range(len(m.days)) if i >= 210 and m.days[i] >= (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=730)).strftime("%Y-%m-%d"))
                r = run_backtest(m, c, start_i)
                r.pop("curve", None)
                res.append({"name": name, "over": over, "bench": bench, **r})
                print(f"{name}: {r['portfolio']['total_pct']} % (index {r['benchmark']['total_pct']} %), t {r['tracking_t']}, "
                      f"{r['trades']} affärer, kostnad {r['costs_sek']} kr")
            except Exception as e:
                res.append({"name": name, "error": str(e)})
                print(f"{name}: fel {e}")
    ic = None
    try:
        m = Market(prices, universe, fund, cfg, use_quality=False)
        end = completed_days(m.days, now)[-1]
        start_i = next(i for i in range(len(m.days)) if i >= 210 and m.days[i] >= (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=730)).strftime("%Y-%m-%d"))
        ic = ic_study(m, start_i, cfg["limits"]["hold_days"])
        for k, v in sorted(ic.items(), key=lambda kv: -(kv[1]["ic"])):
            print(f"IC {k}: {v['ic']:+.3f} (ICIR {v['icir']}, t {v['t']}, {v['pos_pct']} % positiva, halvor {v['first_half']:+.3f}/{v['second_half']:+.3f})")
    except Exception as e:
        print(f"IC-studien misslyckades: {e}")
    return {"run": now.isoformat(), "variants": res, "ic": ic, "ic_horizon": cfg["limits"]["hold_days"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data")
    ap.add_argument("--mode", choices=["trade", "export", "research", "tune"], default="export")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from encrypt_data import parse_users
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    universe = load_universe()
    users = parse_users(os.environ.get("SITE_USERS", ""))
    token, repo = os.environ.get("GH_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    now = datetime.now(timezone.utc)
    data = Path(a.data)
    data.mkdir(parents=True, exist_ok=True)
    if token and repo and not users:
        print("Portföljen hoppas över: SITE_USERS saknas.")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        store = open_store(token, repo, Path(tmp), data / "portfolio_state.json")
        state = load_state(store, users)
        if a.mode == "tune":
            if not state:
                print("Ingen portfölj att justera än.")
                return 0
            prices = fetch_prices(universe, list(cfg["benchmark"]["parts"]))
            fund_file = data / "pf_fund.json"
            cache = json.loads(fund_file.read_text(encoding="utf-8")) if fund_file.exists() else {}
            fund = fetch_fundamentals(universe, cache, now)
            fund_file.write_text(json.dumps(fund), encoding="utf-8")
            print("\n".join(tune(state, cfg, universe, prices, fund, now)))
            save_state(store, users, state, f"Portfölj: veckojustering {now:%Y-%m-%d}")
            cfg = effective(cfg, state)
        if a.mode == "research":
            prices = fetch_prices(universe, list(cfg["benchmark"]["parts"]))
            fund_file = data / "pf_fund.json"
            cache = json.loads(fund_file.read_text(encoding="utf-8")) if fund_file.exists() else {}
            fund = fetch_fundamentals(universe, cache, now)
            fund_file.write_text(json.dumps(fund), encoding="utf-8")
            n_ana = sum(1 for f in fund.values() if f.get("analysts"))
            print(f"Nyckeltal: {len(fund)} bolag, {n_ana} med analytikerhistorik, fel: {sum(1 for f in fund.values() if f.get('error'))}")
            se_u = load_se_universe() if os.environ.get("PF_RESEARCH_SE") else {}
            se_prices = fetch_prices(se_u, SE_BENCH)
            se_fund_file = data / "pf_fund_se.json"
            se_cache = json.loads(se_fund_file.read_text(encoding="utf-8")) if se_fund_file.exists() else {}
            se_fund = fetch_fundamentals(se_u, se_cache, now)
            se_fund_file.write_text(json.dumps(se_fund), encoding="utf-8")
            print(f"Sverige: {len(se_prices['bars'])} instrument med kurser, {sum(1 for f in se_fund.values() if f.get('events'))} med rapporter, "
                  f"{sum(1 for f in se_fund.values() if f.get('analysts'))} med analytikerhistorik")
            out = research(cfg, universe, prices, fund, now, se=(se_u, se_prices, se_fund) if se_u else None)
            if isinstance(store, Path):
                (data / "pf_research.json").write_text(json.dumps(out), encoding="utf-8")
            else:
                import intraday_archive as ia
                key = ia.archive_key(store, users)
                store.write("research.enc.json", ia.pack(key, out))
                store.write("auth.json", ia.wrap_key(key, users))
                store.commit_and_push(f"Portföljforskning {now:%Y-%m-%d %H:%M}Z")
            return 0
        if a.mode == "trade":
            ny = now.astimezone(NY)
            if state and state["last_day"] >= expected_last_session(ny):
                print(f"Portföljen är redan uppdaterad till {state['last_day']}.")
            else:
                prices = fetch_prices(universe, list(cfg["benchmark"]["parts"]))
                fund_file = data / "pf_fund.json"
                cache = json.loads(fund_file.read_text(encoding="utf-8")) if fund_file.exists() else {}
                fund = fetch_fundamentals(universe, cache, now)
                fund_file.write_text(json.dumps(fund), encoding="utf-8")
                before = state["last_day"] if state else None
                state, msgs = trade(state, effective(cfg, state), universe, prices, fund, now)
                print("\n".join(msgs))
                if state:
                    save_state(store, users, state, f"Portfölj {state['last_day']}" + ("" if state["last_day"] != before else " (backtest)"))
        out = export(state, effective(cfg, state), universe, now)
    (data / "portfolio.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"Portfölj: {out['status']}" + (f", {out['last_day']}, NAV {_sv(out['nav'])} kr, {len(out['positions'])} innehav" if state else ""))
    return 0


def expected_last_session(ny: datetime) -> str:
    """The latest weekday whose US session has closed (holidays make this an over-estimate, which only
    means one extra fetch that finds nothing new)."""
    d = ny if (ny.hour, ny.minute) >= (16, 20) else ny - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")


if __name__ == "__main__":
    sys.exit(main())
