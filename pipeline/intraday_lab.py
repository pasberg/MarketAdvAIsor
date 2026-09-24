"""Intraday strategy lab: established day-trading rules tested on the 5-minute archive.

Every trade opens and closes within one session (Nordic 09:00–17:25, US 15:30–22:00,
currencies 08:00–22:00 Stockholm time); commodities are left out (their bars run around the clock).
A signal uses only bars up to its own close; the trade is entered at the next bar's open.
A stop is filled at the stop level, or at the open if the bar opened beyond it.

Rules (at most two settings each, standard values from the sources):
  orb        Opening range breakout (Crabel 1990; Zarattini & Aziz 2023): the first 15/30 minutes
             set a range; the first close outside it opens a trade that way, stop at the other side.
  momentum   Intraday momentum (Gao, Han, Li & Zhou 2018): the first half hour's direction is
             traded in the last half hour, when the move was larger than 0 / 0.25 %.
  gap        Gap fade: an opening gap of more than 0.5 / 1 % against the previous close is faded
             from the second bar, target the previous close, stop one gap further away.
  vwap       VWAP reversion: from one hour after the open until one hour before the close, a close
             more than 1.5 / 2 standard deviations from the session VWAP is traded back towards it;
             exit at the VWAP or the close.
  open_close Benchmark: long every session from the first bar's open to the last bar's close.

Costs per trade (round trip, % of price): 0.03 (shares, low courtage), 0.08 (the site's default),
0.15 (a certificate's spread). The setting is chosen on the first 60 % of the days (training) and
reported on the last 40 % (control).

Usage: python pipeline/intraday_lab.py --data data   (reads data/archive_bars.json, writes data/intralab.json)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fetch_market_data import UNIVERSE  # noqa: E402

COSTS = (0.03, 0.08, 0.15)
BASE_COST = 0.08
TRAIN = 0.6
MARKET = {"Stockholm": "Norden", "Köpenhamn": "Norden", "Oslo": "Norden", "Helsingfors": "Norden",
          "Nasdaq": "USA", "NYSE": "USA", "Index Sthlm": "Index", "Index USA": "Index", "Valuta": "Valutor"}


# ---------- one session: arrays of open/high/low/close/volume, minute of day ----------
def exit_with_stop(d, s: int, start: int, stop: float | None, target: float | None = None, end: int | None = None):
    """Follow a trade from bar `start` (entered at its open). Returns (exit price, exit bar)."""
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    end = len(o) - 1 if end is None else end
    for k in range(start, end + 1):
        # the stop is checked first when both are touched in one bar (conservative)
        if stop is not None and (l[k] <= stop if s > 0 else h[k] >= stop):
            return (min(stop, o[k]) if s > 0 else max(stop, o[k])), k
        if target is not None and (h[k] >= target if s > 0 else l[k] <= target):
            return (max(target, o[k]) if s > 0 else min(target, o[k])), k
    return c[end], end


def orb(d, minutes: int):
    k = minutes // 5
    if len(d["o"]) < k + 3:
        return None
    hi, lo = d["h"][:k].max(), d["l"][:k].min()
    for i in range(k, len(d["o"]) - 1):
        s = 1 if d["c"][i] > hi else -1 if d["c"][i] < lo else 0
        if s:
            entry = d["o"][i + 1]
            px, _ = exit_with_stop(d, s, i + 1, lo if s > 0 else hi)
            return s, entry, px
    return None


def momentum(d, threshold: float):
    n = len(d["o"])
    if n < 14:
        return None
    first = d["c"][5] / d["o"][0] - 1
    if abs(first) * 100 <= threshold:
        return None
    s = 1 if first > 0 else -1
    i = n - 6  # the last half hour (six 5-minute bars)
    return s, d["o"][i], d["c"][-1], {"size": abs(first) * 100}


def gap(d, g: float):
    prev = d.get("prev")
    if prev is None or len(d["o"]) < 3:
        return None
    gp = d["o"][0] / prev - 1
    if abs(gp) * 100 <= g:
        return None
    s = -1 if gp > 0 else 1
    entry = d["o"][1]
    if s * (prev - entry) <= 0:  # already filled during the first bar
        return None
    stop = entry * (1 + gp)  # one gap further away from the previous close
    px, _ = exit_with_stop(d, s, 1, stop, target=prev)
    return s, entry, px, {"size": abs(gp) * 100}


def vwap_revert(d, k: float):
    o, h, l, c, v = d["o"], d["h"], d["l"], d["c"], d["v"]
    n = len(o)
    if n < 30:
        return None
    tp = (h + l + c) / 3
    w = np.where(v > 0, v, 1.0)
    vwap = np.cumsum(tp * w) / np.cumsum(w)
    dev = c - vwap
    for i in range(12, n - 13):
        sd = dev[: i + 1].std()
        if sd <= 0:
            continue
        s = 1 if dev[i] < -k * sd else -1 if dev[i] > k * sd else 0
        if s:
            entry = o[i + 1]
            for j in range(i + 1, n):
                if s * (c[j] - vwap[j]) >= 0:
                    return s, entry, c[j]
            return s, entry, c[-1]
    return None


def open_close(d):
    return (1, d["o"][0], d["c"][-1]) if len(d["o"]) >= 3 else None


FAMILIES = {
    "orb": dict(name="Utbrott ur öppningsintervallet", fn=orb, grid=[{"minutes": 15}, {"minutes": 30}],
                desc="Första 15/30 minuterna sätter ett intervall; första stängning utanför ger affär åt det hållet, stop på andra sidan (Crabel; Zarattini & Aziz)"),
    "momentum": dict(name="Intradagsmomentum", fn=momentum, grid=[{"threshold": 0.0}, {"threshold": 0.25}],
                     desc="Första halvtimmens riktning handlas under sista halvtimmen (Gao, Han, Li & Zhou)"),
    "gap": dict(name="Gap-fyllnad", fn=gap, grid=[{"g": 0.5}, {"g": 1.0}],
                desc="Öppning mer än 0,5/1 % från gårdagens stängning handlas tillbaka mot den; stop ett gap längre bort"),
    "vwap": dict(name="Rekyl mot VWAP", fn=vwap_revert, grid=[{"k": 1.5}, {"k": 2.0}],
                 desc="Mer än 1,5/2 standardavvikelser från dagens VWAP (mitt på dagen) handlas tillbaka mot VWAP"),
    "open_close": dict(name="Köp öppning, sälj stängning", fn=open_close, grid=[{}],
                       desc="Jämförelse: lång varje dag från första till sista stapeln"),
}


# ---------- data ----------
def sessions(series: dict) -> list[dict]:
    """Split one instrument's bars into sessions (Stockholm dates), with the previous close."""
    t = np.asarray(series["t"], dtype=np.int64)
    if not len(t):
        return []
    arr = {k: np.asarray([np.nan if x is None else x for x in series[k]], dtype=float) for k in ("o", "h", "l", "c", "v")}
    day = t // 86400
    out, prev = [], None
    for dd in np.unique(day):
        m = day == dd
        d = {k: a[m] for k, a in arr.items()}
        ok = ~(np.isnan(d["o"]) | np.isnan(d["h"]) | np.isnan(d["l"]) | np.isnan(d["c"]))
        d = {k: a[ok] for k, a in d.items()}
        d["v"] = np.nan_to_num(d["v"])
        if len(d["o"]) >= 3:
            d["day"], d["prev"] = int(dd), prev
            out.append(d)
            prev = float(d["c"][-1])
    return out


def run_rule(sess: dict[str, list[dict]], fn, params: dict) -> list[dict]:
    trades = []
    for sym, days in sess.items():
        for d in days:
            r = fn(d, **params)
            if r:
                s, entry, px = r[:3]
                if entry > 0:
                    trades.append({"sym": sym, "day": d["day"], "ret": s * (px / entry - 1) * 100, "s": s,
                                   **(r[3] if len(r) > 3 else {})})
    return trades


def stats(trades: list[dict], cost: float = BASE_COST) -> dict:
    n = len(trades)
    if not n:
        return {"n": 0}
    r = np.array([t["ret"] - cost for t in trades])
    daily = {}
    for t, x in zip(trades, r):
        daily.setdefault(t["day"], []).append(x)
    d = np.array([np.mean(v) for v in daily.values()])
    sd = r.std(ddof=1) if n > 1 else 0.0
    return {"n": n, "win": float((r > 0).mean()), "avg": float(r.mean()), "sum": float(r.sum()),
            "t": float(r.mean() / sd * math.sqrt(n)) if sd > 0 else None,
            "sharpe": float(d.mean() / d.std(ddof=1) * math.sqrt(252)) if len(d) > 2 and d.std(ddof=1) > 0 else None,
            "days": len(d), "long": float(np.mean([t["s"] > 0 for t in trades]))}


# how the gap and momentum rules did in different situations (all days, the widest setting: no new
# parameters are chosen from this, it only describes where the results come from)
BREAKDOWN = {
    "gap": {"dir": ("Riktning", lambda t: "Gap upp → kort (fyllnad nedåt)" if t["s"] < 0 else "Gap ned → lång (fyllnad uppåt)"),
            "size": ("Gapets storlek", lambda t: "0,5–1 %" if t["size"] < 1 else "1–2 %" if t["size"] < 2 else "Mer än 2 %")},
    "momentum": {"dir": ("Riktning", lambda t: "Upp första halvtimmen → lång" if t["s"] > 0 else "Ned första halvtimmen → kort"),
                 "size": ("Första halvtimmens rörelse", lambda t: "under 0,25 %" if t["size"] < 0.25 else "0,25–0,5 %" if t["size"] < 0.5
                          else "0,5–1 %" if t["size"] < 1 else "Mer än 1 %")},
}


def breakdown(trades: list[dict], groups: dict, market: dict) -> list[dict]:
    out = []
    by = dict(groups, market=("Marknad", lambda t: market[t["sym"]]))
    for key, (title, fn) in by.items():
        g: dict[str, list] = {}
        for t in trades:
            g.setdefault(fn(t), []).append(t)
        rows = [{"label": k, "n": len(v), "win": stats(v)["win"], "gross": float(np.mean([t["ret"] for t in v])),
                 "net": stats(v)["avg"], "t": stats(v)["t"]}
                for k, v in sorted(g.items(), key=lambda kv: (np.mean([t.get("size", 0) for t in kv[1]]) if key == "size" else 0, kv[0]))]
        out.append({"title": title, "rows": rows})
    return out


def run(bars: dict[str, dict], kinds: dict[str, dict]) -> dict:
    sess = {s: sessions(x) for s, x in bars.items() if kinds.get(s, {}).get("kind") in ("stock", "index", "fx")}
    sess = {s: v for s, v in sess.items() if v}
    days = sorted({d["day"] for v in sess.values() for d in v})
    if len(days) < 10:
        return {"part": "intralab", "days": len(days), "families": []}
    split = days[int(len(days) * TRAIN)]
    market = {s: MARKET.get(kinds[s]["exchange"], "Övrigt") for s in sess}
    fams = []
    for key, fam in FAMILIES.items():
        variants = []
        for params in fam["grid"]:
            tr = run_rule(sess, fam["fn"], params)
            train, test = [t for t in tr if t["day"] < split], [t for t in tr if t["day"] >= split]
            variants.append({"params": params, "train": stats(train), "test": stats(test), "all": stats(tr),
                             "by_cost": {str(c): stats(test, c).get("avg") for c in COSTS},
                             "by_market": {m: stats([t for t in test if market[t["sym"]] == m]) for m in sorted(set(market.values()))}})
        best = max(variants, key=lambda v: v["train"].get("avg", -1e9))
        te, trn = best["test"], best["train"]
        robust = bool(te.get("n", 0) >= 100 and trn.get("avg", -1) > 0 and te.get("avg", -1) > 0 and (te.get("t") or 0) > 2)
        entry = {"id": key, "name": fam["name"], "desc": fam["desc"], "best": best, "variants": variants, "robust": robust}
        if key in BREAKDOWN:
            wide = fam["grid"][0]
            entry["breakdown"] = {"params": wide, "groups": breakdown(run_rule(sess, fam["fn"], wide), BREAKDOWN[key], market)}
        fams.append(entry)
    iso = lambda d: datetime.fromtimestamp(d * 86400, timezone.utc).strftime("%Y-%m-%d")
    return {"part": "intralab", "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "days": len(days), "from": iso(days[0]), "to": iso(days[-1]), "split": iso(split),
            "universe": len(sess), "costs": list(COSTS), "base_cost": BASE_COST, "families": fams}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    f = Path(a.data) / "archive_bars.json"
    if not f.exists():
        print("Intradagslabbet hoppas över: data/archive_bars.json saknas (arkivet har inte körts).")
        return 0
    bars = json.loads(f.read_text(encoding="utf-8"))["symbols"]
    res = run(bars, UNIVERSE)
    (Path(a.data) / "intralab.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(f"Intradagslabb: {res['days']} dagar, {res.get('universe', 0)} instrument")
    for fam in res["families"]:
        b = fam["best"]
        te = b["test"]
        print(f"  {fam['name']:34} {json.dumps(b['params']):20} kontroll: {te.get('n', 0):5} affärer, "
              f"vinst {100 * te.get('win', 0):3.0f} %, snitt {te.get('avg', float('nan')):+.3f} % "
              f"(0,03: {b['by_cost']['0.03'] or float('nan'):+.3f}, 0,15: {b['by_cost']['0.15'] or float('nan'):+.3f}), "
              f"t {te.get('t') or float('nan'):.1f}{'  ROBUST' if fam['robust'] else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
