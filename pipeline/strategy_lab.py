"""Strategy lab: test established trading strategies on the fetched history, honestly.

Every strategy family has a small parameter grid. Strategies are long-only unless marked
`short=True` (then they also short, paying a yearly short cost). All trade at the close after
the signal (no look-ahead) and pay a cost on every position change.
Results are measured three ways:

- train / test: best parameters picked on the first 60 % of the period, reported on the last 40 %
- walk-forward: parameters (and, for the champion, the strategy family) are re-chosen every
  step using only data up to that point, then traded on the next step. The joined steps
  are an out-of-sample track record of the selection process itself.
- robustness: share of a family's parameter sets that were profitable in the test period.

The equal-weight portfolio of all instruments (buy and hold) is the benchmark.

The lab runs twice: on daily bars (about four years, all families) and on weekly bars (ten
years, the long-horizon families in WEEKLY_FAMILIES with grids in weeks), so bear markets
like 2018, 2020 and 2022 are part of the weekly test.

Adding a strategy: write a function (df, **params) -> position Series (0/1, decided at the
close) and add it to FAMILIES with a grid. See pipeline/lab/BACKLOG.md.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

YEAR = 252  # daily bars per year
WEEKS = 52  # weekly bars per year
COST_PCT = 0.15  # % of the position, round trip (spread + courtage)
SHORT_COST_PCT_YEAR = 3.0  # yearly cost of holding a short (bear certificate / mini short financing, borrow fee)


# ---------- indicators ----------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    pc = df.c.shift(1)
    tr = pd.concat([df.h - df.l, (df.h - pc).abs(), (df.l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def hold(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """Position from entry/exit conditions: enter on entry, stay until exit."""
    e, x = entry.fillna(False).to_numpy(), exit_.fillna(False).to_numpy()
    pos = np.zeros(len(e))
    for i in range(len(e)):
        prev = pos[i - 1] if i else 0
        pos[i] = 1 if (prev == 0 and e[i]) else (0 if (prev == 1 and x[i]) else prev)
    return pd.Series(pos, index=entry.index)


def hold_ls(long_in, long_out, short_in, short_out) -> pd.Series:
    """Long/short position from entry/exit conditions (1 long, -1 short, 0 flat)."""
    li, lo, si, so = (x.fillna(False).to_numpy() for x in (long_in, long_out, short_in, short_out))
    pos = np.zeros(len(li))
    for i in range(len(li)):
        prev = pos[i - 1] if i else 0
        if prev == 1 and lo[i]:
            prev = 0
        elif prev == -1 and so[i]:
            prev = 0
        if prev == 0:
            prev = 1 if li[i] else (-1 if si[i] else 0)
        pos[i] = prev
    return pd.Series(pos, index=long_in.index)


# ---------- strategies: (df with o,h,l,c) -> position at the close (1 long, 0 flat, -1 short) ----------
def buy_hold(df):
    return pd.Series(1.0, index=df.index)


def price_sma(df, n):
    return (df.c > sma(df.c, n)).astype(float)


def sma_cross(df, fast, slow):
    return (sma(df.c, fast) > sma(df.c, slow)).astype(float)


def donchian(df, n, m):
    return hold(df.c >= df.h.rolling(n).max().shift(1), df.c <= df.l.rolling(m).min().shift(1))


def tsmom(df, lookback):
    return (df.c / df.c.shift(lookback) - 1 > 0).astype(float)


def macd(df, trend_filter):
    m = ema(df.c, 12) - ema(df.c, 26)
    pos = m > ema(m, 9)
    if trend_filter:
        pos &= df.c > sma(df.c, 200)
    return pos.astype(float)


def rsi2(df, threshold):
    trend = df.c > sma(df.c, 200)
    return hold((rsi(df.c, 2) < threshold) & trend, df.c > sma(df.c, 5))


def bollinger(df, k):
    mid, sd = sma(df.c, 20), df.c.rolling(20).std()
    return hold((df.c < mid - k * sd) & (df.c > sma(df.c, 200)), df.c > mid)


def breakout_52w(df, exit_n):
    return hold(df.c >= df.h.rolling(YEAR).max().shift(1), df.c < sma(df.c, exit_n))


def chandelier(df, mult):
    a, hh = atr(df, 22), df.h.rolling(22).max()
    return hold((df.c > ema(df.c, 50)) & (df.c >= df.h.rolling(20).max().shift(1)), df.c < hh - mult * a)


# long/short versions: short when the long rule says "out"
def price_sma_ls(df, n):
    return np.sign(df.c - sma(df.c, n)).fillna(0.0)


def sma_cross_ls(df, fast, slow):
    return np.sign(sma(df.c, fast) - sma(df.c, slow)).fillna(0.0)


def tsmom_ls(df, lookback):
    return np.sign(df.c / df.c.shift(lookback) - 1).fillna(0.0)


def donchian_ls(df, n, m):
    hi_n, lo_n = df.h.rolling(n).max().shift(1), df.l.rolling(n).min().shift(1)
    hi_m, lo_m = df.h.rolling(m).max().shift(1), df.l.rolling(m).min().shift(1)
    return hold_ls(df.c >= hi_n, df.c <= lo_m, df.c <= lo_n, df.c >= hi_m)


def rsi2_ls(df, threshold):
    r, trend, s5 = rsi(df.c, 2), sma(df.c, 200), sma(df.c, 5)
    return hold_ls((r < threshold) & (df.c > trend), df.c > s5, (r > 100 - threshold) & (df.c < trend), df.c < s5)


# ---------- rotation strategies: (dict of all prices) -> DataFrame of positions ----------
STOCKS_ONLY = {"OMXS30", "SPX", "NDX100", "GOLD", "SILVER", "COPPER", "BRENT"}


def closes(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({k: v.c for k, v in prices.items()}).sort_index().ffill()


def xs_momentum(prices, lookback, top):
    """Each month hold the `top` stocks with the best return over `lookback` days, if that return is positive."""
    c = closes({k: v for k, v in prices.items() if k not in STOCKS_ONLY})
    mom = c / c.shift(lookback) - 1
    month_end = c.index.to_series().groupby(c.index.to_period("M")).transform("max") == c.index.to_series()
    pos = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    for d in c.index[month_end.to_numpy()]:
        m = mom.loc[d].dropna()
        winners = m[m > 0].nlargest(top).index
        pos.loc[d] = 0.0
        pos.loc[d, winners] = 1.0
    return pos.ffill().fillna(0.0)


FAMILIES = {
    "buy_hold": dict(name="Köp och behåll", desc="Äger instrumentet hela tiden (jämförelse)", term="Lång",
                     fn=buy_hold, grid=[{}]),
    "price_sma": dict(name="Pris över medelvärde", desc="Äger när priset ligger över sitt glidande medelvärde (Faber)",
                      term="Lång", fn=price_sma, grid=[{"n": n} for n in (50, 100, 200)]),
    "sma_cross": dict(name="Medelvärdeskorsning", desc="Äger när det korta medelvärdet ligger över det långa (golden cross)",
                      term="Medel–lång", fn=sma_cross, grid=[{"fast": f, "slow": s} for f, s in ((10, 50), (20, 100), (50, 200))]),
    "donchian": dict(name="Donchian-utbrott", desc="Köper nytt N-dagarshögsta, säljer vid M-dagarslägsta (Turtle)",
                     term="Medel", fn=donchian, grid=[{"n": n, "m": m} for n, m in ((20, 10), (55, 20), (100, 50))]),
    "tsmom": dict(name="Tidsseriemomentum", desc="Äger när avkastningen över perioden är positiv",
                  term="Medel–lång", fn=tsmom, grid=[{"lookback": n} for n in (63, 126, 252)]),
    "macd": dict(name="MACD", desc="Äger när MACD ligger över signallinjen, ev. bara över 200-dagars",
                 term="Kort–medel", fn=macd, grid=[{"trend_filter": f} for f in (False, True)]),
    "rsi2": dict(name="RSI(2) rekyl", desc="Köper kraftigt översålt i upptrend, säljer över 5-dagars (Connors)",
                 term="Kort", fn=rsi2, grid=[{"threshold": t} for t in (5, 10, 20)]),
    "bollinger": dict(name="Bollinger-rekyl", desc="Köper under nedre bandet i upptrend, säljer vid mittlinjen",
                      term="Kort", fn=bollinger, grid=[{"k": k} for k in (2.0, 2.5)]),
    "breakout_52w": dict(name="52-veckorshögsta", desc="Köper nytt årshögsta, säljer under medelvärdet",
                         term="Medel–lång", fn=breakout_52w, grid=[{"exit_n": n} for n in (20, 50)]),
    "chandelier": dict(name="Trend med ATR-stop", desc="Köper utbrott i upptrend, följer med en stop 2,5–3,5 ATR under toppen",
                       term="Medel", fn=chandelier, grid=[{"mult": m} for m in (2.5, 3.5)]),
    "price_sma_ls": dict(name="Pris mot medelvärde, köp/blanka", desc="Köpt över medelvärdet, blankad under", term="Lång",
                         short=True, fn=price_sma_ls, grid=[{"n": n} for n in (50, 100, 200)]),
    "sma_cross_ls": dict(name="Medelvärdeskorsning, köp/blanka", desc="Köpt när korta medelvärdet ligger över det långa, blankad annars",
                         term="Medel–lång", short=True, fn=sma_cross_ls, grid=[{"fast": f, "slow": s} for f, s in ((20, 100), (50, 200))]),
    "tsmom_ls": dict(name="Tidsseriemomentum, köp/blanka", desc="Köpt efter uppgång, blankad efter nedgång (managed futures/CTA)",
                     term="Medel–lång", short=True, fn=tsmom_ls, grid=[{"lookback": n} for n in (63, 126, 252)]),
    "donchian_ls": dict(name="Donchian, köp/blanka", desc="Köper nytt högsta och blankar nytt lägsta (Turtle åt båda håll)",
                        term="Medel", short=True, fn=donchian_ls, grid=[{"n": n, "m": m} for n, m in ((20, 10), (55, 20))]),
    "rsi2_ls": dict(name="RSI(2), köp/blanka", desc="Köper översålt i upptrend, blankar överköpt i nedtrend (kort sikt)",
                    term="Kort", short=True, fn=rsi2_ls, grid=[{"threshold": t} for t in (5, 10)]),
    "xs_momentum": dict(name="Relativ styrka (rotation)", desc="Äger varje månad de aktier som gått bäst senaste halvåret/året, om uppgången är positiv",
                        term="Medel–lång", fn=xs_momentum, portfolio=True,
                        grid=[{"lookback": lb, "top": k} for lb in (126, 252) for k in (5, 10)]),
}


# Weekly bars, ten years: the long-horizon strategies with their grids in weeks.
# Short-term strategies tuned for daily bars (RSI(2), Bollinger, ATR stop, 52-week high) are left out.
def _weekly(key: str, grid: list[dict], **changes) -> dict:
    return {**FAMILIES[key], **changes, "grid": grid}


WEEKLY_FAMILIES = {
    "buy_hold": _weekly("buy_hold", [{}]),
    "price_sma": _weekly("price_sma", [{"n": n} for n in (20, 43)],
                         desc="Äger när priset ligger över sitt medelvärde (Faber: 10 månader ≈ 43 veckor)"),
    "sma_cross": _weekly("sma_cross", [{"fast": 4, "slow": 20}, {"fast": 10, "slow": 40}],
                         desc="Äger när det korta medelvärdet ligger över det långa (10/40 veckor ≈ 50/200 dagar)"),
    "tsmom": _weekly("tsmom", [{"lookback": n} for n in (26, 52)], desc="Äger när avkastningen senaste 6 eller 12 månaderna är positiv"),
    "donchian": _weekly("donchian", [{"n": 4, "m": 4}, {"n": 20, "m": 10}],
                        desc="Köper nytt N-veckorshögsta, säljer vid M-veckorslägsta (Donchians 4-veckorsregel)"),
    "macd": _weekly("macd", [{"trend_filter": False}], desc="Äger när MACD (12/26/9 veckor) ligger över signallinjen"),
    "xs_momentum": _weekly("xs_momentum", [{"lookback": lb, "top": k} for lb in (26, 52) for k in (5, 10)],
                           desc="Äger varje månad de aktier som gått bäst senaste halvåret/året, om uppgången är positiv"),
    "price_sma_ls": _weekly("price_sma_ls", [{"n": 43}], desc="Köpt över 43-veckorsmedelvärdet, blankad under"),
    "tsmom_ls": _weekly("tsmom_ls", [{"lookback": n} for n in (26, 52)], desc="Köpt efter uppgång, blankad efter nedgång (6 eller 12 månader)"),
}


# ---------- evaluation ----------
def load_prices(data_dir: Path, series: str = "day") -> dict[str, pd.DataFrame]:
    d = json.loads((data_dir / "daily.json").read_text(encoding="utf-8"))
    out = {}
    for sym, x in d["symbols"].items():
        s = x.get("series", {}).get(series)
        if not s or len(s["c"]) < 60:
            continue
        idx = pd.to_datetime(s["t"], unit="s")
        df = pd.DataFrame({"o": s["o"], "h": s["h"], "l": s["l"], "c": s["c"]}, index=idx, dtype=float).dropna()
        out[sym] = df[~df.index.duplicated()]
    return out


def strategy_returns(prices: dict[str, pd.DataFrame], fn, params: dict, cost_pct: float = COST_PCT, rotation: bool = False,
                     ppy: int = YEAR):
    """Daily returns per instrument (columns) and positions, trading at the close after the signal.

    Per-instrument strategies give each instrument an equal share of the capital (cash when out).
    Rotation strategies split the capital over the instruments they hold.
    """
    if rotation:
        c = closes(prices)
        pos = fn(prices, **params).reindex(index=c.index, columns=c.columns).fillna(0.0)
        held = pos.sum(axis=1).replace(0, np.nan)
        w = pos.div(held, axis=0).fillna(0.0) * c.shape[1]  # so the equal-weight mean gives 1/held each
        r = c.pct_change().fillna(0.0)
        turn = w.diff().abs().fillna(w.abs())
        return w.shift(1).fillna(0.0) * r - turn.shift(1).fillna(0.0) * cost_pct / 100 / 2, pos
    rets, poss = {}, {}
    for sym, df in prices.items():
        pos = fn(df, **params).reindex(df.index).fillna(0.0)
        r = df.c.pct_change().fillna(0.0)
        turn = pos.diff().abs().fillna(pos.abs())
        held = pos.shift(1).fillna(0.0)
        short_cost = (held < 0) * SHORT_COST_PCT_YEAR / 100 / ppy
        rets[sym] = held * r - turn.shift(1).fillna(0.0) * cost_pct / 100 / 2 - short_cost
        poss[sym] = pos
    return pd.DataFrame(rets), pd.DataFrame(poss)


def portfolio(rets: pd.DataFrame) -> pd.Series:
    """Equal weight across the instruments that have data that day."""
    return rets.mean(axis=1, skipna=True).fillna(0.0)


def metrics(r: pd.Series, pos: pd.DataFrame | None = None, ppy: int = YEAR) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {}
    eq = (1 + r).cumprod()
    years = len(r) / ppy
    cagr = eq.iloc[-1] ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    sd = r.std()
    out = {"cagr": cagr * 100, "sharpe": (r.mean() / sd * math.sqrt(ppy)) if sd > 0 else 0.0,
           "mdd": ((eq / eq.cummax()) - 1).min() * 100, "total": (eq.iloc[-1] - 1) * 100, "days": len(r)}
    if pos is not None:
        p = pos.loc[r.index]
        out["exposure"] = float(p.abs().mean().mean() * 100)
        out["short"] = float((p < 0).mean().mean() * 100)
        out["trades_per_year"] = float((p.diff().clip(lower=0).sum().sum()) / max(p.shape[1], 1) / years)
    return {k: round(float(v), 3) for k, v in out.items()}


def walk_forward(candidates: list[tuple[str, dict, pd.Series]], index: pd.DatetimeIndex, start: int, step: int, ppy: int = YEAR):
    """Re-pick the candidate with the best Sharpe on data up to each step; trade it on the next step."""
    pieces, picks = [], []
    for a in range(start, len(index), step):
        b = min(a + step, len(index))
        past = slice(index[0], index[a - 1])
        def score(r):
            x = r.loc[past].iloc[-ppy * 2:]  # the last two years before the step
            return x.mean() / x.std() if x.std() > 0 else -1e9
        key, params, r = max(candidates, key=lambda c: score(c[2]))
        pieces.append(r.iloc[a:b])
        picks.append({"from": str(index[a].date()), "family": key, "params": params})
    return (pd.concat(pieces) if pieces else pd.Series(dtype=float)), picks


def run(prices: dict[str, pd.DataFrame], cost_pct: float = COST_PCT, families: dict | None = None, ppy: int = YEAR) -> dict:
    """Test all families on one price series set. Daily bars by default; weekly with WEEKLY_FAMILIES and ppy=WEEKS."""
    families = families or FAMILIES
    M = lambda r, pos=None: metrics(r, pos, ppy)
    per_family, all_cands = {}, []
    index = None
    for key, fam in families.items():
        runs = []
        for params in fam["grid"]:
            rets, pos = strategy_returns(prices, fam["fn"], params, cost_pct, fam.get("portfolio", False), ppy)
            r = portfolio(rets)
            index = r.index if index is None else index
            runs.append((params, r, pos))
        per_family[key] = runs
        all_cands += [(key, p, r) for p, r, _ in runs]
    n = len(index)
    split = int(n * .6)
    if ppy == YEAR:  # daily: one year before the first re-selection, then every half year
        start, step = max(min(YEAR, int(n * .4)), 60), max(min(126, n // 8), 20)
    else:  # weekly: two years before the first re-selection, then every half year
        start, step = max(min(2 * ppy, int(n * .4)), 20), max(min(ppy // 2, n // 8), 4)
    train, test = slice(index[0], index[split - 1]), slice(index[split], index[-1])

    fams = []
    for key, runs in per_family.items():
        fam = families[key]
        best = max(runs, key=lambda x: M(x[1].loc[train]).get("sharpe", -9))
        test_sharpes = [M(r.loc[test]).get("sharpe", 0) for _, r, _ in runs]
        wf_r, wf_picks = walk_forward([(key, p, r) for p, r, _ in runs], index, start, step, ppy)
        pos_best = best[2]
        fams.append({
            "id": key, "name": fam["name"], "desc": fam["desc"], "term": fam["term"], "configs": len(runs),
            "best": {"params": best[0], "train": M(best[1].loc[train], pos_best), "test": M(best[1].loc[test], pos_best)},
            "robust": {"share_positive": round(sum(s > 0 for s in test_sharpes) / len(test_sharpes), 2)},
            "wf": M(wf_r), "wf_last": wf_picks[-1]["params"] if wf_picks else best[0],
        })
    bh = next(f for f in fams if f["id"] == "buy_hold")
    for f in fams:
        f["robust"]["beats_bh_test"] = f["best"]["test"].get("sharpe", -9) > bh["best"]["test"].get("sharpe", 0)
        f["robust"]["ok"] = bool(f["robust"]["beats_bh_test"] and f["robust"]["share_positive"] >= .6 and f["id"] != "buy_hold")

    # champion: the walk-forward choice among all families and parameter sets
    champ_r, champ_picks = walk_forward(all_cands, index, start, step, ppy)
    last = champ_picks[-1] if champ_picks else {"family": "buy_hold", "params": {}}
    fam = families[last["family"]]
    signals = []
    rot = fam["fn"](prices, **last["params"]) if fam.get("portfolio") else None
    for sym, df in prices.items():
        if rot is not None:
            if sym not in rot.columns:
                continue
            pos = (rot[sym] > 0).astype(int).reindex(df.index).ffill().fillna(0)
        else:
            pos = fam["fn"](df, **last["params"]).reindex(df.index).fillna(0).astype(int)
        changes = pos.ne(pos.shift())
        signals.append({"sym": sym, "pos": int(pos.iloc[-1]), "since": str(changes[changes].index[-1].date())})
    bh_r = per_family["buy_hold"][0][1]
    monthly = lambda r: [{"t": str(d.date()), "v": round(float(v), 4)} for d, v in (1 + r).cumprod().resample("ME").last().items()]
    return {
        "universe": len(prices), "cost_pct": cost_pct, "short_cost_pct_year": SHORT_COST_PCT_YEAR, "configs_tested": len(all_cands),
        "period": {"start": str(index[0].date()), "split": str(index[split].date()), "wf_start": str(index[start].date()), "end": str(index[-1].date())},
        "families": sorted(fams, key=lambda f: -f["wf"].get("sharpe", -9)),
        "bars": "week" if ppy == WEEKS else "day", "configs_per_family": {k: len(v["grid"]) for k, v in families.items()},
        "benchmark": {"train": bh["best"]["train"], "test": bh["best"]["test"], "wf": M(bh_r.iloc[start:])},
        "champion": {"family": last["family"], "name": fam["name"], "params": last["params"], "wf": M(champ_r),
                     "picks": champ_picks, "signals": sorted(signals, key=lambda s: (-abs(s["pos"]), -s["pos"], s["sym"]))},
        "curves": {"champion": monthly(champ_r), "benchmark": monthly(bh_r.iloc[start:])},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data")
    ap.add_argument("--cost", type=float, default=COST_PCT)
    a = ap.parse_args()
    data_dir = Path(a.data)
    prices = load_prices(data_dir)
    if len(prices) < 5:
        print("Strategilabb: för lite data")
        return 0
    res = run(prices, a.cost)
    weekly = load_prices(data_dir, "week")
    if len(weekly) >= 5:
        res["weekly"] = run(weekly, a.cost, WEEKLY_FAMILIES, WEEKS)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = data_dir / "lab.json"
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    history = prev.get("history", [])
    today = now[:10]
    history = [h for h in history if h["date"] != today] + [{
        "date": today, "champion": res["champion"]["family"], "params": res["champion"]["params"],
        "wf_sharpe": res["champion"]["wf"].get("sharpe"), "bh_sharpe": res["benchmark"]["wf"].get("sharpe"),
        **({"week_champion": res["weekly"]["champion"]["family"], "week_wf_sharpe": res["weekly"]["champion"]["wf"].get("sharpe"),
            "week_bh_sharpe": res["weekly"]["benchmark"]["wf"].get("sharpe")} if "weekly" in res else {})}]
    out.write_text(json.dumps({"part": "lab", "generated": now, **res, "history": history[-400:]}), encoding="utf-8")
    c = res["champion"]
    print(f"Strategilabb: {res['configs_tested']} varianter på {res['universe']} instrument; mästare: {c['name']} {c['params']}")
    if "weekly" in res:
        w = res["weekly"]
        print(f"Veckodata: {w['configs_tested']} varianter, {w['period']['start']} – {w['period']['end']}; mästare: {w['champion']['name']} {w['champion']['params']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
