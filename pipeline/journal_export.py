"""Copy the trading journals from the `journal` branch into the site data (data/journal.json).

For every trade the page asks about (journal "ctxReq": instrument, time, long/short view) the
market context at entry is computed from the 5-minute archive and today's bars: opening gap,
first half hour, the day's move so far, side of the VWAP, daily trend and time since the open.
The page groups trades by these to show which situations a source trades and which ones pay.

The page keeps each account's journal encrypted with a key derived from that account's
password (SHA-256 of "marketadvaisor-journal-v1:" + the login key) and can sync a copy to
journal/<user id>.enc.json on the `journal` branch. This script decrypts those copies with
the passwords in SITE_USERS, so the journals are published like the rest of the data:
encrypted, readable by the site's accounts (e.g. for Claude to help interpret them).
"""
from __future__ import annotations

import base64
import calendar
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

sys.path.insert(0, str(Path(__file__).parent))
from encrypt_data import ITERATIONS, derive_kek, parse_users, user_id  # noqa: E402

PREFIX = b"marketadvaisor-journal-v1:"


def journal_key(user: str, password: str, iterations: int = ITERATIONS) -> bytes:
    return hashlib.sha256(PREFIX + derive_kek(user, password, iterations)).digest()


def open_box(key: bytes, box: dict) -> dict:
    if box.get("v") == 0:
        return json.loads(box["plain"])
    return json.loads(AESGCM(key).decrypt(base64.b64decode(box["iv"]), base64.b64decode(box["ct"]), None))


# ---------- market context at entry ----------
DAY = 86400


def parse_at(at: str) -> tuple[int, bool] | None:
    """'YYYY-MM-DDTHH:MM' (Stockholm time, as the page stores it) or a date -> (epoch as wall clock, has time)."""
    try:
        d, _, hm = (at or "").partition("T")
        y, m, dd = (int(x) for x in d.split("-"))
        h, mi = (int(x) for x in hm.split(":")[:2]) if hm else (0, 0)
    except ValueError:
        return None
    has_time = bool(hm) and (h, mi) != (0, 0)
    return calendar.timegm((y, m, dd, h, mi, 0)), has_time


def _ema(vals: list[float], n: int) -> float:
    k, e = 2 / (n + 1), vals[0]
    for v in vals[1:]:
        e += k * (v - e)
    return e


def context(bars: dict | None, daily: dict | None, at: str, view: str) -> dict | None:
    """The market around one entry. bars: 5-minute series {t, o, h, l, c, v}; daily: day series.
    Returns None when the data does not reach the entry yet (the next run tries again)."""
    p = parse_at(at)
    if not p:
        return None
    t, has_time = p
    s = -1 if view == "S" else 1
    day = t // DAY
    feats, nums = {}, {}
    # daily trend and the previous close, from the day series
    prev_close = None
    if daily and daily.get("t"):
        closes = [(dt // DAY, c) for dt, c in zip(daily["t"], daily["c"]) if c is not None]
        before = [c for d, c in closes if d < day]
        if len(before) >= 50:
            ema = _ema(before[-200:], 50)
            feats["trend"] = "Med dagstrenden" if s * (before[-1] - ema) > 0 else "Mot dagstrenden"
        if before:
            prev_close = before[-1]
    idx = [i for i, bt in enumerate(bars["t"]) if bt // DAY == day] if bars else []
    if idx:
        o, h, l, c, v, bt = (bars[k] for k in ("o", "h", "l", "c", "v", "t"))
        prev_idx = [i for i, x in enumerate(bt) if x // DAY < day]
        if prev_idx:
            prev_close = c[prev_idx[-1]]
        open0, t0 = o[idx[0]], bt[idx[0]]
        if prev_close:
            nums["gap"] = (open0 / prev_close - 1) * 100
        if has_time:
            if t > bt[idx[-1]] + 300:
                return None  # the day's bars do not reach the entry yet
            done = [i for i in idx if bt[i] + 300 <= t]  # bars closed before the entry
            px = c[done[-1]] if done else open0
            minutes = (t - t0) / 60
            nums["minutes"] = minutes
            nums["day"] = (px / open0 - 1) * 100
            feats["day"] = "Med dagens rörelse" if s * nums["day"] > 0 else "Mot dagens rörelse"
            if minutes >= 30 and len(idx) >= 6:
                nums["first30"] = (c[idx[5]] / open0 - 1) * 100
                feats["mom"] = "Med första halvtimmen" if s * nums["first30"] > 0 else "Mot första halvtimmen"
            else:
                feats["mom"] = "Under första halvtimmen"
            if done:
                w = [max(v[i] or 0, 1) for i in done]
                vwap = sum((h[i] + l[i] + c[i]) / 3 * wi for i, wi in zip(done, w)) / sum(w)
                nums["vwap"] = (px / vwap - 1) * 100
                feats["vwap"] = "Med VWAP (köp över, sälj under)" if s * (px - vwap) > 0 else "Mot VWAP (köp under, sälj över)"
            feats["tod"] = ("0–30 min" if minutes < 30 else "30–60 min" if minutes < 60 else "1–2 h" if minutes < 120 else "Mer än 2 h") + " efter öppning"
    elif daily and daily.get("t"):
        today = [i for i, dt in enumerate(daily["t"]) if dt // DAY == day]
        if today and prev_close:
            nums["gap"] = (daily["o"][today[0]] / prev_close - 1) * 100
        elif not today:
            return None
    if "gap" in nums:
        g = nums["gap"]
        feats["gap"] = "Inget gap (under 0,5 %)" if abs(g) < 0.5 else "Med gapet" if s * g > 0 else "Mot gapet (fyllnad)"
    return {"feats": feats, "nums": {k: round(x, 3) for k, x in nums.items()}} if feats else None


def load_market(data_dir: Path) -> tuple[dict, dict]:
    """5-minute bars (archive + the last five sessions) and daily bars per instrument."""
    sys.path.insert(0, str(Path(__file__).parent))
    from intraday_archive import merge_bars
    read = lambda n: json.loads((data_dir / n).read_text(encoding="utf-8")) if (data_dir / n).exists() else {}
    bars = read("archive_bars.json").get("symbols", {})
    for sym, x in read("hour.json").get("symbols", {}).items():
        s5 = x.get("series", {}).get("intra5")
        if s5:
            bars[sym] = merge_bars(bars.get(sym), s5)
    daily = {sym: x["series"]["day"] for sym, x in read("daily.json").get("symbols", {}).items() if x.get("series", {}).get("day")}
    return bars, daily


def add_context(journal: dict, known: dict, market) -> dict:
    """Context for the journal's requests; `known` holds contexts from earlier runs."""
    out = {}
    for req in journal.get("ctxReq") or []:
        k = req.get("k")
        if not k:
            continue
        if known.get(k):
            out[k] = known[k]
            continue
        bars, daily = market()
        sym = (req.get("und") or "").upper()
        c = context(bars.get(sym), daily.get(sym), req.get("at"), req.get("dir"))
        if c:
            out[k] = c
    return out


def export(files: dict[str, bytes], users: dict[str, str], iterations: int = ITERATIONS,
           previous: dict | None = None, market=None) -> dict:
    """files: {user id: encrypted box} -> {"journals": {user name: journal with uid and context}}"""
    by_id = {user_id(u): (u, p) for u, p in users.items()}
    out, errors = {}, {}
    prev = (previous or {}).get("journals", {})
    for uid, raw in files.items():
        if uid not in by_id:
            errors[uid[:8]] = "inget konto i SITE_USERS"
            continue
        user, password = by_id[uid]
        try:
            j = open_box(journal_key(user, password, iterations), json.loads(raw))
        except Exception:
            errors[user] = "kunde inte dekrypteras (lösenordet ändrat efter senaste synk?)"
            continue
        j["uid"] = uid
        if market:
            j["context"] = add_context(j, (prev.get(user) or {}).get("context") or {}, market)
        out[user] = j
    return {"part": "journal", "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "journals": out, "errors": errors}


def main() -> int:
    users = parse_users(os.environ.get("SITE_USERS", ""))
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/journal.json")
    if not users or subprocess.run(["git", "fetch", "-q", "--depth=1", "origin", "journal"],
                                   capture_output=True).returncode != 0:
        return 0
    ls = subprocess.run(["git", "ls-tree", "--name-only", "FETCH_HEAD", "journal/"], capture_output=True, text=True).stdout.split()
    files = {}
    for path in ls:
        if path.endswith(".enc.json"):
            files[Path(path).name[:-len(".enc.json")]] = subprocess.run(["git", "show", f"FETCH_HEAD:{path}"], capture_output=True).stdout
    if not files:
        return 0
    cache = {}

    def market():  # loaded once, and only when a trade needs a new context
        if "m" not in cache:
            cache["m"] = load_market(out.parent)
        return cache["m"]

    previous = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    res = export(files, users, previous=previous, market=market)
    out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(f"Dagbok: {len(res['journals'])} konton exporterade" + (f", fel: {res['errors']}" if res["errors"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
