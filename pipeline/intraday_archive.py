"""Archive 5-minute bars for every instrument, one encrypted file per trading day.

Yahoo only serves 60 days of 5-minute bars, so intraday strategies cannot be tested on
more history unless it is kept. This script collects it:

- The bars come from data/hour.json (the last five sessions, fetched anyway for the site).
  While the archive holds fewer than BACKFILL_BELOW days, it fetches the 60 days Yahoo has.
- Each Stockholm trading day is one file on the `archive` branch:
  days/YYYY/MM/YYYY-MM-DD.enc.json = gzip + AES-256-GCM of
  {"date", "interval": "5m", "tz", "complete", "symbols": {SYM: {t, o, h, l, c, v}}}.
  Times are Stockholm wall clock encoded as UTC epoch seconds, as everywhere on the site.
- The archive key is random, created once and stored wrapped for every account in
  SITE_USERS (auth.json on the branch, the same scheme as the site). Every run re-wraps it
  for the current accounts, so an added account can read the archive; as long as one
  account keeps its password between runs the archive stays readable.
- A day is rewritten only while it is incomplete (the run was during the day). Normal
  commits on top of the branch keep old days untouched; the branch is fetched without
  file contents (partial clone), so a run downloads only the index and the days it merges.
- data/archive.json (published encrypted with the site) summarises the archive.

Usage: python pipeline/intraday_archive.py --data data   (needs GH_TOKEN, GITHUB_REPOSITORY, SITE_USERS)
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

sys.path.insert(0, str(Path(__file__).parent))
from encrypt_data import derive_kek, parse_users, seal, user_id  # noqa: E402
from restore_log import unseal  # noqa: E402

BRANCH = "archive"
BACKFILL_BELOW = 20  # days
LOCAL = ZoneInfo("Europe/Stockholm")
COMPLETE_AFTER = 22 * 60 + 30  # a run after 22:30 Stockholm time closes the day
FIELDS = ("t", "o", "h", "l", "c", "v")


# ---------- bars ----------
def by_day(series: dict[str, dict]) -> dict[str, dict[str, dict]]:
    """{sym: {t, o, ...}} -> {date: {sym: {t, o, ...}}} by the bar's Stockholm date."""
    out: dict[str, dict[str, dict]] = {}
    for sym, s in series.items():
        for i, t in enumerate(s["t"]):
            day = datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")
            d = out.setdefault(day, {}).setdefault(sym, {k: [] for k in FIELDS})
            for k in FIELDS:
                d[k].append(s[k][i])
    return out


def merge_bars(old: dict | None, new: dict) -> dict:
    """Union of two bar sets by time; the newer value wins for a time present in both."""
    rows = {}
    for s in (old, new):
        if s:
            for i, t in enumerate(s["t"]):
                rows[t] = tuple(s[k][i] for k in FIELDS)
    ts = sorted(rows)
    return {k: [rows[t][j] for t in ts] for j, k in enumerate(FIELDS)}


def merge_day(old: dict | None, new: dict[str, dict]) -> dict[str, dict]:
    syms = dict(old or {})
    for sym, s in new.items():
        syms[sym] = merge_bars(syms.get(sym), s)
    return syms


def bars_from_site(data_dir: Path) -> dict[str, dict]:
    """The 5-day 5-minute history the site already fetched (data/hour.json, series intra5)."""
    f = data_dir / "hour.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text(encoding="utf-8"))
    return {sym: x["series"]["intra5"] for sym, x in d["symbols"].items() if x.get("series", {}).get("intra5")}


def bars_backfill() -> dict[str, dict]:
    """The 60 days of 5-minute bars Yahoo keeps, for every instrument in the universe."""
    import yfinance as yf

    from fetch_market_data import SESSION, SYMBOLS, UNIVERSE, _download, frame_to_series

    frames = _download(yf, [y for y, _, _ in SYMBOLS.values()], "5m", "60d")
    out = {}
    for sym, (ysym, _, _) in SYMBOLS.items():
        kind = UNIVERSE[sym]["kind"]
        s = frame_to_series(frames.get(ysym), "5m", 100_000, session=SESSION.get(kind), fine=kind == "fx")
        if s:
            out[sym] = s
    return out


# ---------- the archive branch ----------
class Branch:
    """The archive branch, fetched without file contents; files are read one by one on demand."""

    def __init__(self, url: str, workdir: Path):
        self.url, self.dir = url, workdir
        workdir.mkdir(parents=True, exist_ok=True)
        self.git("init", "-q")
        self.git("config", "user.name", "github-actions[bot]")
        self.git("config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
        self.git("remote", "add", "origin", url)
        self.git("config", "core.repositoryformatversion", "1")
        self.git("config", "extensions.partialClone", "origin")
        self.git("config", "remote.origin.promisor", "true")
        self.git("config", "remote.origin.partialclonefilter", "blob:none")
        r = self.git("fetch", "-q", "--filter=blob:none", "--depth=1", "origin", BRANCH, check=False)
        self.head = self.git("rev-parse", "FETCH_HEAD").strip() if r.returncode == 0 else None
        self.changes: dict[str, bytes] = {}

    def git(self, *args, check=True, input: bytes | None = None, text=True):
        r = subprocess.run(["git", *args], cwd=self.dir, capture_output=True, input=input,
                           text=text and input is None)
        if check and r.returncode:
            raise RuntimeError(f"git {args[0]}: {r.stderr}")
        return r if not check else r.stdout

    def read(self, path: str) -> bytes | None:
        if not self.head:
            return None
        r = subprocess.run(["git", "show", f"{self.head}:{path}"], cwd=self.dir, capture_output=True)
        return r.stdout if r.returncode == 0 else None

    def write(self, path: str, data: bytes):
        self.changes[path] = data

    def commit_and_push(self, message: str) -> bool:
        if not self.changes:
            return False
        env = dict(os.environ, GIT_INDEX_FILE=str(self.dir / "archive.index"))
        run = lambda *a, **kw: subprocess.run(["git", *a], cwd=self.dir, env=env, capture_output=True, check=True, **kw)
        if self.head:
            run("read-tree", self.head)
        for path, data in self.changes.items():
            blob = run("hash-object", "-w", "--stdin", input=data).stdout.decode().strip()
            run("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}")
        tree = run("write-tree").stdout.decode().strip()
        parent = ["-p", self.head] if self.head else []
        commit = run("commit-tree", tree, *parent, "-m", message).stdout.decode().strip()
        run("push", "-q", "origin", f"{commit}:refs/heads/{BRANCH}")
        return True


def archive_key(branch: Branch, users: dict[str, str]) -> bytes:
    """Unwrap the archive key with any current account, or create it for a new archive."""
    raw = branch.read("auth.json")
    if raw:
        auth = json.loads(raw)
        for user, password in users.items():
            box = auth["users"].get(user_id(user))
            if box:
                try:
                    return AESGCM(derive_kek(user, password, auth["iterations"])).decrypt(
                        base64.b64decode(box["iv"]), base64.b64decode(box["ct"]), None)
                except Exception:  # this account's password changed: try the next one
                    continue
        raise RuntimeError("Inget konto i SITE_USERS kan låsa upp arkivet (alla lösenord ändrade?)")
    return AESGCM.generate_key(bit_length=256)


def wrap_key(key: bytes, users: dict[str, str]) -> bytes:
    from encrypt_data import ITERATIONS, SALT_PREFIX
    auth = {"v": 1, "kdf": "PBKDF2-SHA256", "iterations": ITERATIONS, "saltPrefix": SALT_PREFIX.decode(),
            "users": {user_id(u): seal(derive_kek(u, p), key) for u, p in users.items()}}
    return json.dumps(auth).encode()


def pack(key: bytes, obj) -> bytes:
    return json.dumps({"v": 2, "z": "gzip", **seal(key, gzip.compress(json.dumps(obj, separators=(",", ":")).encode(), 9, mtime=0))}).encode()


def unpack(key: bytes, raw: bytes):
    return json.loads(unseal(key, json.loads(raw)))


def day_path(day: str) -> str:
    return f"days/{day[:4]}/{day[5:7]}/{day}.enc.json"


def update(branch: Branch, users: dict[str, str], series: dict[str, dict], now: datetime) -> dict:
    key = archive_key(branch, users)
    index = json.loads(branch.read("index.json") or b'{"days": {}}')
    local = now.astimezone(LOCAL)
    today, late = local.strftime("%Y-%m-%d"), local.hour * 60 + local.minute >= COMPLETE_AFTER
    written = []
    for day, syms in sorted(by_day(series).items()):
        known = index["days"].get(day)
        if known and known.get("complete"):
            continue  # a finished day is never rewritten
        old = unpack(key, branch.read(day_path(day)))["symbols"] if known else None
        merged = merge_day(old, syms)
        complete = day < today or late
        branch.write(day_path(day), pack(key, {"date": day, "interval": "5m", "tz": "Europe/Stockholm",
                                               "complete": complete, "symbols": merged}))
        index["days"][day] = {"complete": complete, "symbols": len(merged), "bars": sum(len(s["t"]) for s in merged.values())}
        written.append(day)
    index["updated"] = now.isoformat(timespec="seconds")
    branch.write("index.json", json.dumps(index, indent=0, sort_keys=True).encode())
    branch.write("auth.json", wrap_key(key, users))
    branch.write("README.md", README.encode())
    return {"index": index, "written": written}


def status(index: dict, now: datetime) -> dict:
    days = sorted(index["days"])
    return {"part": "archive", "generated": now.isoformat(timespec="seconds"), "days": len(days),
            "first": days[0] if days else None, "last": days[-1] if days else None,
            "symbols": max((d["symbols"] for d in index["days"].values()), default=0),
            "bars": sum(d["bars"] for d in index["days"].values())}


README = """# Intradagsarkiv

5-minutersstaplar för alla instrument på sajten, en krypterad fil per handelsdag
(`days/ÅÅÅÅ/MM/ÅÅÅÅ-MM-DD.enc.json`). Skrivs av `pipeline/intraday_archive.py` två gånger per
handelsdag. Nyckeln finns i `auth.json`, inlindad per konto i SITE_USERS. Ändra inte grenen för hand.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    users = parse_users(os.environ.get("SITE_USERS", ""))
    token, repo = os.environ.get("GH_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    if not users or not token or not repo:
        print("Arkivet hoppas över: SITE_USERS, GH_TOKEN eller GITHUB_REPOSITORY saknas.")
        return 0
    now = datetime.now(timezone.utc)
    with tempfile.TemporaryDirectory() as tmp:
        branch = Branch(f"https://x-access-token:{token}@github.com/{repo}.git", Path(tmp))
        index = json.loads(branch.read("index.json") or b'{"days": {}}')
        series = bars_from_site(Path(a.data))
        if len(index["days"]) < BACKFILL_BELOW:
            print("Arkivet är nytt eller kort — hämtar Yahoos 60 dagar med 5-minutersdata.")
            try:
                back = bars_backfill()
                series = {s: merge_bars(back.get(s), series.get(s) or back.get(s)) for s in set(back) | set(series)}
            except Exception as e:
                print(f"::warning::Bakåtfyllningen misslyckades: {e}")
        if not series:
            print("Ingen 5-minutersdata att arkivera.")
            return 0
        res = update(branch, users, series, now)
        branch.commit_and_push(f"Arkiv {now:%Y-%m-%d %H:%M}Z: {', '.join(res['written']) or 'inga nya dagar'}")
    st = status(res["index"], now)
    Path(a.data, "archive.json").write_text(json.dumps(st), encoding="utf-8")
    print(f"Arkiv: {st['days']} handelsdagar ({st['first']} – {st['last']}), {st['symbols']} instrument, "
          f"{st['bars']} staplar; skrev {len(res['written'])} dagar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
