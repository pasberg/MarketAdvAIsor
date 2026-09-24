"""Decrypt the published site data (gh-pages) to a local folder, for analysis outside the browser.

Reads the account from the environment variables MAA_USER and MAA_PASSWORD (an account that
exists in the SITE_USERS secret). The decrypted files are written to --out and must never be
committed: they are the private data the login protects.

Usage: MAA_USER=... MAA_PASSWORD=... python pipeline/read_site_data.py --out /tmp/maa-data [--parts lab,log]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from restore_log import restore  # noqa: E402

PARTS = ("intra", "hour", "daily", "log", "lab", "archive", "journal", "calib")


def git_show(path: str) -> bytes | None:
    r = subprocess.run(["git", "show", f"FETCH_HEAD:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default=",".join(PARTS))
    a = ap.parse_args()
    user, password = os.environ.get("MAA_USER", ""), os.environ.get("MAA_PASSWORD", "")
    if not user or not password:
        print("Sätt MAA_USER och MAA_PASSWORD (ett konto i SITE_USERS).")
        return 1
    if subprocess.run(["git", "fetch", "-q", "--depth=1", "origin", "gh-pages"]).returncode != 0:
        print("Kunde inte hämta gh-pages.")
        return 1
    auth = git_show("data/auth.json")
    if not auth:
        print("Ingen publicerad data.")
        return 1
    auth = json.loads(auth)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for part in [p for p in a.parts.split(",") if p]:
        box = git_show(f"data/{part}.enc.json")
        if not box:
            print(f"{part}: saknas")
            continue
        try:
            plain = restore(auth, json.loads(box), {user: password})
        except Exception:  # wrong password: the key does not decrypt
            plain = None
        if plain is None:
            print("Fel användarnamn eller lösenord.")
            return 1
        (out / f"{part}.json").write_bytes(plain)
        print(f"{part}: {len(plain) // 1024} kB -> {out / (part + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
