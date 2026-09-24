"""Copy the trading journals from the `journal` branch into the site data (data/journal.json).

The page keeps each account's journal encrypted with a key derived from that account's
password (SHA-256 of "marketadvaisor-journal-v1:" + the login key) and can sync a copy to
journal/<user id>.enc.json on the `journal` branch. This script decrypts those copies with
the passwords in SITE_USERS, so the journals are published like the rest of the data:
encrypted, readable by the site's accounts (e.g. for Claude to help interpret them).
"""
from __future__ import annotations

import base64
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


def export(files: dict[str, bytes], users: dict[str, str], iterations: int = ITERATIONS) -> dict:
    """files: {user id: encrypted box} -> {"journals": {user name: journal}}"""
    by_id = {user_id(u): (u, p) for u, p in users.items()}
    out, errors = {}, {}
    for uid, raw in files.items():
        if uid not in by_id:
            errors[uid[:8]] = "inget konto i SITE_USERS"
            continue
        user, password = by_id[uid]
        try:
            out[user] = open_box(journal_key(user, password, iterations), json.loads(raw))
        except Exception:
            errors[user] = "kunde inte dekrypteras (lösenordet ändrat efter senaste synk?)"
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
    res = export(files, users)
    out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(f"Dagbok: {len(res['journals'])} konton exporterade" + (f", fel: {res['errors']}" if res["errors"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
