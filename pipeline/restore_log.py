"""Restore data/log.json from the published (encrypted) site when the Actions cache lost it.

The log only lives in the Actions cache and, encrypted, on the gh-pages branch. If the
cache was evicted, the workflow decrypts the published copy with the first account in
SITE_USERS so the history of logged picks is not lost.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from encrypt_data import derive_kek, parse_users, user_id


def _git_show(path: str) -> bytes | None:
    try:
        return subprocess.run(["git", "show", f"FETCH_HEAD:{path}"], capture_output=True, check=True).stdout
    except subprocess.CalledProcessError:
        return None


def unseal(key: bytes, box: dict) -> bytes:
    return AESGCM(key).decrypt(base64.b64decode(box["iv"]), base64.b64decode(box["ct"]), None)


def restore(auth: dict, box: dict, users: dict[str, str]) -> bytes | None:
    for user, password in users.items():
        wrapped = auth["users"].get(user_id(user))
        if wrapped:
            dek = unseal(derive_kek(user, password, auth["iterations"]), wrapped)
            return unseal(dek, box)
    return None


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/log.json")
    if out.exists():
        return 0
    users = parse_users(os.environ.get("SITE_USERS", ""))
    if not users:
        return 0
    if subprocess.run(["git", "fetch", "-q", "--depth=1", "origin", "gh-pages"]).returncode != 0:
        return 0
    auth, box = _git_show("data/auth.json"), _git_show("data/log.enc.json")
    if not auth or not box:
        print("Ingen publicerad logg att återställa.")
        return 0
    plain = restore(json.loads(auth), json.loads(box), users)
    if plain:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(plain)
        print(f"Återställde {out} från gh-pages ({len(json.loads(plain).get('entries', []))} förslag).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
