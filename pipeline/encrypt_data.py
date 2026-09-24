"""Encrypt the published data files so only logged-in users can read them.

The site is static and the repository is public, so a login screen alone would
not protect anything. Instead the data files are encrypted and the page decrypts
them in the browser after login:

- A fresh random data key (AES-256-GCM) encrypts each data file -> <name>.enc.json. The file is
  gzip-compressed first ("z": "gzip" in the box); price data shrinks to about a fifth.
- For every user in SITE_USERS a key is derived from the password with
  PBKDF2-SHA256 and used to wrap (encrypt) the data key. auth.json holds the
  wrapped keys, indexed by a SHA-256 hash of the user name (names are not published).
- The salt is derived from the user name, so the derived key stays the same across
  deploys and a browser that chose "remember me" keeps working after each update.

SITE_USERS format: one "user:password" per line (or separated by ";").
With no users an empty auth.json is written: the page stays locked.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

ITERATIONS = 250_000
SALT_PREFIX = b"marketadvaisor-v1:"
PARTS = ("intra", "hour", "daily", "log", "lab", "archive", "journal", "calib", "intralab")


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def norm_user(user: str) -> str:
    return user.strip().lower()


def user_id(user: str) -> str:
    return hashlib.sha256(norm_user(user).encode()).hexdigest()


def user_salt(user: str) -> bytes:
    return hashlib.sha256(SALT_PREFIX + norm_user(user).encode()).digest()[:16]


def derive_kek(user: str, password: str, iterations: int = ITERATIONS) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=user_salt(user), iterations=iterations)
    return kdf.derive(password.encode())


def parse_users(raw: str) -> dict[str, str]:
    users = {}
    for line in raw.replace(";", "\n").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        user, password = line.split(":", 1)
        if norm_user(user) and password:
            users[norm_user(user)] = password
    return users


def seal(key: bytes, plaintext: bytes) -> dict:
    iv = os.urandom(12)
    return {"iv": b64(iv), "ct": b64(AESGCM(key).encrypt(iv, plaintext, None))}


def encrypt_dir(src: Path, dst: Path, users: dict[str, str], iterations: int = ITERATIONS) -> list[str]:
    dst.mkdir(parents=True, exist_ok=True)
    dek = AESGCM.generate_key(bit_length=256)
    written = []
    for part in PARTS:
        f = src / f"{part}.json"
        if not f.exists():
            continue
        out = dst / f"{part}.enc.json"
        packed = gzip.compress(f.read_bytes(), compresslevel=9, mtime=0)
        out.write_text(json.dumps({"v": 2, "z": "gzip", **seal(dek, packed)}), encoding="utf-8")
        written.append(out.name)
    auth = {"v": 1, "kdf": "PBKDF2-SHA256", "iterations": iterations, "saltPrefix": SALT_PREFIX.decode(),
            "users": {user_id(u): seal(derive_kek(u, p, iterations), dek) for u, p in users.items()}}
    (dst / "auth.json").write_text(json.dumps(auth), encoding="utf-8")
    # when each part was generated, so the page only downloads parts that changed
    manifest = {}
    for part in PARTS:
        f = src / f"{part}.json"
        if f.exists():
            manifest[part] = json.loads(f.read_text(encoding="utf-8")).get("generated")
    (dst / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="data", help="katalog med intra.json, hour.json, daily.json")
    ap.add_argument("--dst", default="site/data")
    a = ap.parse_args()
    users = parse_users(os.environ.get("SITE_USERS", ""))
    written = encrypt_dir(Path(a.src), Path(a.dst), users)
    print(f"Krypterade {', '.join(written) or 'inga filer'} för {len(users)} användare -> {a.dst}")
    if not users:
        print("::warning::SITE_USERS saknas — sidan publiceras låst tills hemligheten läggs till.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
