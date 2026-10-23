"""One-off research fetch: two years of hourly bars for commodities and indices, encrypted.

Yahoo keeps 60-minute bars for 730 days (5-minute bars only 60). The bars are converted to
Stockholm wall clock (as everywhere on the site), packed, encrypted with a fresh key wrapped for
the accounts in SITE_USERS (same scheme as the site) and written to --out:
  auth.json, hourly.enc.json
The workflow pushes --out to the `research` branch. The analysis runs elsewhere, on the decrypted copy.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: E402

from encrypt_data import parse_users, seal  # noqa: E402
from fetch_market_data import _download, frame_to_series  # noqa: E402
from intraday_archive import wrap_key  # noqa: E402

# name -> Yahoo symbol
SYMBOLS = {
    "GOLD": "GC=F", "SILVER": "SI=F", "COPPER": "HG=F", "BRENT": "BZ=F", "WTI": "CL=F", "NATGAS": "NG=F",
    "OMXS30": "^OMX", "DAX": "^GDAXI", "SPX": "^GSPC", "NDX100": "^NDX",
    "ES_FUT": "ES=F", "NQ_FUT": "NQ=F",  # US index futures: trade while Avanza's certificates do (08:00–22:00)
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="research_out")
    a = ap.parse_args()
    users = parse_users(os.environ.get("SITE_USERS", ""))
    if not users:
        print("SITE_USERS saknas — inget skrivs.")
        return 1
    import yfinance as yf
    data = {}
    for name, y in SYMBOLS.items():  # one at a time: a ticker Yahoo answers oddly must not stop the rest
        s = None
        try:
            df = _download(yf, [y], "60m", "730d").get(y)
            if df is not None and hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
                df = df.droplevel(0, axis=1) if y in df.columns.get_level_values(0) else df.droplevel(-1, axis=1)
            s = frame_to_series(df, "60m", 100_000)
        except Exception as e:
            print(f"{name:7} {y:7} fel: {e}")
        if s:
            data[name] = s
        print(f"{name:7} {y:7} {len(s['t']) if s else 0} staplar")
    key = AESGCM.generate_key(bit_length=256)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "interval": "60m",
               "tz": "Europe/Stockholm", "symbols": data}
    packed = gzip.compress(json.dumps(payload, separators=(",", ":")).encode(), 9, mtime=0)
    (out / "hourly.enc.json").write_text(json.dumps({"v": 2, "z": "gzip", **seal(key, packed)}), encoding="utf-8")
    (out / "auth.json").write_bytes(wrap_key(key, users))
    (out / "README.md").write_text("# Forskningsdata\n\nTimdata två år bakåt, krypterad (research/fetch_hourly.py).\n", encoding="utf-8")
    return 0 if data else 1


if __name__ == "__main__":
    sys.exit(main())
