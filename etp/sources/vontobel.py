"""Vontobel (products marked "VON", traded at Avanza and Nordnet).

Status: not implemented. The product search at
https://certificates.vontobel.com/SE/SV/ reportedly offers CSV/PDF export,
but the export format and the site's terms have not been checked yet —
this environment had no network access to vontobel.com.

To finish:
1. Confirm in Vontobel's terms that automated download/reuse is allowed, or
   ask Vontobel for a data feed.
2. Save one real export as tests/fixtures/vontobel_sample.csv.
3. Map its columns to `Product` in `parse()` and add a test against the fixture.
"""
from __future__ import annotations

from ..models import Product
from .base import SourceUnavailable


class VontobelSource:
    issuer = "Vontobel"

    def fetch(self) -> list[Product]:
        raise SourceUnavailable("Vontobel: exportformat och villkor ej verifierade (ingen nätverksåtkomst).")

    def parse(self, raw: str) -> list[Product]:
        raise SourceUnavailable("Vontobel: kolumnmappning saknas — behöver en riktig exportfil.")
