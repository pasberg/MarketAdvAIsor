"""Morgan Stanley (Avanza Markets, products marked "AVA", listed on NGM).

Status: not implemented. Product pages exist per ISIN at
https://etp.morganstanley.com/se/sv/product-details/<isin>, but whether a product
list or export is offered, and under which terms, has not been checked yet —
this environment had no network access to morganstanley.com.

To finish:
1. Confirm the terms, or ask Morgan Stanley / Avanza for a data feed.
2. Save one real product list (or product page) as a fixture under tests/fixtures/.
3. Map it to `Product` in `parse()` and add a test against the fixture.
"""
from __future__ import annotations

from ..models import Product
from .base import SourceUnavailable

PRODUCT_URL = "https://etp.morganstanley.com/se/sv/product-details/{isin}"


class MorganStanleySource:
    issuer = "Morgan Stanley"

    def fetch(self) -> list[Product]:
        raise SourceUnavailable("Morgan Stanley: produktlista och villkor ej verifierade (ingen nätverksåtkomst).")

    def parse(self, raw: str) -> list[Product]:
        raise SourceUnavailable("Morgan Stanley: mappning saknas — behöver en riktig produktlista.")

    @staticmethod
    def product_url(isin: str) -> str:
        return PRODUCT_URL.format(isin=isin.lower())
