"""Common product model for exchange-traded products (certificates, mini futures etc.).

Every issuer adapter maps its own data to `Product`, so the selection logic
never needs to know where a product came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ProductType(str, Enum):
    BULL_BEAR = "bull_bear"      # daily leverage, no knock-out
    MINI_FUTURE = "mini_future"  # financing level + stop-loss (knock-out) level
    TURBO = "turbo"              # knock-out at financing level
    WARRANT = "warrant"          # option with strike and expiry
    TRACKER = "tracker"


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class Product:
    isin: str
    issuer: str
    name: str
    type: ProductType
    direction: Direction
    underlying: str                      # normalised underlying name, e.g. "VOLVO B"
    currency: str = "SEK"
    leverage: float | None = None        # daily leverage for bull/bear, current gearing for minis
    financing_level: float | None = None  # in the underlying's currency
    knockout_level: float | None = None   # stop-loss / knock-out level, underlying currency
    ratio: float | None = None            # products per underlying
    bid: float | None = None
    ask: float | None = None
    venue: str | None = None              # e.g. "NGM", "Nasdaq Stockholm"
    url: str | None = None
    updated_at: datetime | None = None
    extra: dict = field(default_factory=dict)

    @property
    def spread_pct(self) -> float | None:
        if not self.bid or not self.ask or self.bid <= 0:
            return None
        mid = (self.bid + self.ask) / 2
        return (self.ask - self.bid) / mid * 100

    def gearing(self, underlying_price: float) -> float | None:
        """Effective leverage of a mini future/turbo at a given underlying price."""
        if self.financing_level is None:
            return self.leverage
        dist = underlying_price - self.financing_level
        if self.direction is Direction.SHORT:
            dist = -dist
        return underlying_price / dist if dist > 0 else None
