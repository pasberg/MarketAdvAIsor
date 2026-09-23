"""Issuer adapters. Each exposes `fetch() -> list[Product]`."""
from .base import Source, SourceUnavailable
from .morganstanley import MorganStanleySource
from .vontobel import VontobelSource

__all__ = ["Source", "SourceUnavailable", "VontobelSource", "MorganStanleySource"]
