from __future__ import annotations

from typing import Protocol

from ..models import Product


class SourceUnavailable(RuntimeError):
    """Raised when a source cannot be read yet (no access, unknown format, no permission)."""


class Source(Protocol):
    issuer: str

    def fetch(self) -> list[Product]: ...
