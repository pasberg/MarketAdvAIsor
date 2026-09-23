"""Pick suitable leveraged products for a trade plan.

Mirrors the rules shown in the mockup:
- Intraday: bull/bear certificates with daily leverage (high and balanced tier).
- Week: mini futures whose knock-out level lies beyond the trade's stop loss,
  in an aggressive and a conservative leverage tier.
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import Direction, Product, ProductType


@dataclass
class TradePlan:
    underlying: str
    direction: Direction
    price: float       # current underlying price
    entry: float
    stop_loss: float


@dataclass
class Pick:
    product: Product
    tier: str          # e.g. "x10", "aggressive"
    gearing: float
    reason: str


def _matches(p: Product, plan: TradePlan, types: set[ProductType]) -> bool:
    return (p.type in types and p.direction is plan.direction
            and p.underlying.casefold() == plan.underlying.casefold())


def _spread_ok(p: Product, max_spread_pct: float | None) -> bool:
    if max_spread_pct is None:
        return True
    s = p.spread_pct
    return s is None or s <= max_spread_pct  # unknown spread is not a reason to drop


def select_certificates(products: list[Product], plan: TradePlan,
                        tiers: tuple[int, ...] = (10, 5),
                        max_spread_pct: float | None = 0.3) -> list[Pick]:
    """One bull/bear certificate per leverage tier, tightest spread first."""
    cands = [p for p in products
             if _matches(p, plan, {ProductType.BULL_BEAR}) and p.leverage
             and _spread_ok(p, max_spread_pct)]
    picks = []
    for lev in tiers:
        same = [p for p in cands if abs(p.leverage - lev) < 1e-6]
        if not same:
            continue
        best = min(same, key=lambda p: (p.spread_pct is None, p.spread_pct or 0))
        picks.append(Pick(best, f"x{lev}", float(lev), f"Daglig hävstång x{lev}"))
    return picks


def mini_future_targets(plan: TradePlan, min_financing_pct: float, risk_buffer: float = 1.6
                        ) -> tuple[float, float]:
    """Target financing level and knock-out limit for a mini future.

    The financing distance is at least `min_financing_pct` of the entry and at
    least `risk_buffer` times the entry→stop distance, so the knock-out is never
    hit before the plan's own stop loss.
    Returns (financing_level_limit, knockout_limit).
    """
    risk = abs(plan.entry - plan.stop_loss)
    dist = max(risk * risk_buffer, plan.entry * min_financing_pct)
    sign = 1 if plan.direction is Direction.LONG else -1
    fin = plan.entry - sign * dist
    return fin, plan.stop_loss


def select_mini_futures(products: list[Product], plan: TradePlan,
                        tiers: dict[str, float] | None = None,
                        max_spread_pct: float | None = 0.5) -> list[Pick]:
    """Pick one mini future (or turbo) per tier.

    `tiers` maps a tier name to the minimum financing distance as a share of the
    entry price, e.g. {"aggressive": 0.12, "conservative": 0.20}.
    A product qualifies when its financing level is at or beyond the target and
    its knock-out level lies beyond the stop loss. Among qualifying products the
    one with the highest gearing (closest to the target) wins.
    """
    tiers = tiers or {"aggressive": 0.12, "conservative": 0.20}
    long = plan.direction is Direction.LONG
    cands = [p for p in products
             if _matches(p, plan, {ProductType.MINI_FUTURE, ProductType.TURBO})
             and p.financing_level is not None and _spread_ok(p, max_spread_pct)]
    picks, used = [], set()
    for name, pct in tiers.items():
        fin_limit, ko_limit = mini_future_targets(plan, pct)
        ok = []
        for p in cands:
            if p.isin in used:
                continue
            ko = p.knockout_level if p.knockout_level is not None else p.financing_level
            fin_ok = p.financing_level <= fin_limit if long else p.financing_level >= fin_limit
            ko_ok = ko < ko_limit if long else ko > ko_limit
            g = p.gearing(plan.price)
            if fin_ok and ko_ok and g:
                ok.append((g, p))
        if not ok:
            continue
        g, best = max(ok, key=lambda t: t[0])
        used.add(best.isin)
        picks.append(Pick(best, name, g,
                          f"Finansieringsnivå {best.financing_level:.2f}, knock-out bortom SL {plan.stop_loss:.2f}"))
    return picks
