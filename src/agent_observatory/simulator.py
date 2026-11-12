"""Single-night hotel demand simulator (a clearly labelled simulator, not real hotel data).

One decision = one stay date. The hotel has `rooms_left` unsold rooms and expects `base_demand` more booking
requests between now and arrival if it charges the same as the competitor median. Guests compare prices, so demand
follows a logit curve in our price relative to that median (steeper when `sensitivity` is higher), requests arrive
as a Poisson process, and the hotel sells min(requests, rooms_left).

Because the model is known, the expected revenue of any price can be computed exactly, so every decision can be
scored against the revenue-maximising (oracle) price. That is what makes the evals objective.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Scenario:
    id: str
    days_to_arrival: int
    capacity: int
    rooms_left: int
    current_price: float
    reference_price: float  # the hotel's usual rate for this kind of night
    base_demand: float  # expected remaining booking requests when priced at the competitor median
    sensitivity: float  # slope of the logit demand curve; higher means guests switch hotels sooner
    competitor_rates: tuple[float, ...]
    floor: float
    ceiling: float
    event: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["competitor_rates"] = list(self.competitor_rates)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Scenario:
        return cls(**{**d, "competitor_rates": tuple(d["competitor_rates"])})


def expected_demand(s: Scenario, price: float) -> float:
    """Expected remaining booking requests at `price`."""
    anchor = float(np.median(s.competitor_rates))
    return 2.0 * s.base_demand / (1.0 + math.exp(s.sensitivity * (price / anchor - 1.0)))


def expected_rooms_sold(s: Scenario, price: float) -> float:
    """E[min(D, rooms_left)] for D ~ Poisson(expected_demand)."""
    lam, k = expected_demand(s, price), s.rooms_left
    if k <= 0:
        return 0.0
    # E[min(D,k)] = sum_{j=0}^{k-1} P(D > j)
    pmf = math.exp(-lam)
    cdf = pmf
    total = 0.0
    for j in range(k):
        total += 1.0 - cdf
        pmf *= lam / (j + 1)
        cdf += pmf
    return total


def expected_revenue(s: Scenario, price: float) -> float:
    return price * expected_rooms_sold(s, price)


def oracle_price(s: Scenario, step: float = 1.0) -> float:
    """Revenue-maximising price on a 1-unit grid between floor and ceiling."""
    grid = np.arange(s.floor, s.ceiling + step / 2, step)
    revenues = [expected_revenue(s, float(p)) for p in grid]
    return float(grid[int(np.argmax(revenues))])


def generate_scenarios(n: int, seed: int = 7) -> list[Scenario]:
    """A seeded mix of situations a revenue manager sees: quiet nights, compression from events,
    nearly sold-out dates, far-out dates and competitors undercutting."""
    rng = np.random.default_rng(seed)
    kinds = ["quiet", "event", "nearly_full", "far_out", "undercut"]
    out = []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        capacity = int(rng.choice([60, 90, 120, 180]))
        ref = float(rng.choice([90, 110, 130, 160]))
        days = int(rng.integers(1, 15))
        sold_share = float(rng.uniform(0.3, 0.75))
        demand_share = float(rng.uniform(0.2, 0.6))
        sensitivity = float(rng.uniform(3.0, 7.0))
        comp_shift = float(rng.uniform(-0.05, 0.08))
        event = None
        if kind == "event":
            demand_share *= 2.5
            sensitivity *= 0.6
            comp_shift += 0.25
            event = str(rng.choice(["city marathon", "trade fair", "concert", "football final"]))
        elif kind == "nearly_full":
            sold_share = float(rng.uniform(0.88, 0.95))
        elif kind == "far_out":
            days = int(rng.integers(45, 120))
            demand_share *= 1.6
        elif kind == "undercut":
            comp_shift -= 0.2
            sensitivity *= 1.4
        elif kind == "quiet":
            demand_share *= 0.5
        rooms_left = max(1, capacity - int(round(capacity * sold_share)))
        comps = tuple(round(ref * (1 + comp_shift + float(rng.normal(0, 0.04)))) for _ in range(4))
        out.append(
            Scenario(
                id=f"S{i + 1:02d}-{kind}",
                days_to_arrival=days,
                capacity=capacity,
                rooms_left=rooms_left,
                current_price=ref,
                reference_price=ref,
                base_demand=round(capacity * demand_share, 1),
                sensitivity=round(sensitivity, 2),
                competitor_rates=comps,
                floor=round(ref * 0.6),
                ceiling=round(ref * 2.2),
                event=event,
            )
        )
    return out
