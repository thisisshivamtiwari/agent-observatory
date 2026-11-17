"""Bootstrap intervals over scenarios. With 30 scenarios a pass rate moves 3.3 points per scenario, so every
comparison between versions carries a 95% interval, and the gate can ask whether a drop is real or noise."""

from __future__ import annotations

import numpy as np

N_BOOT = 10_000
SEED = 0
Z_95, Z_80 = 1.959964, 0.841621  # two-sided 95% confidence, 80% power


def _per_scenario(result: dict, metric: str) -> dict[str, float]:
    if metric == "pass_rate":
        return {r["scenario"]: float(r["passed"]) for r in result["rows"]}
    if metric == "revenue_capture":
        return {r["scenario"]: float(r["capture"]) for r in result["rows"]}
    raise ValueError(metric)


def _resample(n: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def interval(result: dict, metric: str) -> dict:
    """Mean of a per-scenario metric with a percentile bootstrap 95% interval."""
    x = np.array(list(_per_scenario(result, metric).values()))
    boot = x[_resample(len(x))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": round(float(x.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def paired(candidate: dict, baseline: dict, metric: str) -> dict:
    """Candidate minus baseline on the same scenarios. The same resampled scenarios are used for both versions,
    so scenario difficulty cancels out and the interval is much tighter than comparing two separate intervals."""
    c, b = _per_scenario(candidate, metric), _per_scenario(baseline, metric)
    if c.keys() != b.keys():
        raise ValueError("candidate and baseline were scored on different scenarios")
    d = np.array([c[k] - b[k] for k in sorted(b)])
    boot = d[_resample(len(d))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    return {
        "diff": round(float(d.mean()), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "p_worse": round(float((boot < 0).mean()), 4),
        "sd": round(sd, 4),
        "n": len(d),
    }


def detectable_drop(sd: float, n: int) -> float:
    """Smallest true drop a paired comparison of n scenarios finds 80% of the time at the 5% level (normal approx.)."""
    return (Z_95 + Z_80) * sd / np.sqrt(n)


def scenarios_needed(sd: float, drop: float) -> int:
    """Scenarios needed to detect a drop of this size with 80% power, given the spread of per-scenario differences."""
    return int(np.ceil(((Z_95 + Z_80) * sd / drop) ** 2))


def cost_per_pass(result: dict) -> dict:
    """Estimated cost per 1k passed decisions: total cost over passed decisions, with a bootstrap interval.
    Undefined (None) when no decision passed, because no amount of spend buys a good decision."""
    rows = result["rows"]
    cost = np.array([r["decision"]["cost_usd"] for r in rows])
    ok = np.array([float(r["passed"]) for r in rows])
    if ok.sum() == 0:
        return {"usd_per_1k_passed": None, "ci95": None}
    idx = _resample(len(rows))
    passed = ok[idx].sum(axis=1)
    boot = cost[idx].sum(axis=1)[passed > 0] / passed[passed > 0] * 1000
    lo, hi = np.percentile(boot, [2.5, 97.5])
    point = float(cost.sum() / ok.sum() * 1000)
    return {"usd_per_1k_passed": round(point, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}
