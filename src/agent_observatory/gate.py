"""The CI eval gate: a candidate eval result fails if quality drops, or latency or cost grow, beyond the limits in
evals/gate.yaml compared with the accepted baseline result.

Two modes for the quality checks (pass rate and revenue capture):
- "fixed": fail when the drop is larger than a fixed tolerance (the v1 gate).
- "ci": fail only when the paired bootstrap 95% interval of candidate minus accepted lies entirely below zero, i.e.
  the drop is real rather than scenario noise. A drop above the fixed tolerance that the interval can't confirm is
  reported as a warning, with the number of scenarios needed to settle it.
Latency and cost always use fixed relative limits: they are measured on every decision and are not sampling noise
in the same way."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from . import stats
from .agent import ROOT

GATE_FILE = ROOT / "evals" / "gate.yaml"
QUALITY = (("pass_rate", "pass rate", "max_pass_rate_drop"), ("revenue_capture", "revenue capture", "max_capture_drop"))


def _mean(result: dict, metric: str) -> float:
    return result["pass_rate"] if metric == "pass_rate" else result["revenue_capture_mean"]


def _ops(candidate: dict, baseline: dict, rules: dict) -> list[str]:
    fails = []
    for key, limit in (("latency_ms_p95", "max_latency_growth"), ("cost_usd_per_1k_decisions_est", "max_cost_growth")):
        b, c = baseline["ops"][key], candidate["ops"][key]
        if b and c > b * (1 + rules[limit]):
            fails.append(f"{key} {c} vs accepted {b} (+{c / b - 1:.0%} > +{rules[limit]:.0%})")
    return fails


def check_fixed(candidate: dict, baseline: dict, rules: dict) -> list[str]:
    fails = []
    for metric, label, limit in QUALITY:
        c, b = _mean(candidate, metric), _mean(baseline, metric)
        if b - c > rules[limit]:
            fails.append(f"{label} {c:.1%} vs accepted {b:.1%} (drop {b - c:.1%} > {rules[limit]:.0%})")
    return fails + _ops(candidate, baseline, rules)


def check_ci(candidate: dict, baseline: dict, rules: dict) -> tuple[list[str], list[str]]:
    """Returns (fails, warnings)."""
    fails, warns = [], []
    for metric, label, limit in QUALITY:
        p = stats.paired(candidate, baseline, metric)
        lo, hi = p["ci95"]
        span = f"{label} {p['diff']:+.1%} vs accepted, 95% CI [{lo:+.1%}, {hi:+.1%}], n={p['n']}"
        if hi < 0:
            fails.append(f"{span}: confidently worse")
        elif -p["diff"] > rules[limit]:
            need = stats.scenarios_needed(p["sd"], rules[limit]) if p["sd"] else p["n"]
            warns.append(
                f"{span}: drop above the {rules[limit]:.0%} tolerance but not confirmed; n={p['n']} detects drops of "
                f"{stats.detectable_drop(p['sd'], p['n']):.0%} or more, ~{need} scenarios needed for {rules[limit]:.0%}"
            )
    return fails + _ops(candidate, baseline, rules), warns


def check(candidate: dict, baseline: dict, rules: dict) -> list[str]:
    if rules.get("mode", "fixed") == "ci":
        return check_ci(candidate, baseline, rules)[0]
    return check_fixed(candidate, baseline, rules)


def warnings(candidate: dict, baseline: dict, rules: dict) -> list[str]:
    return check_ci(candidate, baseline, rules)[1] if rules.get("mode", "fixed") == "ci" else []


def gate(candidate_path: Path, baseline_path: Path | None = None) -> tuple[list[str], list[str]]:
    rules = yaml.safe_load(GATE_FILE.read_text())
    baseline_path = baseline_path or ROOT / rules["accepted"]
    cand, base = json.loads(Path(candidate_path).read_text()), json.loads(Path(baseline_path).read_text())
    return check(cand, base, rules), warnings(cand, base, rules)
