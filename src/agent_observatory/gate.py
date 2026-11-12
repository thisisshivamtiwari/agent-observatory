"""The CI eval gate: a candidate eval result fails if quality drops, or latency or cost grow, beyond the limits in
evals/gate.yaml compared with the accepted baseline result."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .agent import ROOT

GATE_FILE = ROOT / "evals" / "gate.yaml"


def check(candidate: dict, baseline: dict, rules: dict) -> list[str]:
    fails = []
    drop = baseline["pass_rate"] - candidate["pass_rate"]
    if drop > rules["max_pass_rate_drop"]:
        fails.append(
            f"pass rate {candidate['pass_rate']:.1%} vs accepted {baseline['pass_rate']:.1%} "
            f"(drop {drop:.1%} > {rules['max_pass_rate_drop']:.0%})"
        )
    cdrop = baseline["revenue_capture_mean"] - candidate["revenue_capture_mean"]
    if cdrop > rules["max_capture_drop"]:
        fails.append(
            f"revenue capture {candidate['revenue_capture_mean']:.1%} vs accepted "
            f"{baseline['revenue_capture_mean']:.1%} (drop {cdrop:.1%} > {rules['max_capture_drop']:.0%})"
        )
    for key, limit in (("latency_ms_p95", "max_latency_growth"), ("cost_usd_per_1k_decisions_est", "max_cost_growth")):
        b, c = baseline["ops"][key], candidate["ops"][key]
        if b and c > b * (1 + rules[limit]):
            fails.append(f"{key} {c} vs accepted {b} (+{c / b - 1:.0%} > +{rules[limit]:.0%})")
    return fails


def gate(candidate_path: Path, baseline_path: Path | None = None) -> list[str]:
    rules = yaml.safe_load(GATE_FILE.read_text())
    baseline_path = baseline_path or ROOT / rules["accepted"]
    return check(json.loads(Path(candidate_path).read_text()), json.loads(Path(baseline_path).read_text()), rules)
