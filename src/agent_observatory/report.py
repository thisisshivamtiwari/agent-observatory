"""Regenerate the README results tables from results/*.json (so every number in the README comes from a run)."""

from __future__ import annotations

import json
import re

import yaml

from .agent import ROOT
from .gate import GATE_FILE, check

ORDER = ["v1", "v2", "v3-regressed", "v4", "v5"]
START, END = "<!-- results:start -->", "<!-- results:end -->"


def describe(version: str) -> str:
    return yaml.safe_load((ROOT / "prompts" / f"{version}.yaml").read_text())["change"]


def tables() -> str:
    rules = yaml.safe_load(GATE_FILE.read_text())
    accepted = json.loads((ROOT / rules["accepted"]).read_text())
    results = {v: json.loads((ROOT / "results" / f"eval-{v}.json").read_text()) for v in ORDER}
    base = json.loads((ROOT / "results" / "baselines.json").read_text())
    any_r = results[ORDER[0]]
    out = [
        f"Model: `{any_r['model']}` (local, greedy decoding), {any_r['scenarios']} scenarios, "
        f"accepted version: **{rules['active_prompt']}**.",
        "",
        "| Version | Change | Pass rate | Revenue capture | p95 latency / decision | Tokens / decision "
        "| Est. cost / 1k decisions | Tool calls | Gate vs accepted |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for v, r in results.items():
        o = r["ops"]
        fails = check(r, accepted, rules)
        verdict = "PASS" if not fails else f"FAIL ({len(fails)} check{'s' if len(fails) > 1 else ''})"
        if v == rules["active_prompt"]:
            verdict = "accepted"
        out.append(
            f"| {v} | {describe(v)} | {r['pass_rate']:.0%} | {r['revenue_capture_mean']:.1%} | "
            f"{o['latency_ms_p95'] / 1000:.1f} s | {o['tokens_per_decision']:.0f} | ${o['cost_usd_per_1k_decisions_est']:.3f} | "
            f"{o['tool_calls_per_decision']:.1f} | {verdict} |"
        )
    checks = list(any_r["check_pass_rates"])
    out += ["", "Share of decisions passing each check:", "", "| Version | " + " | ".join(checks) + " |"]
    out.append("|---" * (len(checks) + 1) + "|")
    for v, r in results.items():
        out.append(f"| {v} | " + " | ".join(f"{r['check_pass_rates'][c]:.0%}" for c in checks) + " |")
    out += ["", "Non-LLM reference policies on the same scenarios:", ""]
    out.append("| Policy | Revenue capture | Decisions at >= 95% of oracle |")
    out.append("|---|---|---|")
    for name, b in base.items():
        out.append(f"| {name} | {b['revenue_capture_mean']:.1%} | {b['share_at_95pct_capture']:.0%} |")
    caught = sum(1 for v, r in results.items() if v != rules["active_prompt"] and check(r, accepted, rules))
    summary = {
        "model": any_r["model"],
        "scenarios": any_r["scenarios"],
        "accepted": rules["active_prompt"],
        "accepted_pass_rate": accepted["pass_rate"],
        "accepted_revenue_capture": accepted["revenue_capture_mean"],
        "versions_rejected_by_gate": caught,
        "versions_compared": len(results) - 1,
        "best_heuristic_capture": max(b["revenue_capture_mean"] for k, b in base.items() if not k.startswith("oracle")),
    }
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return "\n".join(out)


def write_readme_tables() -> None:
    path = ROOT / "README.md"
    text = path.read_text()
    new = re.sub(re.escape(START) + ".*?" + re.escape(END), START + "\n" + tables() + "\n" + END, text, flags=re.S)
    path.write_text(new)
    print("README tables updated")
