"""Regenerate the README results tables from results/*.json (so every number in the README comes from a run)."""

from __future__ import annotations

import json
import re

import yaml

from . import stats
from .agent import ROOT
from .gate import GATE_FILE, check, check_ci, check_fixed

ORDER = ["v1", "v2", "v3-regressed", "v4", "v5"]
LARGE_SET = "n150"
START, END = "<!-- results:start -->", "<!-- results:end -->"


def describe(version: str) -> str:
    return yaml.safe_load((ROOT / "prompts" / f"{version}.yaml").read_text())["change"]


def pct_ci(x: dict) -> str:
    lo, hi = x["ci95"]
    return f"{x['mean']:.0%} [{lo:.0%}, {hi:.0%}]"


def diff_ci(p: dict, digits: int = 0) -> str:
    lo, hi = p["ci95"]
    return f"{p['diff']:+.{digits}%} [{lo:+.{digits}%}, {hi:+.{digits}%}]"


def usd_ci(c: dict) -> str:
    if c["usd_per_1k_passed"] is None:
        return "none passed"
    lo, hi = c["ci95"]
    return f"${c['usd_per_1k_passed']:.2f} [${lo:.2f}, ${hi:.2f}]"


def verdict(fails: list[str]) -> str:
    return "PASS" if not fails else f"FAIL ({len(fails)} check{'s' if len(fails) > 1 else ''})"


def intervals(results: dict, accepted: dict, accepted_name: str, rules: dict) -> tuple[list[str], dict]:
    """Bootstrap intervals per version and paired differences vs the accepted version, with both gate modes."""
    out = [
        "",
        f"Uncertainty: 95% bootstrap intervals over the {accepted['scenarios']} scenarios; differences are paired "
        "(same scenarios, candidate minus accepted). Cost per passed decision is total estimated cost over passed decisions.",
        "",
        "| Version | Pass rate | Revenue capture | Pass rate vs accepted | Capture vs accepted "
        "| Est. cost / 1k passed decisions | Fixed-tolerance gate | Interval gate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    data = {}
    for v, r in results.items():
        row = {
            "pass_rate": stats.interval(r, "pass_rate"),
            "revenue_capture": stats.interval(r, "revenue_capture"),
            "cost_per_1k_passed": stats.cost_per_pass(r),
        }
        if v == accepted_name:
            dp = dc = "-"
            fixed = ci = "accepted"
        else:
            row["paired_pass_rate"] = stats.paired(r, accepted, "pass_rate")
            row["paired_revenue_capture"] = stats.paired(r, accepted, "revenue_capture")
            fails, warns = check_ci(r, accepted, rules)
            row["gate_fixed"], row["gate_ci"], row["gate_ci_warnings"] = check_fixed(r, accepted, rules), fails, warns
            dp, dc = diff_ci(row["paired_pass_rate"]), diff_ci(row["paired_revenue_capture"], 1)
            quality = [f for f in fails if "confidently worse" in f]
            fixed = verdict(row["gate_fixed"])
            ci = "FAIL: quality drop confirmed" if quality else ("FAIL: latency/cost only" if fails else "PASS")
        data[v] = row
        out.append(
            f"| {v} | {pct_ci(row['pass_rate'])} | {pct_ci(row['revenue_capture'])} | {dp} | {dc} | "
            f"{usd_ci(row['cost_per_1k_passed'])} | {fixed} | {ci} |"
        )
    return out, data


def large_set(rules: dict) -> tuple[list[str], dict]:
    """The same comparisons on the larger scenario set, for the versions run on it."""
    paths = {v: ROOT / "results" / f"eval-{v}-{LARGE_SET}.json" for v in ORDER}
    runs = {v: json.loads(p.read_text()) for v, p in paths.items() if p.exists()}
    acc = rules["active_prompt"]
    if acc not in runs or len(runs) < 2:
        return [], {}
    a = runs[acc]
    n = a["scenarios"]
    out = [
        "",
        f"The same versions on {n} scenarios (`evals/scenarios-{LARGE_SET}.json`; its first 30 are the golden set):",
        "",
        "| Version | Pass rate | Revenue capture | Pass rate vs accepted | Capture vs accepted "
        "| Smallest pass-rate drop detectable | Interval gate (quality) |",
        "|---|---|---|---|---|---|---|",
    ]
    data = {}
    for v, r in runs.items():
        row = {"pass_rate": stats.interval(r, "pass_rate"), "revenue_capture": stats.interval(r, "revenue_capture")}
        if v == acc:
            cells = ["-", "-", "-", "accepted"]
        else:
            pp, pc = stats.paired(r, a, "pass_rate"), stats.paired(r, a, "revenue_capture")
            fails, _ = check_ci(r, a, rules)
            quality = [f for f in fails if "confidently worse" in f]
            row.update(paired_pass_rate=pp, paired_revenue_capture=pc, quality_fails=quality)
            row["detectable_pass_rate_drop"] = round(float(stats.detectable_drop(pp["sd"], pp["n"])), 4)
            cells = [
                diff_ci(pp),
                diff_ci(pc, 1),
                f"{row['detectable_pass_rate_drop']:.0%}",
                f"FAIL ({len(quality)} confirmed)" if quality else "no confirmed drop",
            ]
        data[v] = row
        out.append(f"| {v} | {pct_ci(row['pass_rate'])} | {pct_ci(row['revenue_capture'])} | " + " | ".join(cells) + " |")
    return out, data


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
        f"| Est. cost / 1k decisions | Tool calls | Gate vs accepted ({rules.get('mode', 'fixed')} mode) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for v, r in results.items():
        o = r["ops"]
        fails = check(r, accepted, rules)
        gate_cell = "accepted" if v == rules["active_prompt"] else verdict(fails)
        out.append(
            f"| {v} | {describe(v)} | {r['pass_rate']:.0%} | {r['revenue_capture_mean']:.1%} | "
            f"{o['latency_ms_p95'] / 1000:.1f} s | {o['tokens_per_decision']:.0f} | ${o['cost_usd_per_1k_decisions_est']:.3f} | "
            f"{o['tool_calls_per_decision']:.1f} | {gate_cell} |"
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
    iv_lines, iv = intervals(results, accepted, rules["active_prompt"], rules)
    big_lines, big = large_set(rules)
    out += iv_lines + big_lines
    (ROOT / "results" / "intervals.json").write_text(
        json.dumps({"n_boot": stats.N_BOOT, "seed": stats.SEED, "golden": iv, LARGE_SET: big}, indent=1) + "\n"
    )
    caught = sum(1 for v, r in results.items() if v != rules["active_prompt"] and check(r, accepted, rules))
    summary = {
        "model": any_r["model"],
        "scenarios": any_r["scenarios"],
        "accepted": rules["active_prompt"],
        "accepted_pass_rate": accepted["pass_rate"],
        "accepted_revenue_capture": accepted["revenue_capture_mean"],
        "versions_rejected_by_gate": caught,
        "versions_compared": len(results) - 1,
        "gate_mode": rules.get("mode", "fixed"),
        "versions_with_confirmed_quality_drop": sum(
            1 for v, row in iv.items() if any("confidently worse" in f for f in row.get("gate_ci", []))
        ),
        "accepted_cost_per_1k_passed_est": iv[rules["active_prompt"]]["cost_per_1k_passed"]["usd_per_1k_passed"],
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
