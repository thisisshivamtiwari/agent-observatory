"""Scenario evals: run the agent over the golden scenario set, score every decision against the simulator's oracle,
and compute the operational metrics (latency, tokens, cost) from the trace file."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median

import numpy as np

from .agent import ROOT, Config, Decision, load_prompt, run_decision
from .simulator import Scenario, expected_revenue, oracle_price
from .tracing import make_tracer, read_spans

SCENARIOS = ROOT / "evals" / "scenarios.json"
CAPTURE_MIN = 0.95  # a decision passes only if it earns at least 95% of the oracle's expected revenue


def load_scenarios(path: Path = SCENARIOS) -> list[Scenario]:
    return [Scenario.from_dict(d) for d in json.loads(Path(path).read_text())]


def score(s: Scenario, dec: Decision) -> dict:
    """Checks for one decision. If the agent fails to publish a valid price, the current price stays live, so
    revenue capture is computed for the price that would actually be on sale."""
    opt = oracle_price(s)
    valid = dec.final_tool == "set_price" and dec.price is not None
    within = valid and s.floor <= dec.price <= s.ceiling
    live = dec.price if within else s.current_price
    capture = expected_revenue(s, live) / expected_revenue(s, opt)
    cur = s.current_price
    if opt > cur * 1.05:
        direction = valid and dec.price > cur
    elif opt < cur * 0.95:
        direction = valid and dec.price < cur
    else:
        direction = valid and abs(dec.price - cur) <= 0.1 * cur
    checks = {
        "final_action": valid,
        "within_limits": bool(within),
        "used_forecast": "forecast_demand" in dec.tool_calls,
        "right_direction": bool(direction),
        "revenue_capture": capture >= CAPTURE_MIN,
    }
    return {
        "scenario": s.id,
        "oracle_price": opt,
        "agent_price": dec.price,
        "capture": round(capture, 4),
        "checks": checks,
        "passed": all(checks.values()),
    }


def ops_from_traces(path: Path) -> dict:
    """Per-decision latency, tokens, cost and tool calls, aggregated from the spans alone."""
    spans = read_spans(path)
    by_trace: dict[str, list[dict]] = {}
    for sp in spans:
        by_trace.setdefault(sp["trace_id"], []).append(sp)
    lat, tok, cost, tools, errs = [], [], [], [], 0
    for rows in by_trace.values():
        llm = [r for r in rows if r["name"] == "llm.call"]
        lat.append(sum(r["attributes"]["llm.latency_ms"] for r in llm))
        tok.append(sum(r["attributes"]["gen_ai.usage.input_tokens"] + r["attributes"]["gen_ai.usage.output_tokens"] for r in llm))
        cost.append(sum(r["attributes"]["llm.cost_usd_est"] for r in llm))
        tools.append(sum(1 for r in rows if r["name"].startswith("tool.")))
        errs += sum(1 for r in rows if r["status"] == "ERROR")
    return {
        "decisions": len(by_trace),
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 1),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
        "tokens_per_decision": round(mean(tok), 1),
        "cost_usd_per_1k_decisions_est": round(1000 * mean(cost), 4),
        "tool_calls_per_decision": round(mean(tools), 2),
        "error_spans": errs,
        "spans": len(spans),
    }


def run_eval(version: str, client, backend: str, out_dir: Path = ROOT / "results", trace_dir: Path = ROOT / "traces") -> dict:
    cfg, prompt, scenarios = Config.load(), load_prompt(version), load_scenarios()
    trace_path = Path(trace_dir) / f"{version}-{backend}.jsonl"
    provider, tracer = make_tracer(trace_path)
    rows = []
    for s in scenarios:
        dec = run_decision(client, s, prompt, tracer, cfg, version)
        rows.append({**score(s, dec), "decision": asdict(dec)})
    provider.shutdown()
    check_names = rows[0]["checks"].keys()
    result = {
        "prompt_version": version,
        "model": client.model_name,
        "backend": backend,
        "scenarios": len(rows),
        "pass_rate": round(mean(r["passed"] for r in rows), 4),
        "check_pass_rates": {c: round(mean(r["checks"][c] for r in rows), 4) for c in check_names},
        "revenue_capture_mean": round(mean(r["capture"] for r in rows), 4),
        "revenue_capture_median": round(median(r["capture"] for r in rows), 4),
        "ops": ops_from_traces(trace_path),
        "rows": rows,
    }
    out = Path(out_dir) / f"eval-{version}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    return result


def baseline_policies(scenarios: list[Scenario]) -> dict:
    """Non-LLM reference points scored the same way."""
    med = lambda s: float(np.median(s.competitor_rates))  # noqa: E731
    policies = {
        "keep current price": lambda s: s.current_price,
        "match competitor median": med,
        "5% under lowest competitor": lambda s: round(min(s.competitor_rates) * 0.95),
        "oracle (knows the demand curve)": oracle_price,
    }
    out = {}
    for name, f in policies.items():
        caps = []
        for s in scenarios:
            p = min(max(f(s), s.floor), s.ceiling)
            caps.append(expected_revenue(s, p) / expected_revenue(s, oracle_price(s)))
        out[name] = {
            "revenue_capture_mean": round(mean(caps), 4),
            "share_at_95pct_capture": round(mean(c >= CAPTURE_MIN for c in caps), 4),
        }
    return out
