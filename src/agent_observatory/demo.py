"""A short terminal walk-through used for the demo video: replay the accepted version, open one decision's trace,
then show the gate rejecting a broken prompt edit. Reads only committed files."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from .agent import ROOT
from .evals import run_eval
from .gate import gate
from .llm import ReplayClient
from .tracing import read_spans

B, DIM, GRN, RED, CYN, END = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[36m", "\033[0m"


def pause(s: float = 0.6) -> None:
    time.sleep(s)


def span_tree(version: str, scenario: str) -> list[str]:
    spans = read_spans(ROOT / "traces" / f"{version}-mlx.jsonl")
    root = next(s for s in spans if s["name"] == "decision" and s["attributes"]["scenario.id"] == scenario)
    kids = sorted((s for s in spans if s["trace_id"] == root["trace_id"] and s is not root), key=lambda s: s["start_ns"])
    llm_ms = sum(s["attributes"].get("llm.latency_ms", 0) for s in kids)
    lines = [f"{B}decision{END} {scenario}  prompt {version}  {llm_ms:,.0f} ms model time"]
    for i, s in enumerate(kids):
        a, last = s["attributes"], i == len(kids) - 1
        branch = "└─" if last else "├─"
        if s["name"] == "llm.call":
            tokens = f"{a['gen_ai.usage.input_tokens']} in / {a['gen_ai.usage.output_tokens']} out tokens"
            detail = f"{tokens}  {a['llm.latency_ms']:,.0f} ms"
            lines.append(f"  {branch} {CYN}llm.call{END}   {DIM}{detail}{END}")
        else:
            args = a.get("tool.args", "")
            lines.append(f"  {branch} {GRN}{s['name']}{END} {DIM}{args[:52]}{END}")
    return lines


def main() -> int:
    print(f"{B}1. Replay the accepted prompt (v2) on 30 simulated stay dates{END}")
    print(f"{DIM}   recorded model responses, so CI needs no GPU and no API key{END}")
    with tempfile.TemporaryDirectory() as tmp:
        accepted = run_eval("v2", ReplayClient(ROOT / "cassettes" / "v2.jsonl"), "replay", Path(tmp), Path(tmp))
    pause()
    o = accepted["ops"]
    print(f"   pass rate {accepted['pass_rate']:.0%} · revenue capture {accepted['revenue_capture_mean']:.1%} of the optimum")
    print(f"   p95 {o['latency_ms_p95'] / 1000:.1f} s per decision · {o['tokens_per_decision']:.0f} tokens per decision")
    pause(1.2)
    row = next(r for r in accepted["rows"] if r["passed"])
    print(f"\n{B}2. Open one decision's trace{END}")
    pause()
    for line in span_tree("v2", row["scenario"]):
        print(line)
        pause(0.25)
    print(f"   published {row['agent_price']:g} · best possible {row['oracle_price']:g} · {row['capture']:.0%} of the optimum")
    pause(1.2)
    print(f'\n{B}3. Someone edits the prompt: "undercut the lowest competitor by 5%"{END}')
    pause()
    bad = json.loads((ROOT / "results" / "eval-v3-regressed.json").read_text())
    for f in gate(ROOT / "results" / "eval-v3-regressed.json"):
        print(f"   {RED}GATE FAIL{END} {f}")
    print(f"   {RED}{B}gate: FAIL{END}, so CI goes red on the pull request")
    pause(0.8)
    bad_c, ok_c = bad["revenue_capture_mean"], accepted["revenue_capture_mean"]
    print(f"   {DIM}mean revenue capture alone would have passed it: {bad_c:.1%} vs {ok_c:.1%}{END}")
    return 0
