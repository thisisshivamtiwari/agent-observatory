"""Command line: python -m agent_observatory <command>.

scenarios                        regenerate evals/scenarios.json (seeded)
scenarios -n 150 --set n150      write the larger evals/scenarios-n150.json (its first 30 are the golden set)
eval --prompt v2 --backend mlx   run the local model, record the cassette, write results/eval-v2.json
eval --prompt v2 --backend replay
eval --prompt v2 --set n150      the same on a named scenario set (results/eval-v2-n150.json)
baselines                        score the non-LLM policies, write results/baselines.json
gate --candidate results/eval-v3-regressed.json
ci                               what GitHub Actions runs: replay the active prompt, check it reproduces the
                                 committed result, then gate it against the accepted result
report                           regenerate the README results tables from results/*.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import yaml

from .agent import ROOT
from .evals import baseline_policies, load_scenarios, run_eval, scenario_file
from .gate import GATE_FILE, gate
from .llm import MlxClient, RecordingClient, ReplayClient
from .simulator import generate_scenarios

CASSETTES = ROOT / "cassettes"


def client_for(backend: str, version: str, scenario_set: str = ""):
    cassette = CASSETTES / (f"{version}-{scenario_set}.jsonl" if scenario_set else f"{version}.jsonl")
    if backend == "mlx":
        return RecordingClient(MlxClient(), cassette)
    if backend == "replay":
        return ReplayClient(cassette)
    raise SystemExit(f"unknown backend {backend}")


def summary_line(r: dict) -> str:
    o = r["ops"]
    return (
        f"{r['prompt_version']}: pass {r['pass_rate']:.0%}, revenue capture {r['revenue_capture_mean']:.1%}, "
        f"p95 {o['latency_ms_p95'] / 1000:.1f}s, {o['tokens_per_decision']:.0f} tokens, "
        f"${o['cost_usd_per_1k_decisions_est']:.3f}/1k decisions (est.)"
    )


def cmd_ci() -> int:
    rules = yaml.safe_load(GATE_FILE.read_text())
    version = rules["active_prompt"]
    committed = json.loads((ROOT / "results" / f"eval-{version}.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        fresh = run_eval(version, client_for("replay", version), "replay", Path(tmp), Path(tmp))
        fails, warns = gate(Path(tmp) / f"eval-{version}.json")
    print(summary_line(fresh))
    for key in ("pass_rate", "revenue_capture_mean", "check_pass_rates"):
        if fresh[key] != committed[key]:
            fails.append(f"replayed {key} {fresh[key]} differs from committed {committed[key]}")
    report_gate(fails, warns)
    return 1 if fails else 0


def report_gate(fails: list[str], warns: list[str]) -> None:
    for w in warns:
        print("GATE WARN:", w)
    for f in fails:
        print("GATE FAIL:", f)
    print("gate:", "FAIL" if fails else "PASS")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agent_observatory")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("scenarios")
    sc.add_argument("-n", type=int, default=30)
    sc.add_argument("--set", default="", dest="scenario_set")
    ev = sub.add_parser("eval")
    ev.add_argument("--prompt", required=True)
    ev.add_argument("--backend", choices=["mlx", "replay"], default="replay")
    ev.add_argument("--set", default="", dest="scenario_set")
    sub.add_parser("baselines")
    gt = sub.add_parser("gate")
    gt.add_argument("--candidate", required=True)
    gt.add_argument("--baseline")
    sub.add_parser("ci")
    sub.add_parser("report")
    sub.add_parser("demo")
    a = ap.parse_args(argv)

    if a.cmd == "scenarios":
        path = scenario_file(a.scenario_set)
        path.write_text(json.dumps([s.to_dict() for s in generate_scenarios(a.n)], indent=1) + "\n")
        print(f"wrote {a.n} scenarios to {path.relative_to(ROOT)}")
    elif a.cmd == "eval":
        client = client_for(a.backend, a.prompt, a.scenario_set)
        r = run_eval(a.prompt, client, a.backend, scenario_set=a.scenario_set)
        print(summary_line(r))
    elif a.cmd == "baselines":
        out = ROOT / "results" / "baselines.json"
        out.write_text(json.dumps(baseline_policies(load_scenarios()), indent=1) + "\n")
        print(out.read_text())
    elif a.cmd == "gate":
        fails, warns = gate(Path(a.candidate), Path(a.baseline) if a.baseline else None)
        report_gate(fails, warns)
        return 1 if fails else 0
    elif a.cmd == "ci":
        return cmd_ci()
    elif a.cmd == "report":
        from .report import write_readme_tables

        write_readme_tables()
    elif a.cmd == "demo":
        from .demo import main as demo

        return demo()
    return 0


if __name__ == "__main__":
    sys.exit(main())
