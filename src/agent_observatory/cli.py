"""Command line: python -m agent_observatory <command>.

scenarios                        regenerate evals/scenarios.json (seeded)
eval --prompt v2 --backend mlx   run the local model, record the cassette, write results/eval-v2.json
eval --prompt v2 --backend replay
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
from .evals import SCENARIOS, baseline_policies, load_scenarios, run_eval
from .gate import GATE_FILE, gate
from .llm import MlxClient, RecordingClient, ReplayClient
from .simulator import generate_scenarios

CASSETTES = ROOT / "cassettes"


def client_for(backend: str, version: str):
    cassette = CASSETTES / f"{version}.jsonl"
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
        fails = gate(Path(tmp) / f"eval-{version}.json")
    print(summary_line(fresh))
    for key in ("pass_rate", "revenue_capture_mean", "check_pass_rates"):
        if fresh[key] != committed[key]:
            fails.append(f"replayed {key} {fresh[key]} differs from committed {committed[key]}")
    for f in fails:
        print("GATE FAIL:", f)
    print("gate:", "FAIL" if fails else "PASS")
    return 1 if fails else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agent_observatory")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("scenarios")
    sc.add_argument("-n", type=int, default=30)
    ev = sub.add_parser("eval")
    ev.add_argument("--prompt", required=True)
    ev.add_argument("--backend", choices=["mlx", "replay"], default="replay")
    sub.add_parser("baselines")
    gt = sub.add_parser("gate")
    gt.add_argument("--candidate", required=True)
    gt.add_argument("--baseline")
    sub.add_parser("ci")
    sub.add_parser("report")
    sub.add_parser("demo")
    a = ap.parse_args(argv)

    if a.cmd == "scenarios":
        SCENARIOS.write_text(json.dumps([s.to_dict() for s in generate_scenarios(a.n)], indent=1) + "\n")
        print(f"wrote {a.n} scenarios to {SCENARIOS.relative_to(ROOT)}")
    elif a.cmd == "eval":
        r = run_eval(a.prompt, client_for(a.backend, a.prompt), a.backend)
        print(summary_line(r))
    elif a.cmd == "baselines":
        out = ROOT / "results" / "baselines.json"
        out.write_text(json.dumps(baseline_policies(load_scenarios()), indent=1) + "\n")
        print(out.read_text())
    elif a.cmd == "gate":
        fails = gate(Path(a.candidate), Path(a.baseline) if a.baseline else None)
        for f in fails:
            print("GATE FAIL:", f)
        print("gate:", "FAIL" if fails else "PASS")
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
