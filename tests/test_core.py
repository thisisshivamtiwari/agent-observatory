import json

import pytest

from agent_observatory.agent import Config, load_prompt, parse_action, run_decision
from agent_observatory.evals import load_scenarios, ops_from_traces, score
from agent_observatory.gate import check
from agent_observatory.llm import CassetteMiss, Completion, RecordingClient, ReplayClient
from agent_observatory.simulator import expected_revenue, expected_rooms_sold, generate_scenarios, oracle_price
from agent_observatory.tracing import make_tracer, read_spans


class ScriptedClient:
    """Returns canned replies in order, so the agent loop can be tested without a model."""

    model_name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, messages, max_tokens=200):
        return Completion(self.replies.pop(0), input_tokens=100, output_tokens=20, latency_ms=50.0, model="scripted")


def test_scenarios_are_seeded_and_committed_set_matches():
    assert generate_scenarios(30) == generate_scenarios(30)
    assert [s.to_dict() for s in generate_scenarios(30)] == [s.to_dict() for s in load_scenarios()]


def test_rooms_sold_never_exceeds_rooms_left_and_falls_with_price():
    for s in generate_scenarios(10):
        lo, hi = expected_rooms_sold(s, s.floor), expected_rooms_sold(s, s.ceiling)
        assert 0 <= hi <= lo <= s.rooms_left


def test_oracle_is_best_on_grid():
    s = generate_scenarios(3)[2]
    best = expected_revenue(s, oracle_price(s))
    assert all(expected_revenue(s, p) <= best + 1e-9 for p in range(int(s.floor), int(s.ceiling) + 1))


def test_parse_action_takes_first_json_object():
    assert parse_action('ok {"tool": "set_price", "args": {"price": 120}} trailing') == {
        "tool": "set_price",
        "args": {"price": 120},
    }
    with pytest.raises(ValueError):
        parse_action("no json here")


def run(replies, tmp_path, scenario=None):
    s = scenario or load_scenarios()[0]
    provider, tracer = make_tracer(tmp_path / "t.jsonl")
    dec = run_decision(ScriptedClient(replies), s, load_prompt("v2"), tracer, Config(), "v2")
    provider.shutdown()
    return s, dec


def test_agent_loop_traces_every_step(tmp_path):
    s = load_scenarios()[0]
    opt = oracle_price(s)
    s, dec = run(
        [
            '{"tool": "get_hotel_state", "args": {}}',
            json.dumps({"tool": "forecast_demand", "args": {"prices": [opt * 0.8, opt]}}),
            "not json",
            json.dumps({"tool": "set_price", "args": {"price": opt, "reason": "best forecast"}}),
        ],
        tmp_path,
    )
    assert dec.final_tool == "set_price" and dec.price == opt
    assert dec.tool_calls == ["get_hotel_state", "forecast_demand", "set_price"]
    assert dec.errors == ["parse: no JSON object in reply"]
    names = [r["name"] for r in read_spans(tmp_path / "t.jsonl")]
    assert names.count("llm.call") == 4 and names.count("decision") == 1
    ops = ops_from_traces(tmp_path / "t.jsonl")
    assert ops["decisions"] == 1 and ops["latency_ms_p95"] == 200.0 and ops["tokens_per_decision"] == 480
    assert score(s, dec)["passed"]


def test_no_final_action_keeps_current_price(tmp_path):
    s, dec = run(['{"tool": "get_hotel_state", "args": {}}'] * Config().max_steps, tmp_path)
    r = score(s, dec)
    assert dec.final_tool is None and not r["passed"]
    assert r["capture"] == pytest.approx(expected_revenue(s, s.current_price) / expected_revenue(s, oracle_price(s)), abs=1e-4)


def test_cassette_records_then_replays_and_misses_on_change(tmp_path):
    path = tmp_path / "c.jsonl"
    msgs = [{"role": "user", "content": "hi"}]
    rec = RecordingClient(ScriptedClient(['{"tool": "x"}']), path)
    first = rec.complete(msgs)
    replay = ReplayClient(path, model="scripted")
    assert replay.complete(msgs) == first
    with pytest.raises(CassetteMiss):
        replay.complete([{"role": "user", "content": "changed"}])


RULES = {"max_pass_rate_drop": 0.05, "max_capture_drop": 0.02, "max_latency_growth": 0.5, "max_cost_growth": 0.5}


def result(pass_rate, capture, p95=1000.0, cost=0.1):
    return {
        "pass_rate": pass_rate,
        "revenue_capture_mean": capture,
        "ops": {"latency_ms_p95": p95, "cost_usd_per_1k_decisions_est": cost},
    }


def test_gate_passes_equal_and_fails_regressions():
    base = result(0.7, 0.95)
    assert check(base, base, RULES) == []
    assert len(check(result(0.6, 0.95), base, RULES)) == 1
    assert len(check(result(0.7, 0.90), base, RULES)) == 1
    assert len(check(result(0.7, 0.95, p95=2000), base, RULES)) == 1
    assert len(check(result(0.7, 0.95, cost=0.2), base, RULES)) == 1


def test_prompts_render():
    for v in ("v1", "v2", "v3-regressed"):
        p = load_prompt(v)
        assert "{tools}" not in p["system"].format(tools="T") and json.dumps(p)
