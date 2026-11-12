"""A tool-using pricing agent: the model answers with one JSON tool call per turn until it calls a final tool."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from opentelemetry.trace import Status, StatusCode

from .simulator import Scenario
from .tools import FINAL_TOOLS, TOOL_SPECS, ToolError, call_tool

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    max_steps: int = 8
    max_tokens: int = 160
    usd_per_mtok_in: float = 0.10
    usd_per_mtok_out: float = 0.40

    @classmethod
    def load(cls, path: Path = ROOT / "config.yaml") -> Config:
        return cls(**yaml.safe_load(Path(path).read_text()))

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.usd_per_mtok_in + tokens_out * self.usd_per_mtok_out) / 1e6


@dataclass
class Decision:
    scenario_id: str
    final_tool: str | None = None
    price: float | None = None
    reason: str = ""
    tool_calls: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    guardrail_blocks: int = 0
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0


def load_prompt(version: str) -> dict:
    return yaml.safe_load((ROOT / "prompts" / f"{version}.yaml").read_text())


def parse_action(text: str) -> dict:
    """First JSON object in the model's reply, which must look like {"tool": ..., "args": {...}}."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict) or "tool" not in obj:
        raise ValueError('reply must be {"tool": name, "args": {...}}')
    args = obj.get("args") or {}
    if not isinstance(args, dict):
        raise ValueError("args must be an object")
    return {"tool": str(obj["tool"]), "args": args}


def guardrail(scenario: Scenario, name: str, args: dict, forecasted: set[float]) -> str | None:
    """Runtime policy checked in code before a final action is accepted (enabled per prompt version).
    Returns the reason to block, or None."""
    if name == "close_out" and scenario.rooms_left > 0:
        return "close_out is not allowed while rooms are left; call set_price"
    if name == "set_price":
        try:
            price = float(args["price"])
        except (KeyError, TypeError, ValueError):
            return 'set_price needs {"price": number, "reason": string}'
        if not scenario.floor <= price <= scenario.ceiling:
            return f"price {price:g} is outside the allowed range {scenario.floor:g}-{scenario.ceiling:g}"
        if price not in forecasted:
            return f"forecast_demand must include price {price:g} before you publish it"
    return None


def run_decision(client, scenario: Scenario, prompt: dict, tracer, cfg: Config, version: str) -> Decision:
    tools = "\n".join(f"- {n}: {d}" for n, d in TOOL_SPECS.items())
    messages = [
        {"role": "system", "content": prompt["system"].format(tools=tools)},
        {"role": "user", "content": prompt["task"].format(stay_date=scenario.id)},
    ]
    dec = Decision(scenario_id=scenario.id)
    guarded, forecasted = bool(prompt.get("guardrails")), set()
    with tracer.start_as_current_span("decision") as root:
        root.set_attribute("scenario.id", scenario.id)
        root.set_attribute("prompt.version", version)
        root.set_attribute("gen_ai.request.model", client.model_name)
        for step in range(cfg.max_steps):
            with tracer.start_as_current_span("llm.call") as sp:
                c = client.complete(messages, cfg.max_tokens)
                cost = cfg.cost(c.input_tokens, c.output_tokens)
                sp.set_attribute("gen_ai.request.model", c.model)
                sp.set_attribute("gen_ai.usage.input_tokens", c.input_tokens)
                sp.set_attribute("gen_ai.usage.output_tokens", c.output_tokens)
                sp.set_attribute("llm.latency_ms", c.latency_ms)
                sp.set_attribute("llm.cost_usd_est", cost)
                sp.set_attribute("step", step)
            dec.llm_calls += 1
            dec.tokens_in += c.input_tokens
            dec.tokens_out += c.output_tokens
            dec.latency_ms += c.latency_ms
            dec.cost_usd += cost
            messages.append({"role": "assistant", "content": c.text})
            try:
                action = parse_action(c.text)
            except ValueError as e:
                dec.errors.append(f"parse: {e}")
                messages.append({"role": "user", "content": f"ERROR: {e}. Reply with one JSON object only."})
                continue
            name, args = action["tool"], action["args"]
            dec.tool_calls.append(name)
            with tracer.start_as_current_span(f"tool.{name}") as sp:
                sp.set_attribute("tool.name", name)
                sp.set_attribute("tool.args", json.dumps(args)[:500])
                block = guardrail(scenario, name, args, forecasted) if guarded and name in FINAL_TOOLS else None
                if block:
                    dec.guardrail_blocks += 1
                    sp.set_attribute("guardrail.blocked", block)
                    sp.set_status(Status(StatusCode.ERROR, "guardrail"))
                    messages.append({"role": "user", "content": f"BLOCKED: {block}"})
                    continue
                if name in FINAL_TOOLS:
                    dec.final_tool, dec.reason = name, str(args.get("reason", ""))[:300]
                    if name == "set_price":
                        try:
                            dec.price = float(args["price"])
                        except (KeyError, TypeError, ValueError):
                            dec.errors.append("set_price without a numeric price")
                            sp.set_status(Status(StatusCode.ERROR, "bad price"))
                    break
                try:
                    result = call_tool(scenario, name, args)
                except ToolError as e:
                    dec.errors.append(f"tool: {e}")
                    sp.set_status(Status(StatusCode.ERROR, str(e)))
                    messages.append({"role": "user", "content": f"ERROR: {e}"})
                    continue
                sp.set_attribute("tool.result", json.dumps(result)[:500])
                if name == "forecast_demand":
                    forecasted.update(f["price"] for f in result["forecasts"])
            messages.append({"role": "user", "content": f"RESULT {name}: {json.dumps(result)}"})
        root.set_attribute("decision.final_tool", dec.final_tool or "none")
        if dec.price is not None:
            root.set_attribute("decision.price", dec.price)
        root.set_attribute("decision.errors", len(dec.errors))
        root.set_attribute("decision.guardrail_blocks", dec.guardrail_blocks)
        if dec.final_tool is None:
            root.set_status(Status(StatusCode.ERROR, "no final action within step budget"))
    return dec
