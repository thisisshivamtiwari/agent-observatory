"""OpenTelemetry tracing to a local JSONL file (one span per line).

Every decision is a root span; each model call and tool call is a child span carrying tokens, latency and
estimated cost (GenAI semantic-convention attribute names where they exist). The same spans could be sent to
any OTLP backend (Jaeger, Grafana Tempo, Langfuse, Honeycomb) by adding an OTLP exporter; the reports in this
repo are computed from the JSONL file so the numbers come from the traces themselves.
"""

from __future__ import annotations

import json
from pathlib import Path

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult


class JsonlSpanExporter(SpanExporter):
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w")

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        for s in spans:
            ctx = s.get_span_context()
            row = {
                "name": s.name,
                "trace_id": f"{ctx.trace_id:032x}",
                "span_id": f"{ctx.span_id:016x}",
                "parent_id": f"{s.parent.span_id:016x}" if s.parent else None,
                "start_ns": s.start_time,
                "end_ns": s.end_time,
                "duration_ms": round((s.end_time - s.start_time) / 1e6, 3),
                "status": s.status.status_code.name,
                "attributes": dict(s.attributes or {}),
            }
            self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self._fh.close()


def make_tracer(path: Path, service: str = "pricing-agent"):
    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(path)))
    return provider, provider.get_tracer("agent_observatory")


def read_spans(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
