"""Model backends.

- MlxClient: a local open model on Apple silicon (default Qwen2.5-1.5B-Instruct, 4-bit), greedy decoding.
- RecordingClient / ReplayClient: a cassette of every request and response, keyed by a hash of the request.
  Live runs record; CI replays, so the eval gate is deterministic, free and needs no GPU or API key.
  A changed prompt produces new request hashes, so CI fails with a cassette miss until someone re-records
  the evals locally and commits the new results, which is the point: no prompt change merges unevaluated.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str


def request_key(model: str, messages: list[dict], max_tokens: int) -> str:
    blob = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


class MlxClient:
    def __init__(self, model: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"):
        from mlx_lm import load  # imported lazily: only available on macOS

        self.model_name = model
        # OBSERVATORY_MODEL_PATH: a local copy of the same weights (skips the Hugging Face download)
        self._model, self._tok = load(os.environ.get("OBSERVATORY_MODEL_PATH", model))

    def complete(self, messages: list[dict], max_tokens: int = 200) -> Completion:
        from mlx_lm import generate

        prompt = self._tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        t0 = time.perf_counter()
        text = generate(self._model, self._tok, prompt=prompt, max_tokens=max_tokens, verbose=False)
        latency = (time.perf_counter() - t0) * 1000
        return Completion(
            text=text,
            input_tokens=len(self._tok.encode(prompt)),
            output_tokens=len(self._tok.encode(text)),
            latency_ms=round(latency, 1),
            model=self.model_name,
        )


class CassetteMiss(Exception):
    pass


def _load_cassette(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["key"]: r for r in rows}


class RecordingClient:
    def __init__(self, inner, path: Path):
        self.inner, self.path, self.model_name = inner, Path(path), inner.model_name
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen = _load_cassette(self.path)

    def complete(self, messages: list[dict], max_tokens: int = 200) -> Completion:
        key = request_key(self.model_name, messages, max_tokens)
        if key in self._seen:
            return Completion(**self._seen[key]["completion"])
        c = self.inner.complete(messages, max_tokens)
        self._seen[key] = {"key": key, "completion": asdict(c)}
        with self.path.open("a") as fh:
            fh.write(json.dumps(self._seen[key]) + "\n")
        return c


class ReplayClient:
    def __init__(self, path: Path, model: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"):
        self.path, self.model_name = Path(path), model
        self._seen = _load_cassette(self.path)

    def complete(self, messages: list[dict], max_tokens: int = 200) -> Completion:
        key = request_key(self.model_name, messages, max_tokens)
        if key not in self._seen:
            raise CassetteMiss(
                f"no recorded response for request {key} in {self.path}. The prompt or scenarios changed: "
                "re-run the evals locally with --backend mlx and commit the cassette and results."
            )
        return Completion(**self._seen[key]["completion"])
