"""Claude wrapper: structured output, disk-backed caching, and a retrieval-only mode.

Two things here are benchmark infrastructure rather than convenience.

Caching: every judgement is keyed by (model, prompt, schema, settings) and
persisted. The validation harness re-judges the same claims under shifted dates
and disabled retrieval, so without a cache the controls cost several times the
headline run and nobody runs them.

`grounded`: when True the prompt forbids the model from relying on anything but
the supplied excerpts. This is what separates a measurement from a vibe. Asking
a model trained on the whole internet "was this original in 2019" invites it to
answer from hindsight; the contamination control (`validate/controls.py`) exists
precisely to check whether the score survives when the evidence is taken away.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

import anthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

MODEL = "claude-opus-5"

# The CLI backend shells out to headless `claude -p`, which bills against the
# user's Claude subscription instead of metered API credits. Two things matter:
# ANTHROPIC_API_KEY must be scrubbed from the subprocess env (a set key silently
# takes precedence over the claude.ai login and fails on an empty API account),
# and there is no server-side schema enforcement, so JSON is requested by
# instruction and extracted defensively, with one retry.
CLI_MODEL = "opus"

GROUNDING_RULE = """\
You are adjudicating evidence for a benchmark that measures which source said \
something first. You must reason ONLY from the dated excerpts provided in this \
message. Do not use anything you remember from training about who originated \
this idea, when it became well known, or how it turned out. If the excerpts do \
not settle the question, say so and lower your confidence rather than filling \
the gap from memory. Recognising an idea is not evidence about its date."""


class LLM:
    def __init__(
        self,
        store=None,
        model: str | None = None,
        effort: str = "high",
        use_cache: bool = True,
        api_key: str | None = None,
        backend: str = "api",   # "api" | "cli"
    ):
        self.backend = backend
        if backend == "cli":
            self.client = None
            self.model = f"cli:{model or CLI_MODEL}"
        else:
            self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
            self.model = model or MODEL
        self.effort = effort
        self.store = store
        self.use_cache = use_cache
        self.calls = 0
        self.cache_hits = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def _key(self, system: str, user: str, schema: dict | None) -> str:
        h = hashlib.sha256()
        for part in (self.model, self.effort, system, user, json.dumps(schema, sort_keys=True)):
            h.update(str(part).encode())
            h.update(b"\x00")
        return h.hexdigest()

    def _call_cli(self, system: str, user: str, schema: dict | None, max_tokens: int) -> str:
        import json as _json
        import subprocess

        prompt = system + "\n\n" + user
        if schema is not None:
            prompt += ("\n\nRespond with ONLY a single JSON object -- no markdown "
                       "fences, no commentary -- valid against this JSON Schema:\n"
                       + _json.dumps(schema))
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        cli_model = self.model.split(":", 1)[1]
        last_err = ""
        for attempt in range(3):
            try:
                r = subprocess.run(
                    ["claude", "-p", "--model", cli_model, "--output-format", "json"],
                    input=prompt, capture_output=True, text=True, env=env, timeout=600,
                )
            except subprocess.TimeoutExpired:
                last_err = "timeout"
                continue
            if r.returncode != 0:
                last_err = (r.stderr or r.stdout or "")[:300]
                time.sleep(5 * (attempt + 1))
                continue
            try:
                wrapper = _json.loads(r.stdout)
                text = wrapper.get("result", "")
                usage = wrapper.get("usage") or {}
                self.input_tokens += int(usage.get("input_tokens") or 0)
                self.output_tokens += int(usage.get("output_tokens") or 0)
            except _json.JSONDecodeError:
                text = r.stdout
            self.calls += 1
            if schema is None:
                return text
            # Extract the JSON object even if the model wrapped it in prose.
            a, b = text.find("{"), text.rfind("}")
            if a != -1 and b > a:
                candidate = text[a : b + 1]
                try:
                    _json.loads(candidate)
                    return candidate
                except _json.JSONDecodeError:
                    pass
            last_err = f"unparseable output: {text[:200]}"
        raise RuntimeError(f"claude CLI failed after retries: {last_err}")

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(
            (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError)
        ),
        reraise=True,
    )
    def _call(self, system: str, user: str, schema: dict | None, max_tokens: int) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort},
            # A safety refusal on one document should degrade that document, not
            # abort a corpus-scale run; the server routes it to a fallback model.
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
        }
        if schema is not None:
            kwargs["output_config"]["format"] = {"type": "json_schema", "schema": schema}
        with self.client.beta.messages.stream(**kwargs) as stream:
            msg = stream.get_final_message()
        self.calls += 1
        self.input_tokens += msg.usage.input_tokens
        self.output_tokens += msg.usage.output_tokens
        if msg.stop_reason == "refusal":
            raise RefusalError(f"refused: {getattr(msg, 'stop_details', None)}")
        return next((b.text for b in msg.content if b.type == "text"), "")

    def json(
        self,
        system: str,
        user: str,
        schema: dict,
        max_tokens: int = 16000,
        grounded: bool = True,
    ) -> dict | None:
        """Structured call. Returns the parsed object, or None if it refused."""
        sys_prompt = f"{GROUNDING_RULE}\n\n{system}" if grounded else system
        key = self._key(sys_prompt, user, schema)
        if self.use_cache and self.store is not None:
            hit = self.store.cache_get(key)
            if hit is not None:
                self.cache_hits += 1
                try:
                    return json.loads(hit)
                except json.JSONDecodeError:
                    pass
        try:
            if self.backend == "cli":
                raw = self._call_cli(sys_prompt, user, schema, max_tokens)
            else:
                raw = self._call(sys_prompt, user, schema, max_tokens)
        except RefusalError:
            return None
        except RuntimeError:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if self.use_cache and self.store is not None:
            self.store.cache_put(key, self.model, raw)
        return data

    def stats(self) -> dict:
        # Opus 5 list pricing; on the CLI backend spend is subscription usage,
        # reported as the equivalent API value so runs stay comparable.
        cost = self.input_tokens / 1e6 * 5.0 + self.output_tokens / 1e6 * 25.0
        return {
            "calls": self.calls, "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "est_cost_usd": round(cost, 3),
        }


class RefusalError(Exception):
    pass
