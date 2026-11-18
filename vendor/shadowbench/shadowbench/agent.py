"""Run ShadowBench episodes against any OpenAI-compatible Chat Completions endpoint with tool calling and image input.

Works with OpenAI, OpenRouter, Anthropic's OpenAI-compatible endpoint, vLLM, SGLang and others. The model gets the
nine environment tools and nothing else. The protocol is the one of the published evaluation:

* system = SYSTEM_PROMPT and the tools are TOOL_SPECS, both from `shadowbench.reviewed_environment`; the first user
  message is USER_PROMPT with the first observation (the target and the hands in their starting pose).
* The model calls tools until it submits or the 100-call budget runs out (the environment then submits). A call made
  after the episode has ended is answered "Episode already ended." and does not count.
* If the model ends its turn without submitting it gets CONTINUE, at most three times; after that the pose it
  reached is scored (`auto_submitted`), as the environment does when the call budget runs out.
* After the episode ends the model gets the last tool results and one more turn, for its closing message.
* Chat Completions tool messages carry text only, so an image a tool returns (`look`, `peek_book`) is sent in a user
  message right after the tool results ("Image returned by look (call <id>)."). In the published evaluation the
  image was attached to the tool result itself.

Each episode directory receives the environment's log (trajectory.jsonl with every call, the pose before and after
it, whether it was accepted and the hidden score; images/ with every image the model received; states/; final.json;
task.json), plus initial.png, result.json and transcript.json (the conversation without image bytes).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time
import traceback
from collections import Counter
from pathlib import Path

from .factory import create
from .reviewed_environment import SYSTEM_PROMPT, TOOL_SPECS

USER_PROMPT = (
    "Reproduce the fixed target silhouette in the upper-left panel. Other panels show your starting hands and shadow. "
    "Use the tools, then submit. It is a hand-shadow puppet: first call plan with what it depicts and which hand parts "
    "will form which of its features. peek_book shows the original book figure, at a cost to your final score."
)
CONTINUE = "Continue manipulating the hands, then call submit."
CONTINUES = 3


def _png_url(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def openai_tools() -> list[dict]:
    return [{"type": "function", "function": {"name": n, "description": d, "parameters": p}}
            for n, (d, p) in TOOL_SPECS.items()]


def parse_arguments(raw) -> dict:
    """Arguments as the environment receives them; unparseable JSON is refused by the environment, as in the evaluation."""
    if isinstance(raw, dict):
        return raw
    try:
        args = json.loads(raw) if raw and raw.strip() else {}
    except (json.JSONDecodeError, TypeError):
        return {"invalid_arguments": True}
    return args if isinstance(args, dict) else {"invalid_arguments": True}


class Pricing:
    """USD per 1M tokens. If the endpoint reports a cost per response (OpenRouter's `usage.cost`), that value wins."""

    def __init__(self, input: float | None = None, output: float | None = None, cached_input: float | None = None):
        self.input, self.output, self.cached_input = input, output, cached_input

    def cost(self, prompt: int, cached: int, completion: int) -> float | None:
        if self.input is None or self.output is None:
            return None
        ci = self.cached_input if self.cached_input is not None else self.input
        return ((prompt - cached) * self.input + cached * ci + completion * self.output) / 1e6


def _get(o, *path, default=0):
    for p in path:
        if o is None:
            return default
        o = o.get(p) if isinstance(o, dict) else getattr(o, p, None)
    return default if o is None else o


async def run_episode(client, model: str, task_id: str, out_dir: Path, *, label: str | None = None,
                      reasoning_effort: str | None = None, extra_body: dict | None = None,
                      pricing: Pricing | None = None, timeout_s: float = 3600, max_turns: int = 300,
                      max_tokens: int | None = None, revision: str | None = None) -> dict:
    """Run one episode with an `openai.AsyncOpenAI` client. `model` is the endpoint's model id; `label` is the name
    written to result.json (defaults to `model`). `revision` pins a certified target revision (default: the one
    the published evaluation used)."""
    out_dir = Path(out_dir)
    if (out_dir / "result.json").exists() or (out_dir / "trajectory.jsonl").exists():
        raise FileExistsError(f"{out_dir} already holds an episode")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = create(task_id, out_dir, revision=revision)
    initial = env.observe()
    initial.save(out_dir / "initial.png")
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": [{"type": "text", "text": USER_PROMPT},
                                                         {"type": "image_url", "image_url": {"url": _png_url(initial)}}]}]
    transcript: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": USER_PROMPT, "step": 0}]
    usage = dict(input=0, output=0, cache_read=0, reasoning=0, cost_usd=0.0, calls=0)
    cost_known = True
    response_models: set = set()
    status, exc_text, continues, auto = "ok", None, 0, False
    t0 = time.time()
    kw = dict(model=model, tools=openai_tools(), tool_choice="auto")
    if reasoning_effort:
        kw["reasoning_effort"] = reasoning_effort
    if extra_body:
        kw["extra_body"] = extra_body
    if max_tokens:
        kw["max_completion_tokens"] = max_tokens
    try:
        for _ in range(max_turns):
            if time.time() - t0 > timeout_s:
                status = "timeout"
                break
            ended = env.done
            resp = await client.chat.completions.create(messages=messages, **kw)
            u = resp.usage
            usage["calls"] += 1
            pt, ct = _get(u, "prompt_tokens"), _get(u, "completion_tokens")
            cached = _get(u, "prompt_tokens_details", "cached_tokens")
            usage["input"] += pt
            usage["output"] += ct
            usage["cache_read"] += cached
            usage["reasoning"] += _get(u, "completion_tokens_details", "reasoning_tokens")
            reported = _get(u, "cost", default=None)
            c = float(reported) if reported is not None else (pricing.cost(pt, cached, ct) if pricing else None)
            if c is None:
                cost_known = False
            else:
                usage["cost_usd"] += c
            if getattr(resp, "model", None):
                response_models.add(resp.model)
            msg = resp.choices[0].message
            calls = list(msg.tool_calls or [])
            am = {"role": "assistant", "content": msg.content or ""}
            if calls:
                am["tool_calls"] = [{"id": tc.id, "type": "function",
                                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                    for tc in calls]
            messages.append(am)
            transcript.append(dict(am, reasoning=getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None),
                                   finish_reason=resp.choices[0].finish_reason))
            if ended:  # the closing turn after the episode ended
                break
            if not calls:
                if continues >= CONTINUES:
                    break
                continues += 1
                messages.append({"role": "user", "content": CONTINUE})
                transcript.append({"role": "user", "content": CONTINUE})
                continue
            images = []
            for tc in calls:
                before = len(env.steps)
                out = await asyncio.to_thread(env.call, tc.function.name, parse_arguments(tc.function.arguments))
                step = before if len(env.steps) > before else None
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": out.text})
                transcript.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name,
                                   "content": out.text, "step": step})
                if out.image is not None:
                    images.append((tc.id, tc.function.name, out.image))
            if images:
                content = []
                for cid, name, img in images:
                    content += [{"type": "text", "text": f"Image returned by {name} (call {cid})."},
                                {"type": "image_url", "image_url": {"url": _png_url(img)}}]
                messages.append({"role": "user", "content": content})
        else:
            status = "max_turns"
    except Exception as e:
        status = f"error: {type(e).__name__}: {e}"
        exc_text = traceback.format_exc()
        (out_dir / "error.txt").write_text(exc_text)
    failed = status != "ok"
    if not env.done and not failed:
        # The model stopped without submitting: score the pose it reached, as the call budget would.
        with env.lock:
            env.done = True
            env.final = {**env.metrics(), "auto_submitted": True}
            (out_dir / "final.json").write_text(json.dumps(env.final, indent=2))
        auto = True
    result = dict(model=label or model, endpoint_model=model, response_models=sorted(response_models),
                  task_id=task_id, target_revision=env.task.metadata["revision"], effort=reasoning_effort,
                  status=status, failed=failed, auto_submitted=auto, seconds=round(time.time() - t0, 1),
                  usage=dict(usage, cost_usd=round(usage["cost_usd"], 6) if cost_known else None),
                  calls=env.calls, images=env.images, refusals=sum(not s["accepted"] for s in env.steps[1:]),
                  tool_counts=dict(Counter(s["tool"] for s in env.steps[1:])),
                  final=env.final if env.done and not failed else None, env_fingerprint=env.version,
                  error=(exc_text or "")[-2000:] or None)
    (out_dir / "result.json").write_text(json.dumps(result, indent=1, default=str))
    (out_dir / "transcript.json").write_text(json.dumps(transcript, indent=1, default=str))
    return result
