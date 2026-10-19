"""ClinEnv with a System One decider inside the unchanged episode loop (same protocol as PhysicianBench).

ClinEnv's `run_stage` calls the model under test once per turn (`environment_controller._call_openai`). With the
S1CLIN patch there, that call goes through `complete()` below:
  arm llm          one sample per turn
  arm clef         K samples (n=K) from the same LLM; duplicates removed; Clef-27B chooses (scripts/clef_server.py)
  arm llm_select   K samples; the LLM chooses among its own candidates (online arm dropped 2026-10-19)
  arm llm_log      K samples; the first is executed (= arm llm); all K and the message history are logged
  arm s1_tool      Clef chooses the tool from the stage's tool menu; the LLM writes the call (tool_choice named)
Candidate rendering, de-duplication, seeded option letters and the deciders are shared with pb_decider, so both
benchmarks use one decision procedure. Each decision is logged to $S1CLIN_DECISIONS_LOG.

Environment:
  S1CLIN_ARM               llm | clef | llm_select | llm_log | s1_tool (default llm)
  S1CLIN_K                 candidates per turn (default 5)
  S1CLIN_MUT_TEMPERATURE   "default" = drop ClinEnv's temperature=0 and use the model's generation config
                           (needed for K distinct samples); a number = use it (e.g. 0 for the paper protocol)
  S1CLIN_MAX_COMPLETION_TOKENS  completion cap per call (default 16384)
  S1CLIN_DECISIONS_LOG     path of the decisions JSONL for this episode
  S1CLIN_SEED              seed prefix for option letters (e.g. "<subject>|<hadm>|run1")
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from src.s1clin.pb_decider import (CE_DECISION_QUESTION, CE_TOOL_QUESTION, MAX_CANDIDATE_CHARS, QUESTION_VERSION,
                                   ClefDecider, FirstDecider, LLMSelectDecider, MessageLog, _tc_name_args,
                                   render_tool, render_turn, tool_name, turn_key)

ARM = os.environ.get("S1CLIN_ARM", "llm")
K = int(os.environ.get("S1CLIN_K", "5"))
_state = {"step": 0}
CURRENT_STAGE = None  # set by the S1CLIN patch in environment_controller.run_episode (0-based stage index)
_msglog = MessageLog(Path(os.environ["S1CLIN_DECISIONS_LOG"]).with_suffix(".messages.jsonl")) \
    if os.environ.get("S1CLIN_DECISIONS_LOG") else None


def _write(rec):
    log = os.environ.get("S1CLIN_DECISIONS_LOG")
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _tool_step(chat_complete, client, model, messages, tools, kwargs, n_messages, epoch):
    """Arm s1_tool: Clef picks the tool from this stage's menu; the LLM writes that call."""
    options, names = [render_tool(t) for t in tools], [tool_name(t) for t in tools]
    seed = hashlib.sha256(f"{os.environ.get('S1CLIN_SEED', '')}|{_state['step']}".encode()).hexdigest()
    t0 = time.time()
    try:
        pick, dist, info = ClefDecider(question=CE_TOOL_QUESTION).choose(messages, options, seed)
    except Exception as e:  # decider failure: the LLM's own unconstrained turn (an `llm`-arm draw)
        pick, dist, info = None, None, {"error": repr(e)[:500]}
    decide_s = round(time.time() - t0, 2)
    kw = dict(kwargs)
    if pick is not None:
        kw["tool_choice"] = {"type": "function", "function": {"name": names[pick]}}
    resp = chat_complete(client, model=model, messages=messages, tools=tools, **kw)
    m = resp.choices[0].message
    _write({"step": _state["step"], "options": names, "pick": pick, "tool": names[pick] if pick is not None else None,
            "distribution": dist, "decider": os.environ.get("S1CLIN_DECIDER", "clef"), "decide_s": decide_s, "info": info,
            "executed": render_turn(m.content, m.tool_calls)[:MAX_CANDIDATE_CHARS],
            "followed": pick is None or any(_tc_name_args(tc)[0] == names[pick] for tc in (m.tool_calls or [])),
            "prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens,
            "n_messages": n_messages, "msg_epoch": epoch, "stage": CURRENT_STAGE, "q": QUESTION_VERSION})
    return resp


class _SelectChat:
    """Adapter giving LLMSelectDecider the .chat() interface it uses on PhysicianBench."""

    def __init__(self, chat_complete, client, model, max_tokens):
        self.f, self.client, self.model, self.max_tokens = chat_complete, client, model, max_tokens

    def chat(self, messages, tools=None, reasoning_effort=None):
        r = self.f(self.client, model=self.model, messages=messages, max_completion_tokens=self.max_tokens)
        return SimpleNamespace(content=r.choices[0].message.content, prompt_tokens=r.usage.prompt_tokens,
                               completion_tokens=r.usage.completion_tokens)


def complete(chat_complete, client, model, messages, tools, kwargs):
    """Drop-in for ClinEnv's MUT chat call. Returns a response whose choices[0] is the executed turn."""
    kwargs = dict(kwargs)
    t = os.environ.get("S1CLIN_MUT_TEMPERATURE", "default")
    if t == "default":
        kwargs.pop("temperature", None)
    else:
        kwargs["temperature"] = float(t)
    max_tokens = int(os.environ.get("S1CLIN_MAX_COMPLETION_TOKENS", "16384"))
    kwargs.setdefault("max_completion_tokens", max_tokens)
    if ARM == "llm":
        return chat_complete(client, model=model, messages=messages, tools=tools, **kwargs)

    _state["step"] += 1
    n_messages, epoch = _msglog.sync(messages) if _msglog else (len(messages), 0)
    if ARM == "s1_tool":
        return _tool_step(chat_complete, client, model, messages, tools, kwargs, n_messages, epoch)
    resp = chat_complete(client, model=model, messages=messages, tools=tools, n=K, **kwargs)
    choices = resp.choices
    keys, uniq, counts = [], [], []
    for i, ch in enumerate(choices):
        key = turn_key(ch.message.content, ch.message.tool_calls)
        if key in keys:
            counts[keys.index(key)] += 1
        else:
            keys.append(key); uniq.append(i); counts.append(1)
    renderings = [render_turn(choices[i].message.content, choices[i].message.tool_calls)[:MAX_CANDIDATE_CHARS]
                  for i in uniq]
    rec = {"step": _state["step"], "n_samples": len(choices), "n_unique": len(uniq), "counts": counts,
           "candidates": renderings, "finish_reasons": [choices[i].finish_reason for i in uniq],
           "prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens,
           "n_messages": n_messages, "msg_epoch": epoch, "stage": CURRENT_STAGE, "q": QUESTION_VERSION}
    t0 = time.time()
    if len(uniq) == 1:
        pick, dist, info = 0, [1.0], {"forced": True}
    else:
        seed = hashlib.sha256(f"{os.environ.get('S1CLIN_SEED', '')}|{_state['step']}".encode()).hexdigest()
        decider = (ClefDecider(question=CE_DECISION_QUESTION) if ARM == "clef" else FirstDecider() if ARM == "llm_log"
                   else LLMSelectDecider(_SelectChat(chat_complete, client, model, max_tokens), None))
        try:
            pick, dist, info = decider.choose(messages, renderings, seed)
        except Exception as e:  # fall back to the first sample, an `llm`-arm draw
            pick, dist, info = 0, None, {"error": repr(e)[:500]}
    rec.update(pick=pick, distribution=dist, decider=ARM, decide_s=round(time.time() - t0, 2), info=info)
    _write(rec)
    return SimpleNamespace(choices=[choices[uniq[pick]]], usage=resp.usage)
