"""PhysicianBench with a System One model as the decision maker inside the unchanged MiniAgent loop (protocol A).

MiniAgent calls `client.chat(messages, tools, ...)` once per step. `DeciderClient` stands in for that client:
  1. it samples K candidate next turns from the same local LLM, in one request with n=K and the provider-default
     sampling (the leaderboard setting); each candidate is a complete assistant turn (a batch of tool calls with
     arguments, or a final answer);
  2. it removes duplicates; if only one distinct candidate remains, that candidate is taken;
  3. otherwise a decider picks one: Clef-27B (`clef`, via scripts/clef_server.py) or the LLM itself (`llm_select`);
  4. it returns the chosen candidate as an ordinary ChatResponse, so MiniAgent, the tools, the trajectory log
     and the graders behave exactly as in the leaderboard harness.
The arm `llm` (the leaderboard baseline) uses the unwrapped LLMClient. Picking one of K i.i.d. samples at random
is the same distribution as `llm`, so no random arm is needed.

Each decision is logged to <job>/logs/agent/decisions.jsonl: candidate renderings, duplicate counts, the decider's
distribution, the choice and the input size the decider saw.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import string
import time
from pathlib import Path
from types import SimpleNamespace

import requests

LETTERS = string.ascii_uppercase
CLEF_URL = "http://127.0.0.1:8021/decide"
# System One deciders behind the same /decide API (scripts/clef_server.py for Clef, scripts/s1_proxy.py for the others).
# A run picks one with S1CLIN_DECIDER (pb_run/clinenv_run --decider); unset means Clef.
DECIDER_URLS = {"clef": CLEF_URL, **{name: f"http://127.0.0.1:{port}/decide" for name, port in (
    ("pplx12", 8022), ("d3", 8023), ("linne", 8024), ("decider31b", 8025), ("amos", 8026))}}
MAX_CANDIDATE_CHARS = 12_000   # rendering cap per candidate (a full consult note fits)

DECISION_QUESTION = (
    "The state shows a clinical task assigned to an AI agent that works in an electronic health record "
    "through FHIR tools, followed by the agent's actions and their results so far. Each option is a "
    "candidate next action for the agent. Choose the candidate that best advances completing the task "
    "correctly, completely and safely.")
TOOL_QUESTION = (
    "The state shows a clinical task assigned to an AI agent that works in an electronic health record "
    "through FHIR tools, followed by the agent's actions and their results so far. Each option is a tool "
    "the agent can use for its next action, or finishing with a final answer. Choose what the agent "
    "should do next to complete the task correctly, completely and safely.")
_CE_STATE = (
    "The state shows an attending-physician agent assessing a hospitalized patient one stage at a time: "
    "it gathers information by asking the patient and the nurse, ordering labs and reviewing prior "
    "admissions, then submits its medication, procedure and diagnosis decisions for the stage and "
    "finalizes it. The agent's actions and their results so far follow. ")
CE_DECISION_QUESTION = _CE_STATE + (
    "Each option is a candidate next action for the agent. Choose the candidate that best advances "
    "completing the stage correctly, completely and safely.")
CE_TOOL_QUESTION = _CE_STATE + (
    "Each option is a tool the agent can use for its next action. Choose the tool the agent should use "
    "next to complete the stage correctly, completely and safely.")
QUESTION_VERSION = 2  # 1: ClinEnv used DECISION_QUESTION (PhysicianBench wording) until 2026-10-19
FINISH_OPTION = "FINISH: stop calling tools and give the final answer (ends the task)"
MAX_TOOL_DESC_CHARS = 300

SELECT_PROMPT = (
    "Before acting, compare these candidate next actions for this task. Each is a complete next turn "
    "(tool calls with their arguments, or a final answer).\n\n{options}\n\n"
    "Which candidate best advances completing the task correctly, completely and safely? "
    "Reply with the letter only.")


# ---------------------------------------------------------------- rendering

def render_turn(content: str | None, tool_calls) -> str:
    """One assistant turn as text: tool calls with full arguments, or the final answer."""
    if not tool_calls:
        return "FINISH with final answer: " + (content or "").strip()
    lines = []
    for tc in tool_calls:
        name, raw = _tc_name_args(tc)
        try:
            args = json.loads(raw) if raw else {}
            arg_s = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
        except json.JSONDecodeError:
            arg_s = raw
        lines.append(f"{name}({arg_s})")
    return "\n".join(lines)


def parse_turn(r: str) -> list[tuple[str, dict]]:
    """Invert render_turn: 'name(k=<json>, ...)' per line -> [(name, args)]; a final answer -> [("FINISH", {})]."""
    if r.startswith("FINISH with final answer:"):
        return [("FINISH", {})]
    calls, dec = [], json.JSONDecoder()
    for line in r.split("\n"):
        m = re.match(r"(\w+)\((.*)\)$", line, re.S)
        if not m:
            continue
        name, s, args, i = m.group(1), m.group(2), {}, 0
        while i < len(s):
            k = re.match(r"\s*(\w+)=", s[i:])
            if not k:
                break
            i += k.end()
            try:
                v, j = dec.raw_decode(s, i)
            except json.JSONDecodeError:
                break
            args[k.group(1)], i = v, j
            i += len(re.match(r"\s*,?\s*", s[i:]).group(0))
        calls.append((name, args))
    return calls


def render_tool(tool: dict) -> str:
    """A tool option: name and the first paragraph of its description."""
    f = tool.get("function", tool)
    desc = " ".join((f.get("description") or "").strip().split("\n\n")[0].split())
    return f"{f['name']}: {desc[:MAX_TOOL_DESC_CHARS]}"


def tool_name(tool: dict) -> str:
    return tool.get("function", tool)["name"]


def _tc_name_args(tc):
    if isinstance(tc, dict):
        return tc["function"]["name"], tc["function"].get("arguments") or ""
    return tc.function.name, tc.function.arguments or ""


def turn_key(content: str | None, tool_calls) -> str:
    """Canonical identity of a turn, for removing duplicate candidates."""
    if not tool_calls:
        return "final:" + " ".join((content or "").split())
    calls = []
    for tc in tool_calls:
        name, raw = _tc_name_args(tc)
        try:
            raw = json.dumps(json.loads(raw), sort_keys=True)
        except json.JSONDecodeError:
            pass
        calls.append(f"{name}:{raw}")
    return "calls:" + "\n".join(calls)


def history_steps(messages: list[dict]) -> tuple[str, list[dict]]:
    """Split MiniAgent messages into the task instruction and the steps so far.

    Each step is {"action": rendered turn, "results": [tool result strings as the LLM saw them]}.
    Later user turns (ClinEnv stage prompts and environment nudges) become steps of their own with action
    "ENVIRONMENT MESSAGE", so multi-stage histories keep their order. PhysicianBench has a single user turn.
    """
    first = next(i for i, m in enumerate(messages) if m["role"] == "user")
    instruction = messages[first]["content"]
    steps = []
    for m in messages[first + 1:]:
        if m["role"] == "assistant":
            steps.append({"action": render_turn(m.get("content"), m.get("tool_calls")), "results": []})
        elif m["role"] == "tool" and steps:
            steps[-1]["results"].append(m["content"] or "")
        elif m["role"] == "user":
            steps.append({"action": "ENVIRONMENT MESSAGE", "results": [m["content"] or ""]})
    return instruction, steps


# ---------------------------------------------------------------- deciders

class ClefDecider:
    """A System One decider over HTTP: Clef by default, or the model named by S1CLIN_DECIDER."""

    def __init__(self, url: str | None = None, timeout: int = 600, question: str = DECISION_QUESTION):
        self.name = os.environ.get("S1CLIN_DECIDER", "clef")
        self.url, self.timeout, self.question = url or DECIDER_URLS[self.name], timeout, question

    def choose(self, messages, renderings, seed):
        instruction, steps = history_steps(messages)
        letters = _letters(len(renderings), seed)
        payload = {"instruction": instruction, "steps": steps, "question": self.question,
                   "options": {letters[i]: r for i, r in enumerate(renderings)}}
        r = requests.post(self.url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        out = r.json()
        dist = [out["distribution"][letters[i]] for i in range(len(renderings))]
        return max(range(len(dist)), key=dist.__getitem__), dist, {
            "letters": letters, "input_tokens": out.get("input_tokens"), "fit": out.get("fit")}


class LLMSelectDecider:
    """The same LLM picks among its own candidates, seeing its full message history (generative readout)."""
    name = "llm_select"

    def __init__(self, client, reasoning_effort: str | None):
        self.client, self.reasoning_effort = client, reasoning_effort

    def choose(self, messages, renderings, seed):
        letters = _letters(len(renderings), seed)
        order = sorted(range(len(renderings)), key=lambda i: letters[i])
        opts = "\n\n".join(f"Candidate {letters[i]}:\n{renderings[i]}" for i in order)
        msgs = messages + [{"role": "user", "content": SELECT_PROMPT.format(options=opts)}]
        resp = self.client.chat(msgs, tools=None, reasoning_effort=self.reasoning_effort)
        text = (resp.content or "").strip()
        m = re.search(r"\b([A-Z])\b", text)
        pick = letters.index(m.group(1)) if m and m.group(1) in letters else 0
        dist = [1.0 if i == pick else 0.0 for i in range(len(renderings))]
        return pick, dist, {"letters": letters, "reply": text[:200], "parsed": bool(m and m.group(1) in letters),
                            "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens}


class FirstDecider:
    """Arm llm_log: execute the first of the K i.i.d. samples (the same distribution as the `llm` arm); the K
    candidates and the message history are logged for the offline decision evaluation."""
    name = "first"

    def choose(self, messages, renderings, seed):
        return 0, None, {}


def _jsonable(o):
    return o.model_dump() if hasattr(o, "model_dump") else str(o)


class MessageLog:
    """Append-only log of the message history the LLM saw, so every logged decision state can be rebuilt offline.

    Lines are {"op": "add", "msg": ...} or {"op": "reset", "keep": n} (the history was rebuilt, e.g. a new ClinEnv
    stage, and only its first n messages are shared with the logged one). A decision row's `n_messages` and
    `msg_epoch` locate its state: replay the log up to that epoch and take the first n_messages messages.
    """

    def __init__(self, path):
        self.path, self.logged, self.epoch = Path(path), [], 0

    def sync(self, messages) -> tuple[int, int]:
        cur = [json.dumps(m, ensure_ascii=False, default=_jsonable, sort_keys=True) for m in messages]
        lcp = 0
        while lcp < min(len(cur), len(self.logged)) and cur[lcp] == self.logged[lcp]:
            lcp += 1
        lines = []
        if lcp < len(self.logged):
            self.epoch += 1
            lines.append(json.dumps({"op": "reset", "keep": lcp}))
        lines += ['{"op": "add", "msg": ' + m + "}" for m in cur[lcp:]]
        if lines:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write("\n".join(lines) + "\n")
        self.logged = cur
        return len(cur), self.epoch


def _letters(k: int, seed: str) -> list[str]:
    """Seeded random letter per candidate. Clef presents choice options sorted by name, so this sets the order."""
    ls = list(LETTERS[:k])
    random.Random(seed).shuffle(ls)
    return ls


# ---------------------------------------------------------------- client wrapper

class DeciderClient:
    def __init__(self, base, decider, k: int, log_path: Path, seed: str):
        self.base, self.decider, self.k = base, decider, k
        self.model_id = f"{base.model_id}+{decider.name}@k{k}"
        self.log_path, self.seed, self.step = Path(log_path), seed, 0
        self.msglog = MessageLog(Path(log_path).with_suffix(".messages.jsonl"))

    def chat(self, messages, tools=None, temperature=None, parallel_tool_calls=True, reasoning_effort=None, **kw):
        self.step += 1
        n_messages, epoch = self.msglog.sync(messages)
        resp = self.base.chat(messages, tools=tools, temperature=temperature,
                              parallel_tool_calls=parallel_tool_calls, reasoning_effort=reasoning_effort, n=self.k)
        choices = resp.raw.choices
        keys, uniq, counts = [], [], []
        for i, ch in enumerate(choices):
            key = turn_key(ch.message.content, ch.message.tool_calls)
            if key in keys:
                counts[keys.index(key)] += 1
            else:
                keys.append(key); uniq.append(i); counts.append(1)
        renderings = [render_turn(choices[i].message.content, choices[i].message.tool_calls)[:MAX_CANDIDATE_CHARS]
                      for i in uniq]
        rec = {"step": self.step, "n_samples": len(choices), "n_unique": len(uniq), "counts": counts,
               "candidates": renderings, "finish_reasons": [choices[i].finish_reason for i in uniq],
               "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens,
               "n_messages": n_messages, "msg_epoch": epoch, "q": QUESTION_VERSION}
        t0 = time.time()
        if len(uniq) == 1:
            pick, dist, info = 0, [1.0], {"forced": True}
        else:
            seed = hashlib.sha256(f"{self.seed}|{self.step}".encode()).hexdigest()
            try:
                pick, dist, info = self.decider.choose(messages, renderings, seed)
            except Exception as e:  # decider failure: fall back to the first sample, which is an `llm`-arm draw
                pick, dist, info = 0, None, {"error": repr(e)[:500]}
        rec.update(pick=pick, distribution=dist, decider=self.decider.name, decide_s=round(time.time() - t0, 2),
                   **{"info": info})
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        from agent.llm_client import ChatResponse  # PhysicianBench only; ClinEnv imports this module too
        chosen = choices[uniq[pick]]
        raw = SimpleNamespace(choices=[chosen], usage=resp.raw.usage)
        return ChatResponse(content=chosen.message.content, tool_calls=chosen.message.tool_calls,
                            prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens, raw=raw)


class ToolChoiceClient:
    """Arm s1_tool: the System One model chooses the tool, the LLM writes the call.

    Each step Clef sees the state and the full tool menu (plus FINISH) and picks one; the LLM then generates with
    every tool still defined in its prompt but constrained to the chosen one: tool_choice="required" keeping only
    calls of that tool (parallel calls of the same tool survive), or, if it called none, a regeneration with the
    tool named; FINISH means tool_choice="none" (a final answer).
    """

    def __init__(self, base, decider, log_path: Path, seed: str):
        self.base, self.decider = base, decider
        self.model_id = f"{base.model_id}+{decider.name}@tool"
        self.log_path, self.seed, self.step = Path(log_path), seed, 0
        self.msglog = MessageLog(Path(log_path).with_suffix(".messages.jsonl"))

    def chat(self, messages, tools=None, temperature=None, parallel_tool_calls=True, reasoning_effort=None, **kw):
        self.step += 1
        n_messages, epoch = self.msglog.sync(messages)
        tools = tools or []
        options = [render_tool(t) for t in tools] + [FINISH_OPTION]
        names = [tool_name(t) for t in tools] + ["FINISH"]
        seed = hashlib.sha256(f"{self.seed}|{self.step}".encode()).hexdigest()
        t0 = time.time()
        try:
            pick, dist, info = self.decider.choose(messages, options, seed)
        except Exception as e:  # decider failure: the LLM's own unconstrained turn (an `llm`-arm draw)
            pick, dist, info = None, None, {"error": repr(e)[:500]}
        decide_s = round(time.time() - t0, 2)
        common = dict(tools=tools, temperature=temperature, parallel_tool_calls=parallel_tool_calls,
                      reasoning_effort=reasoning_effort)
        regen = 0
        if pick is None:
            resp = self.base.chat(messages, **common)
            calls = resp.tool_calls
        elif names[pick] == "FINISH":
            resp = self.base.chat(messages, tool_choice="none", **common)
            calls = None
        else:
            resp = self.base.chat(messages, tool_choice="required", **common)
            calls = [tc for tc in (resp.tool_calls or []) if _tc_name_args(tc)[0] == names[pick]]
            if not calls:
                regen = 1
                resp = self.base.chat(messages, tool_choice={"type": "function", "function": {"name": names[pick]}},
                                      **common)
                calls = [tc for tc in (resp.tool_calls or []) if _tc_name_args(tc)[0] == names[pick]] or None
        chosen = resp.raw.choices[0]
        chosen.message.tool_calls = calls
        rec = {"step": self.step, "options": names, "pick": pick, "tool": names[pick] if pick is not None else None,
               "distribution": dist, "decider": self.decider.name, "decide_s": decide_s, "info": info,
               "executed": render_turn(chosen.message.content, calls)[:MAX_CANDIDATE_CHARS], "regenerated": regen,
               "dropped_calls": len(resp.tool_calls or []) - len(calls or []) if not regen else None,
               "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens,
               "n_messages": n_messages, "msg_epoch": epoch, "q": QUESTION_VERSION}
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        from agent.llm_client import ChatResponse
        return ChatResponse(content=chosen.message.content, tool_calls=calls, prompt_tokens=resp.prompt_tokens,
                            completion_tokens=resp.completion_tokens,
                            raw=SimpleNamespace(choices=[chosen], usage=resp.raw.usage))


class LLMToolClient:
    """Arm llm_tool (control for s1_tool): the same one-tool-per-turn restriction, but the LLM chooses the tool.

    The LLM samples its turn as usual (all tools, tool_choice auto); only the calls of the turn's first tool are kept
    (parallel calls of that tool survive), exactly the filtering s1_tool applies after Clef's choice. A turn without
    tool calls is a final answer (FINISH). s1_tool vs llm_tool isolates who chooses the tool; llm vs llm_tool
    isolates the restriction itself.
    """

    def __init__(self, base, log_path: Path, seed: str):
        self.base = base
        self.model_id = f"{base.model_id}+llm@tool"
        self.log_path, self.seed, self.step = Path(log_path), seed, 0
        self.msglog = MessageLog(Path(log_path).with_suffix(".messages.jsonl"))

    def chat(self, messages, tools=None, temperature=None, parallel_tool_calls=True, reasoning_effort=None, **kw):
        self.step += 1
        n_messages, epoch = self.msglog.sync(messages)
        tools = tools or []
        names = [tool_name(t) for t in tools] + ["FINISH"]
        resp = self.base.chat(messages, tools=tools, temperature=temperature, parallel_tool_calls=parallel_tool_calls,
                              reasoning_effort=reasoning_effort)
        raw_calls = resp.tool_calls or []
        tool = _tc_name_args(raw_calls[0])[0] if raw_calls else "FINISH"
        calls = [tc for tc in raw_calls if _tc_name_args(tc)[0] == tool] or None
        chosen = resp.raw.choices[0]
        chosen.message.tool_calls = calls
        rec = {"step": self.step, "options": names, "pick": names.index(tool) if tool in names else None,
               "tool": tool, "distribution": None, "decider": "llm_first_tool", "decide_s": 0.0, "info": {},
               "executed": render_turn(chosen.message.content, calls)[:MAX_CANDIDATE_CHARS], "regenerated": 0,
               "dropped_calls": len(raw_calls) - len(calls or []),
               "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens,
               "n_messages": n_messages, "msg_epoch": epoch, "q": QUESTION_VERSION}
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        from agent.llm_client import ChatResponse
        return ChatResponse(content=chosen.message.content, tool_calls=calls, prompt_tokens=resp.prompt_tokens,
                            completion_tokens=resp.completion_tokens,
                            raw=SimpleNamespace(choices=[chosen], usage=resp.raw.usage))
