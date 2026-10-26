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
import math
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


ORDER_TOOLS = {"fhir_service_request_create", "fhir_medication_request_create", "fhir_appointment_create",
               "fhir_communication_create_message"}


def turn_kind(raw: str) -> str:
    """act (places an order), document (writes the note), finish, or retrieve; from a rendered candidate turn."""
    names = {n for n, _ in parse_turn(raw)}
    if raw.startswith("write_file(") or "\nwrite_file(" in raw:  # a multi-line note does not parse as one call line
        names.add("write_file")
    if names & ORDER_TOOLS:
        return "act"
    if "write_file" in names:
        return "document"
    if names == {"FINISH"}:
        return "finish"
    return "retrieve"


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
        # S1CLIN_DECIDER_URL: a further instance of the same decider (scripts/s1_up.sh NAME GPU N)
        self.url = url or os.environ.get("S1CLIN_DECIDER_URL") or DECIDER_URLS[self.name]
        self.timeout, self.question = timeout, question

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


class GatedDecider:
    """Arm gate: the System One decider overrules the LLM's own sample only at commitment points (some candidate places
    an order or finishes) and only when it prefers its pick to that sample by a log-odds margin; otherwise the first
    sample, an `llm`-arm draw, is executed. Offline counterpart: route+gate in scripts/pb_combine.py. The logged
    distribution is the decider's; `info.decider_pick` is its argmax, `info.gate` what happened."""

    POINTS = ("commit", "orders")  # orders: only where a candidate creates a service or medication request, so
    # FINISH and notes stay with the LLM (offline counterpart: orders(D) in pb_combine.py; docs/research_loop.md P2')

    def __init__(self, decider, margin: float, points: str = "commit"):
        assert points in self.POINTS
        self.decider, self.margin, self.points = decider, margin, points
        self.name = f"{decider.name}+gate{margin:g}" + ("" if points == "commit" else f"_{points}")

    def point(self, renderings):
        if self.points == "orders":
            return any(n in ("fhir_service_request_create", "fhir_medication_request_create")
                       for r in renderings for n, _ in parse_turn(r))
        return any(turn_kind(r) in ("act", "finish") for r in renderings)

    def choose(self, messages, renderings, seed):
        if not self.point(renderings):
            return 0, None, {"gate": "not a commitment point"}
        pick, dist, info = self.decider.choose(messages, renderings, seed)
        lo = math.log(max(dist[pick], 1e-12)) - math.log(max(dist[0], 1e-12))
        go = lo >= self.margin
        return (pick if go else 0), dist, {**info, "decider_pick": pick, "log_odds": round(lo, 4),
                                           "gate": "overrule" if go else "defer"}


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


# ---------------------------------------------------------------- screening (research loop P7)

SCREEN_QUESTION = (
    "The state shows a clinical task assigned to an AI agent that works in an electronic health record through FHIR "
    "tools, with the agent's actions and the information it has gathered so far. Consider this order for the "
    "patient: {order}. Is this order indicated for this patient as part of completing the task (whether or not the "
    "agent has already placed it)?")
SCREEN_YES, SCREEN_NO = "Yes, the order is indicated", "No, the order is not indicated"
SCREEN_STATE_CHARS = 28_000  # about 8k tokens for instruction, actions and results together
SCREEN_BLIND = "A clinical task in an electronic health record. No further information is available."
CREATE_ARGS = {"fhir_service_request_create": ("service", "code_display", "code_code"),
               "fhir_medication_request_create": ("medication", "medication_display", "medication_code")}
CHECKLIST = (
    "Before you finish: a screening model reviewed orders that came up while working on this task and flagged these "
    "as possibly indicated for this patient and not yet placed:\n{items}\n"
    "Check each against the chart and the task. Place any that are indicated, and only those, then finish.")
# P10 / O1, the factorized screen: whether a candidate repeats a placed order is asked on its own, from the placed
# orders alone, so the agent's orders never enter the indication judgment (G6: they anchor it)
DEDUP_INSTRUCTION = "Compare a candidate clinical order with the orders already placed for the same patient."
DEDUP_QUESTION = ("Candidate order: {order}. Is it the same as, or equivalent to (the same laboratory test, imaging "
                  "study, referral or consult, procedure, or drug), one of the orders already placed?")
DEDUP_YES, DEDUP_NO = "Yes, it repeats an order already placed", "No, it is a new order"
# items with P(repeat) >= DEDUP_MAX are dropped; S1CLIN_REPEAT_JUDGE=none (R-Gemma) keeps only the exact-name filter
DEDUP_DECIDER, DEDUP_MAX = os.environ.get("S1CLIN_REPEAT_JUDGE", "pplx12"), 0.5
# R, the restart: at the first FINISH the LLM is asked once more for the orders the task requires, from its context
# with its own decisions removed (`restart-fresh`) or from the full context (`restart-full`, control); only the order
# calls of that answer are executed, minus exact repeats of placed orders and minus those DEDUP_DECIDER, given only
# the placed orders, judges a repeat (offline counterpart: pb_replay.py A2, where the fresh answer repeated 1.15
# placed or met orders per sample, 0.35 of them by exact name)
RESTART_ASK = ("Based on the task and the patient data gathered so far, place every order that this task requires and "
               "that has not been placed yet, then finish.")
DECISION_TOOLS = ("_create", "write_file")  # orders, messages, appointments, the note


def fresh_context(messages: list[dict]) -> list[dict]:
    """The messages with every decision turn (a tool call ending in DECISION_TOOLS) and the tool results answering it
    removed: the system prompt, the task and the retrieval turns remain."""
    out, drop = [], set()
    for m in messages:
        calls = m.get("tool_calls") or []
        if m["role"] == "assistant" and any(_tc_name_args(tc)[0].endswith(DECISION_TOOLS) for tc in calls):
            drop |= {tc["id"] if isinstance(tc, dict) else tc.id for tc in calls}
        elif m["role"] == "tool" and m.get("tool_call_id") in drop:
            continue
        else:
            out.append(m)
    return out


def turn_orders(raw: str) -> list[tuple[str, str, str]]:
    """(kind, display, code) of each order a rendered turn creates."""
    return [(CREATE_ARGS[n][0], str(a.get(CREATE_ARGS[n][1], "") or ""), str(a.get(CREATE_ARGS[n][2], "") or ""))
            for n, a in parse_turn(raw) if n in CREATE_ARGS]


def order_key(o) -> tuple:
    return (o[0], re.sub(r"\W+", " ", o[1].lower()).strip() or o[2])


def render_order(o) -> str:
    return f"{'MedicationRequest' if o[0] == 'medication' else 'ServiceRequest'}: {o[1] or '?'}" + (
        f" (code {o[2]})" if o[2] else "")


def short_state(messages, budget: int = SCREEN_STATE_CHARS):
    """Instruction and steps with each tool result cut so the whole fits about `budget` characters."""
    instruction, steps = history_steps(messages)
    n = sum(len(s["results"]) for s in steps) or 1
    cap = max(300, (budget - len(instruction) - sum(len(s["action"]) for s in steps)) // n)
    return instruction, [{"action": s["action"][:2000],
                          "results": [r if len(r) <= cap else r[:cap] + " [...]" for r in s["results"]]} for s in steps]


PRIOR_DIR = Path("/ocean/projects/cis260032p/yliy/data/offline/pb_screen")
# R with a shared prefix: a ranker ending in a PREFIX_RUNS suffix replays that logged run of the same task turn by
# turn (its assistant messages, re-executed against a fresh environment, no LLM call) up to its FINISH, where the arm
# branches; every arm then starts from the same state and only the branch differs. The logged arm is llm_log_k5
# (Qwen) unless S1CLIN_PREFIX_ARM names another (R-Gemma: llm_log_k1__gemma4)
PREFIX_ARM = os.environ.get("S1CLIN_PREFIX_ARM", "llm_log_k5")
PREFIX_RUNS = {"-p1": "run1", "-p2": "run2", "-p3": "run3"}  # ranker suffix -> logged run of PREFIX_ARM
PREFIX_JOBS = Path(os.environ.get("PB_JOBS", "/ocean/projects/cis260032p/yliy/data/physicianbench/jobs"))


def logged_turns(job_dir: Path):
    """(assistant messages before the FINISH, the FINISH's final answer or None) of a logged llm_log run."""
    msgs = []
    for line in (job_dir / "logs/agent/decisions.messages.jsonl").read_text().splitlines():
        r = json.loads(line)
        msgs = msgs[:r["keep"]] if r["op"] == "reset" else msgs + [r["msg"]]
    dec = [json.loads(l) for l in (job_dir / "logs/agent/decisions.jsonl").read_text().splitlines()]
    last = dec[-1]["candidates"][dec[-1].get("pick", 0)] if dec else ""
    final = last[len("FINISH with final answer: "):] if turn_kind(last) == "finish" else None
    return [m for m in msgs if m["role"] == "assistant"], final


def _as_choice(content, tool_calls):
    tcs = [SimpleNamespace(id=tc["id"], type="function", function=SimpleNamespace(
        name=tc["function"]["name"], arguments=tc["function"]["arguments"])) for tc in tool_calls or []] or None
    return SimpleNamespace(message=SimpleNamespace(role="assistant", content=content, tool_calls=tcs, refusal=None),
                           finish_reason="tool_calls" if tcs else "stop")


class ScreenClient:
    """Arm screen (P7): the LLM acts as in llm_log (K samples logged, the first executed). At its first FINISH, the
    orders proposed by any candidate so far, plus those in B extra samples that must call a tool, are pooled; orders
    already placed are dropped; a ranker scores the rest and the top M are shown to the LLM as a checklist, and the
    LLM's answer to that is executed instead of the FINISH. The LLM keeps the decision. Rankers: a System One model
    (P(Yes) to SCREEN_QUESTION on the short state, pointwise) or `freq` (how many candidates proposed it; control).
    A ranker `<decider>-task` sees the task instruction only, `<decider>-blind` the order alone (P7').
    A ranker `prior-<decider>` (P9) skips the pooling: the pool is the task-prior list built before the run from the
    task text alone (scripts/pb_prior.py export), already ranked by the decider; `prior-random` shuffles it (control).
    Orders the run has placed are dropped and the top M shown. With B = 0 no extra samples are drawn.
    A suffix `-dedup` (O1: `prior-pplx12-dedup`, `prior-random-dedup`) also drops every pool item that DEDUP_DECIDER,
    given only the run's placed orders, judges a repeat of one of them (P10).
    Rankers `restart-fresh` / `restart-full` (R) show no list: see _restart. With a PREFIX_RUNS suffix (-p1, -p3)
    the run replays that logged run up to its FINISH and branches there; `restart-none-p3` keeps the logged FINISH
    (the replay check).
    """

    def __init__(self, base, ranker: str, k: int, b: int, m: int, log_path: Path, seed: str):
        self.base, self.ranker, self.k, self.b, self.m = base, ranker, k, b, m
        self.model_id = f"{base.model_id}+screen[{ranker}]@k{k}b{b}m{m}"
        self.log_path, self.seed, self.step, self.done = Path(log_path), seed, 0, False
        self.msglog = MessageLog(Path(log_path).with_suffix(".messages.jsonl"))
        self.proposed, self.placed = {}, set()  # order key -> [order, count]; keys of executed orders
        self.placed_orders = []  # executed orders, first of each key
        self.prefix = None
        if ranker[-3:] in PREFIX_RUNS:  # R with a shared prefix
            self.ranker = ranker[:-3]
            src, dst = PREFIX_JOBS / PREFIX_ARM / PREFIX_RUNS[ranker[-3:]] / seed.split("|")[0], Path(log_path).parents[2]
            turns, final = logged_turns(src)
            # the logged turns name the logged run's own workspace (absolute paths): point them at this run's
            fix = lambda x: x.replace(str(src), str(dst)) if isinstance(x, str) else x  # noqa: E731
            turns = [{**m, "content": fix(m.get("content")), "tool_calls": [
                {**tc, "function": {**tc["function"], "arguments": fix(tc["function"]["arguments"])}}
                for tc in m.get("tool_calls") or []]} for m in turns]
            if str(src) in json.dumps(turns) or str(src) in json.dumps(turns, ensure_ascii=False):
                raise RuntimeError("replayed turns still name the logged run's directory")
            self.prefix = (turns, fix(final))
        self.branch = None
        if self.ranker == "branch":  # F role 2: replay to the question's step, execute one option there, then continue
            spec = json.loads(Path(os.environ["S1CLIN_BRANCH_SPEC"]).read_text())
            b = spec[f"{seed.split('|')[0]}|{Path(log_path).parents[3].name}"]
            self.branch = {**b, "option": fix(b["option"])}  # the logged run's workspace paths point at this run's
            if str(src) in self.branch["option"]:
                raise RuntimeError("the branch option still names the logged run's directory")

    def _note(self, turns, executed):
        for t in turns:
            for o in turn_orders(t):
                self.proposed.setdefault(order_key(o), [o, 0])[1] += 1
        for o in turn_orders(executed):
            if order_key(o) not in self.placed:
                self.placed.add(order_key(o))
                self.placed_orders.append(o)

    def _prior(self):
        parts = self.ranker.split("-")
        name, dedup = parts[1], parts[-1] == "dedup"
        # the same pool for every ranker; `random` shuffles pplx12's (R3-d3: prior_ranked_d3.json)
        table = json.loads((PRIOR_DIR / f"prior_ranked_{'pplx12' if name == 'random' else name}.json").read_text())
        pool = [((k, d, c), p) for k, d, c, p in table.get(self.seed.split("|")[0], [])
                if order_key((k, d, c)) not in self.placed]
        self.prior_pool = len(pool)
        if dedup and pool:
            if DEDUP_DECIDER == "none":
                raise ValueError("a -dedup ranker needs a repeat judge (S1CLIN_REPEAT_JUDGE)")
            repeat = self._repeat([o for o, _ in pool])
            pool = [x for x, r in zip(pool, repeat) if r < DEDUP_MAX]
        self.prior_dedup = len(pool)
        if name == "random":
            random.Random(f"{self.seed}|prior").shuffle(pool)
        return [(p, o) for o, p in pool[:self.m]]

    def _repeat(self, orders):
        """P(repeats an order the run has placed), from the placed orders alone (no task text, no trajectory)."""
        from concurrent.futures import ThreadPoolExecutor
        placed = "\n".join(map(render_order, self.placed_orders)) or "(none)"

        def one(o):
            letters = ["A", "B"]
            random.Random(f"dedup|{self.seed}|{order_key(o)}").shuffle(letters)
            r = requests.post(DECIDER_URLS[DEDUP_DECIDER], timeout=600, json={
                "instruction": DEDUP_INSTRUCTION, "steps": [{"action": "Orders already placed", "results": [placed]}],
                "question": DEDUP_QUESTION.format(order=render_order(o)),
                "options": {letters[0]: DEDUP_YES, letters[1]: DEDUP_NO}})
            r.raise_for_status()
            return r.json()["distribution"][letters[0]]
        with ThreadPoolExecutor(8) as ex:
            return list(ex.map(one, orders))

    def _screen(self, messages, common, chosen, rec):
        """Arm screen at the first FINISH: rank, show the top M as a checklist, execute the LLM's answer."""
        t0, screen = time.time(), {"ranker": self.ranker}
        try:
            ranked = self._rank(messages, common, screen)
        except Exception as e:  # screening failure: the LLM's own FINISH stands
            ranked, screen["error"] = [], repr(e)[:500]
        try:
            if ranked:
                items = "\n".join(f"{i + 1}. {render_order(o)}" for i, (s, o) in enumerate(ranked))
                note = CHECKLIST.format(items=items)
                r2 = self.base.chat(messages + [{"role": "user", "content": note}], **common)
                chosen = r2.raw.choices[0]
                if chosen.message.tool_calls:  # keep the reason visible to later steps
                    chosen.message.content = ("[Pre-finish review of flagged orders]\n" + note + "\n\n" +
                                              (chosen.message.content or "")).strip()
                screen["answer"] = render_turn(chosen.message.content, chosen.message.tool_calls)[:MAX_CANDIDATE_CHARS]
                self._note([], screen["answer"])
        except Exception as e:  # screening failure: the LLM's own FINISH stands
            screen["error"] = repr(e)[:500]
        screen["s"] = round(time.time() - t0, 2)
        rec["screen"] = screen
        return chosen

    def _restart(self, messages, common, chosen, rec):
        """Arm R: ask for the remaining orders (RESTART_ASK) from the fresh or the full context; execute only its order
        calls that do not repeat a placed order by name; with none left, the LLM's own FINISH stands."""
        t0, screen = time.time(), {"ranker": self.ranker}
        if self.ranker == "restart-none":
            rec["screen"] = screen
            return chosen
        try:
            ctx = fresh_context(messages) if self.ranker == "restart-fresh" else list(messages)
            screen["context_messages"] = [len(messages), len(ctx)]
            r2 = self.base.chat(ctx + [{"role": "user", "content": RESTART_ASK}], **common).raw.choices[0]
            calls = r2.message.tool_calls or []
            screen["answer"] = render_turn(r2.message.content, calls)[:MAX_CANDIDATE_CHARS]
            keep, dropped = [], {"repeat": 0, "not_order": 0, "variant": 0}
            for tc in calls:
                os_ = turn_orders(render_turn(None, [tc]))
                if not os_:
                    dropped["not_order"] += 1
                elif order_key(os_[0]) in self.placed:
                    dropped["repeat"] += 1
                else:
                    keep.append((tc, os_[0]))
            if keep and self.placed_orders and DEDUP_DECIDER != "none":  # variants of placed orders: the P10 repeat judge
                p = self._repeat([o for _, o in keep])
                screen["p_repeat"] = [round(x, 4) for x in p]
                dropped["variant"] = sum(x >= DEDUP_MAX for x in p)
                keep = [k for k, x in zip(keep, p) if x < DEDUP_MAX]
            keep = [tc for tc, _ in keep]
            screen.update(dropped=dropped, executed=len(keep))
            if keep:
                r2.message.tool_calls = keep
                r2.message.content = ("[Pre-finish review]\n" + RESTART_ASK + "\n\n" + (r2.message.content or "")).strip()
                self._note([], render_turn(r2.message.content, keep))
                chosen = r2
        except Exception as e:  # restart failure: the LLM's own FINISH stands
            screen["error"] = repr(e)[:500]
        screen["s"] = round(time.time() - t0, 2)
        rec["screen"] = screen
        return chosen

    def _rank(self, messages, common, screen):
        """The top M (score, order) pairs of the pool; the pool and flagged orders go into `screen`."""
        if self.ranker.startswith("prior-"):
            ranked = self._prior()
            screen.update(pool=self.prior_pool, pool_after_dedup=self.prior_dedup)
        else:
            extra = self.base.chat(messages, n=self.b, tool_choice="required", **common).raw.choices
            ext = [render_turn(c.message.content, c.message.tool_calls)[:MAX_CANDIDATE_CHARS] for c in extra]
            self._note(ext, "")
            pool = [o for key, (o, _) in self.proposed.items() if key not in self.placed]
            scores = self._score(messages, pool) if pool else []
            ranked = sorted(zip(scores, pool), key=lambda x: -x[0])[:self.m]
            screen.update(pool=len(pool), extra_orders=sum(len(turn_orders(t)) for t in ext))
        screen["flagged"] = [{"order": render_order(o), "score": round(s, 4)} for s, o in ranked]
        return ranked

    def _score(self, messages, orders):
        if self.ranker == "freq":
            return [float(self.proposed[order_key(o)][1]) for o in orders]
        from concurrent.futures import ThreadPoolExecutor
        decider, _, view = self.ranker.partition("-")
        instruction, steps = short_state(messages)
        if view == "task":
            steps = []
        elif view == "blind":
            instruction, steps = SCREEN_BLIND, []
        url = DECIDER_URLS[decider]

        def one(o):
            letters = ["A", "B"]
            random.Random(f"{self.seed}|{order_key(o)}").shuffle(letters)
            r = requests.post(url, timeout=600, json={
                "instruction": instruction, "steps": steps, "question": SCREEN_QUESTION.format(order=render_order(o)),
                "options": {letters[0]: SCREEN_YES, letters[1]: SCREEN_NO}})
            r.raise_for_status()
            return r.json()["distribution"][letters[0]]
        with ThreadPoolExecutor(4) as ex:
            return list(ex.map(one, orders))

    def _branched(self, n_messages, epoch):
        """F role 2: the logged turn before the branch step, the given option at it (rebuilt from its rendering, so
        every option of a question, the agent's own included, is executed the same way); then the agent goes on."""
        from agent.llm_client import ChatResponse
        rec = {"step": self.step, "prefix": True, "n_messages": n_messages, "msg_epoch": epoch, "pick": 0}
        if self.step < self.branch["step"]:
            m = self.prefix[0][self.step - 1]
            chosen = _as_choice(m.get("content"), m.get("tool_calls"))
        else:
            opt, self.done = self.branch["option"], True
            rec["branch"] = True
            if opt.startswith("FINISH with final answer:"):
                chosen = _as_choice(opt[len("FINISH with final answer:"):].strip(), None)
            else:
                calls = parse_turn(opt)
                if not calls:
                    raise RuntimeError("branch option does not parse into tool calls")
                chosen = _as_choice(None, [{"id": f"call_branch_{i}", "type": "function", "function": {
                    "name": n, "arguments": json.dumps(a, ensure_ascii=False)}} for i, (n, a) in enumerate(calls)])
        rendered = render_turn(chosen.message.content, chosen.message.tool_calls)[:MAX_CANDIDATE_CHARS]
        rec["candidates"] = [rendered]
        self._note([rendered], rendered)
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return ChatResponse(content=chosen.message.content, tool_calls=chosen.message.tool_calls, prompt_tokens=0,
                            completion_tokens=0, raw=SimpleNamespace(choices=[chosen], usage=None))

    def _replayed(self, messages, common, n_messages, epoch):
        """The logged run's turn for this step (prefix), or at its FINISH the arm's branch."""
        from agent.llm_client import ChatResponse
        turns, final = self.prefix
        rec = {"step": self.step, "prefix": True, "n_messages": n_messages, "msg_epoch": epoch, "pick": 0}
        if self.step <= len(turns):
            m = turns[self.step - 1]
            chosen = _as_choice(m.get("content"), m.get("tool_calls"))
        else:
            chosen = _as_choice(final or "", None)
            self.done = True
            if self.ranker.startswith("restart-"):
                chosen = self._restart(messages, common, chosen, rec)
            else:  # a checklist arm (prior-*) from the shared prefix
                chosen = self._screen(messages, common, chosen, rec)
        rendered = render_turn(chosen.message.content, chosen.message.tool_calls)[:MAX_CANDIDATE_CHARS]
        rec["candidates"] = [rendered]
        self._note([rendered], rendered)
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return ChatResponse(content=chosen.message.content, tool_calls=chosen.message.tool_calls, prompt_tokens=0,
                            completion_tokens=0, raw=SimpleNamespace(choices=[chosen], usage=None))

    def chat(self, messages, tools=None, temperature=None, parallel_tool_calls=True, reasoning_effort=None, **kw):
        self.step += 1
        n_messages, epoch = self.msglog.sync(messages)
        common = dict(tools=tools, temperature=temperature, parallel_tool_calls=parallel_tool_calls,
                      reasoning_effort=reasoning_effort)
        if self.branch is not None and not self.done:
            return self._branched(n_messages, epoch)
        if self.prefix is not None and not self.done and (self.prefix[1] is not None or self.step <= len(self.prefix[0])):
            return self._replayed(messages, common, n_messages, epoch)
        resp = self.base.chat(messages, n=self.k, **common)
        choices = resp.raw.choices
        renderings = [render_turn(c.message.content, c.message.tool_calls)[:MAX_CANDIDATE_CHARS] for c in choices]
        chosen, rec = choices[0], {"step": self.step, "n_samples": len(choices), "candidates": renderings,
                                   "n_messages": n_messages, "msg_epoch": epoch, "pick": 0, "q": QUESTION_VERSION,
                                   "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens}
        self._note(renderings, renderings[0])
        if not chosen.message.tool_calls and not self.done and self.ranker.startswith("restart-"):
            self.done = True
            chosen = self._restart(messages, common, chosen, rec)
        elif not chosen.message.tool_calls and not self.done:
            self.done = True
            chosen = self._screen(messages, common, chosen, rec)
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        from agent.llm_client import ChatResponse
        return ChatResponse(content=chosen.message.content, tool_calls=chosen.message.tool_calls,
                            prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
                            raw=SimpleNamespace(choices=[chosen], usage=resp.raw.usage))
