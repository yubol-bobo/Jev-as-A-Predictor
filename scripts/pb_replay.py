"""Replay the agent's own context at two points of its logged runs and sample its next turn under different inputs
(research loop A1-A4: why required orders are missed; aggregates only in results/). With S1CLIN_AGENT=gemma4 (X, XA): the runs, samples and
results of the Gemma 4 agent (llm_log_k1__gemma4, sampled by Gemma; results/pb_replay_gemma4.json).

Cases: each task of llm_log_k5/run1 and run3 with a required-order spec. Two points per case, rebuilt from
decisions.messages.jsonl exactly as the LLM saw them:
  finish - the first step whose executed turn is a FINISH (the run is about to end)
  order  - the first step whose executed turn places an order (the order burst begins)
Conditions (the agent's system prompt, tools and sampling: Qwen thinking on, default temperature, 16384 tokens):
  finish_none    the context as it was (resample the FINISH step)
  finish_nudge   + user: NUDGE (check that every order the note or plan recommends was placed)            A1
  finish_ask     + user: ASK (place every order the task requires that is not yet placed)                  A2 control
  finish_fresh   system, task, and only the retrieval turns (every turn with an order, message, appointment
                 or note removed, with its results) + user: ASK                                            A2
  order_none     the context as it was (resample the first order step)
  order_remind   + user: the task instruction repeated (REMIND)                                            A3
  order_plan     + user: PLAN (write the full assessment and plan first, then place every order in it)     A4
Per sample: the orders it creates (pb_decider.turn_orders). Scored against the required-order specs: at `finish`,
the specs the run had not met before that step (open) and duplicates of orders it had placed; at `order`, all
required specs. Raw samples stay under data/offline/pb_replay/.

  .venv/bin/python scripts/pb_replay.py build
  .venv/bin/python scripts/pb_replay.py run --cond finish_fresh [--k 5] [--url URL*w,...] [--workers 24]
  .venv/bin/python scripts/pb_replay.py score [--out results/pb_replay.json]
  .venv/bin/python scripts/pb_replay.py filter    (exploratory, for R: the finish samples after the online filter)
The `filter` command applies R's filter (pb_decider: drop exact repeats of placed orders, then items the pplx12
repeat judge, given only the placed orders, scores P(repeat) >= 0.5) to the finish_ask / finish_fresh samples and
reports coverage, repeats and other orders after it (results/pb_replay_filter.json).
"""

import argparse
import json
import os
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO), str(REPO / "third_party/physicianbench")]
from offline_decisions import replay  # noqa: E402
from pb_offline_orders import matches, required_orders  # noqa: E402
from pb_summarize import JOBS  # noqa: E402
from src.s1clin.pb_decider import order_key, render_turn, turn_kind, turn_orders  # noqa: E402

AGENT = os.environ.get("S1CLIN_AGENT", "qwen")  # the agent whose runs are replayed and who samples (X: gemma4)
MODEL = {"qwen": "qwen3.8-27b", "gemma4": "gemma-4-31b-it"}[AGENT]
OD = Path("/ocean/projects/cis260032p/yliy/data/offline/pb_replay") / ("" if AGENT == "qwen" else AGENT)
RUNS = {"qwen": ("llm_log_k5/run1", "llm_log_k5/run3"), "gemma4": ("llm_log_k1__gemma4/run1",)}[AGENT]
R_RUNS = {"qwen": RUNS, "gemma4": ("llm_log_k1__gemma4/run1", "llm_log_k1__gemma4/run2")}[AGENT]  # R / XR cases
SUFFIX = "" if AGENT == "qwen" else f"_{AGENT}"  # of the results files
NUDGE = ("Before you finish, check that every order your note or plan recommends has actually been placed. Place any "
         "that have not been placed yet, then finish.")
ASK = ("Based on the task and the patient data gathered so far, place every order that this task requires and that "
       "has not been placed yet, then finish.")
REMIND = "Reminder of your task:\n\n{instruction}"
PLAN = ("Before placing any orders, write out your full assessment and plan in your reply, listing every order the "
        "task requires. Then place every order in that plan.")
CONDS = ("finish_none", "finish_nudge", "finish_ask", "finish_fresh", "order_none", "order_remind", "order_plan")
DECISION_TOOLS = ("_create", "write_file")


def cases(runs=RUNS):
    """[{run, task, finish: (epoch, n), order: (epoch, n), open: [spec idx], placed: [orders before finish]}]."""
    out = []
    for slug in runs:
        for jd in sorted(p for p in (JOBS / slug).iterdir() if p.is_dir()):
            dec_p = jd / "logs/agent/decisions.jsonl"
            specs = [s for s in required_orders(jd.name) if not s["negative"]]
            if not specs or not (jd / "metadata.json").exists() or not dec_p.exists():
                continue
            dec = [json.loads(l) for l in dec_p.read_text().splitlines()]
            chosen = [r["candidates"][r.get("pick", 0)] for r in dec]
            fi = next((i for i, c in enumerate(chosen) if turn_kind(c) == "finish"), None)
            oi = next((i for i, c in enumerate(chosen) if turn_orders(c)), None)
            placed = [o for c in chosen[:fi if fi is not None else len(chosen)] for o in turn_orders(c)]
            met = {i for i, s in enumerate(specs) if any(matches(s, *o) for o in placed)}
            loc = lambda i: (dec[i].get("msg_epoch", 0), dec[i]["n_messages"]) if i is not None else None  # noqa: E731
            out.append({"run": slug, "task": jd.name, "finish": loc(fi), "order": loc(oi),
                        "open": [i for i in range(len(specs)) if i not in met], "placed": placed, "n_specs": len(specs)})
    return out


def is_decision(m):
    return m["role"] == "assistant" and any(tc["function"]["name"].endswith(DECISION_TOOLS)
                                            for tc in m.get("tool_calls") or [])


def messages(case, cond):
    point = case["finish"] if cond.startswith("finish") else case["order"]
    if point is None:
        return None
    epochs = replay(JOBS / case["run"] / case["task"] / "logs/agent/decisions.messages.jsonl")
    msgs = epochs[point[0]][:point[1]]
    if cond == "finish_fresh":  # drop every decision turn and the tool results that answer it
        out, drop = [], set()
        for m in msgs:
            if is_decision(m):
                drop |= {tc["id"] for tc in m["tool_calls"]}
            elif m["role"] == "tool" and m.get("tool_call_id") in drop:
                continue
            else:
                out.append(m)
        msgs = out
    extra = {"finish_nudge": NUDGE, "finish_ask": ASK, "finish_fresh": ASK, "order_plan": PLAN,
             "order_remind": REMIND.format(instruction=next(m["content"] for m in msgs if m["role"] == "user"))}.get(cond)
    return msgs + ([{"role": "user", "content": extra}] if extra else [])


def tools():
    from agent.tool_registry import ToolRegistry, register_all_tools
    reg = ToolRegistry()
    register_all_tools(reg)
    return reg.to_openai_tools()


def route(spec, key):
    """The replica for `key`: URL*weight,... -> one URL, stable per key."""
    urls = []
    for part in spec.split(","):
        u, _, w = part.partition("*")
        urls += [u] * int(w or 1)
    return urls[random.Random(key).randrange(len(urls))]


def run(cond, k, url, workers):
    import openai
    path, lock = OD / f"samples_{cond}.jsonl", threading.Lock()
    done = {(x["run"], x["task"]) for x in map(json.loads, path.read_text().splitlines())
            if "error" not in x} if path.exists() else set()  # failed requests are retried
    todo = [c for c in cases() if (c["run"], c["task"]) not in done
            and (c["open"] or not cond.startswith("finish"))]  # at finish, only cases with an open spec are scored
    tl = tools()
    print(f"{path.name}: {len(todo)} cases", flush=True)

    def one(c):
        msgs = messages(c, cond)
        rec = {"run": c["run"], "task": c["task"]}
        if msgs is None:
            rec["skip"] = "no such point"
        else:
            t0 = time.time()
            try:
                cl = openai.OpenAI(base_url=route(url, c["task"]), api_key="none", timeout=3600)
                r = cl.chat.completions.create(model=MODEL, messages=msgs, tools=tl, parallel_tool_calls=True, n=k,
                                               max_completion_tokens=16384,
                                               extra_body={"chat_template_kwargs": {"enable_thinking": True}})
                rec["samples"] = [render_turn(ch.message.content, ch.message.tool_calls) for ch in r.choices]
                rec["s"] = round(time.time() - t0, 1)
            except Exception as e:
                rec["error"] = repr(e)[:300]
        with lock, open(path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))


def boot(d, n=2000):
    rng = random.Random(0)
    if not d:
        return None
    b = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return [round(b[50], 4), round(b[1949], 4)]


def score(out):
    cs = {(c["run"], c["task"]): c for c in cases()}
    res, per = {}, {}
    for cond in CONDS:
        p = OD / f"samples_{cond}.jsonl"
        if not p.exists():
            continue
        rows = {}
        for x in map(json.loads, p.read_text().splitlines()):
            c = cs.get((x["run"], x["task"]))
            if not c or "samples" not in x:
                continue
            specs = [s for s in required_orders(c["task"]) if not s["negative"]]
            target = c["open"] if cond.startswith("finish") else list(range(len(specs)))
            if not target:
                continue
            placed_keys = {order_key(o) for o in c["placed"]}
            cov, dup, new, fin = [], [], [], []
            for smp in x["samples"]:
                os_ = turn_orders(smp)
                cov.append(len({i for i in target for o in os_ if matches(specs[i], *o)}) / len(target))
                dup.append(sum(order_key(o) in placed_keys or any(matches(s, *o) for j, s in enumerate(specs)
                                                                  if j not in c["open"]) for o in os_))
                new.append(len(os_))
                fin.append(turn_kind(smp) == "finish")
            rows[(x["run"], x["task"])] = {"cov": sum(cov) / len(cov), "any": float(any(v > 0 for v in cov)),
                                           "dup": sum(dup) / len(dup), "orders": sum(new) / len(new),
                                           "finish": sum(fin) / len(fin), "s": x.get("s")}
        per[cond] = rows
        m = lambda k: round(sum(r[k] for r in rows.values()) / len(rows), 4) if rows else None  # noqa: E731
        res[cond] = {"cases": len(rows), "coverage": m("cov"), "coverage_any_of_k": m("any"), "duplicates": m("dup"),
                     "orders": m("orders"), "finish_rate": m("finish")}
    for a, b in (("finish_nudge", "finish_none"), ("finish_fresh", "finish_ask"), ("finish_ask", "finish_none"),
                 ("order_remind", "order_none"), ("order_plan", "order_none")):
        if a in per and b in per:
            ks = [k for k in per[a] if k in per[b]]
            d = [per[a][k]["cov"] - per[b][k]["cov"] for k in ks]
            dd = [per[a][k]["dup"] - per[b][k]["dup"] for k in ks]
            res[f"{a}_minus_{b}"] = {"cases": len(ks), "coverage": round(sum(d) / len(d), 4) if d else None,
                                     "ci95": boot(d), "duplicates": round(sum(dd) / len(dd), 4) if dd else None}
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def filt(out):
    import requests
    from src.s1clin.pb_decider import (DECIDER_URLS, DEDUP_DECIDER, DEDUP_INSTRUCTION, DEDUP_MAX, DEDUP_NO,
                                       DEDUP_QUESTION, DEDUP_YES, render_order)
    cs = {(c["run"], c["task"]): c for c in cases()}

    def p_repeat(placed, o, key):
        letters = ["A", "B"]
        random.Random(f"dedup|{key}|{order_key(o)}").shuffle(letters)
        r = requests.post(DECIDER_URLS[DEDUP_DECIDER], timeout=600, json={
            "instruction": DEDUP_INSTRUCTION, "steps": [{"action": "Orders already placed", "results": [placed]}],
            "question": DEDUP_QUESTION.format(order=render_order(o)), "options": {letters[0]: DEDUP_YES, letters[1]: DEDUP_NO}})
        r.raise_for_status()
        return r.json()["distribution"][letters[0]]

    res, per = {}, {}
    for cond in ("finish_ask", "finish_fresh"):
        rows = {}
        for x in map(json.loads, (OD / f"samples_{cond}.jsonl").read_text().splitlines()):
            c = cs.get((x["run"], x["task"]))
            if not c or "samples" not in x or not c["open"]:
                continue
            specs = [s for s in required_orders(c["task"]) if not s["negative"]]
            pk, placed = {order_key(o) for o in c["placed"]}, "\n".join(map(render_order, c["placed"])) or "(none)"
            cov, rep, var, oth, kept = [], [], [], [], []
            for smp in x["samples"]:
                os_ = [o for o in turn_orders(smp) if order_key(o) not in pk]
                if os_ and c["placed"]:
                    with ThreadPoolExecutor(8) as ex:
                        p = list(ex.map(lambda o: p_repeat(placed, o, f"{x['run']}|{x['task']}"), os_))
                    os_ = [o for o, q in zip(os_, p) if q < DEDUP_MAX]
                met = [s for j, s in enumerate(specs) if j not in c["open"]]
                cov.append(len({i for i in c["open"] for o in os_ if matches(specs[i], *o)}) / len(c["open"]))
                var.append(sum(any(matches(s, *o) for s in met) for o in os_))
                oth.append(sum(not any(matches(s, *o) for s in specs) for o in os_))
                kept.append(len(os_))
            rows[(x["run"], x["task"])] = {k: sum(v) / len(v) for k, v in
                                           (("cov", cov), ("variants_left", var), ("other", oth), ("orders", kept))}
        per[cond] = rows
        res[cond] = {k: round(sum(r[k] for r in rows.values()) / len(rows), 4) for k in ("cov", "variants_left", "other", "orders")}
        res[cond]["cases"] = len(rows)
    ks = [k for k in per["finish_fresh"] if k in per["finish_ask"]]
    d = [per["finish_fresh"][k]["cov"] - per["finish_ask"][k]["cov"] for k in ks]
    res["fresh_minus_ask"] = {"cases": len(ks), "coverage": round(sum(d) / len(d), 4), "ci95": boot(d)}
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "score", "filter"])
    ap.add_argument("--cond", choices=CONDS)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--url", default="http://127.0.0.1:8011/v1*2,http://127.0.0.1:8013/v1*1")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=str(REPO / f"results/pb_replay{SUFFIX}.json"))
    a = ap.parse_args()
    os.umask(0o077)
    OD.mkdir(parents=True, exist_ok=True, mode=0o700)
    if a.cmd == "build":
        cs = cases()
        sizes = defaultdict(list)
        for c in cs:
            for cond in ("finish_none", "finish_fresh", "order_none"):
                m = messages(c, cond)
                if m is not None:
                    sizes[cond].append(sum(len(json.dumps(x)) for x in m))
        print(json.dumps({"cases": len(cs), "with_finish": sum(c["finish"] is not None for c in cs),
                          "with_order": sum(c["order"] is not None for c in cs),
                          "with_open_at_finish": sum(bool(c["open"]) and c["finish"] is not None for c in cs),
                          "open_specs_at_finish": sum(len(c["open"]) for c in cs if c["finish"] is not None),
                          "median_chars": {k: sorted(v)[len(v) // 2] for k, v in sizes.items()}}, indent=1))
    elif a.cmd == "run":
        run(a.cond, a.k, a.url, a.workers)
    elif a.cmd == "filter":
        filt(REPO / "results/pb_replay_filter.json")
    else:
        score(a.out)


if __name__ == "__main__":
    main()
