"""F, role 2 (docs/research_loop.md): the selector. On 60 exam questions (the agent's K = 5 sampled next turns at one
step of llm_log_k5/run1), every option is executed from a replay of the logged run and the agent continues to its
FINISH (2 rollouts, 1 for a FINISH option); an option's value is the share of the task's checkpoints passed. Judges
pick one option from the exam request: the six System One models (their exam answers, unchanged) and the same-scale
LLM as A / AG (first-token log-probabilities over the letters), C2 / C8 (thinking, then 'ANSWER=<letter>') and D
(5 sampled letters, majority); baselines: the option the agent executed, the most sampled one, random.

  .venv/bin/python scripts/pb_f2.py pick                       # the 60 questions, the branch spec, per-run task lists
  .venv/bin/python scripts/pb_f2.py judge --cfg A [--url ...]  # LLM selector answers
  .venv/bin/python scripts/pb_f2.py timing --judge pplx12|A|AG|C2|C8|D [--url ...]
  .venv/bin/python scripts/pb_f2.py report
Rollouts: pb_run --arm screen --ranker branch-p1 --k 1 --run N --tasks @<FD>/branch_tasks_runN.txt with
S1CLIN_BRANCH_SPEC=<FD>/branch_spec.json (scripts/f2_run.sh). Patient content stays under data/offline/pb_f/.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from clef_render import fit_request  # noqa: E402
from pb_exam import JOBS, OD as EXAM_OD, SLUG  # noqa: E402
from pb_offline_orders import orders_in, replay  # noqa: E402
from src.s1clin.pb_decider import (DECIDER_URLS, DECISION_QUESTION, _letters, history_steps, parse_turn,  # noqa: E402
                                   render_turn)

FD = Path("/ocean/projects/cis260032p/yliy/data/offline/pb_f")
S1 = ("pplx12", "d3", "clef", "linne", "amos", "decider31b")
MODEL = {"qwen": "qwen3.8-27b", "gemma": "gemma-4-31b-it"}
N_QUESTIONS, PER_TASK, RUN0 = 60, 2, 101
QWEN_TOK = ("/ocean/projects/cis260032p/yliy/huggingface_cache/hub/models--Qwen--Qwen3.8-27B/snapshots/"
            "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer.json")
MAX_CONTEXT, TEMPLATE_ALLOWANCE = 32_768, 512  # as scripts/s1_proxy.py
BUDGET = {"C2": 2048, "C8": 8192}
CP = re.compile(r"^(PASSED|FAILED) \S*test_checkpoint_(\S+)", re.M)


def rebuilds(opt):
    """The option executes as logged: a final answer, or tool calls whose rendering round-trips exactly."""
    if opt.startswith("FINISH with final answer:"):
        return True
    calls = parse_turn(opt)
    tcs = [{"function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)}} for n, a in calls]
    return bool(calls) and render_turn(None, tcs) == opt


def eligible(q):
    if not all(rebuilds(o) for o in q["options"]):
        return False
    if len(set(q["kinds"])) > 1:
        return True
    sets = [frozenset(map(tuple, orders_in(parse_turn(o)))) for o in q["options"]]
    return set(q["kinds"]) == {"act"} and len(set(sets)) > 1


def pick():
    qs = [json.loads(l) for l in (EXAM_OD / "questions.jsonl").read_text().splitlines()]
    qs = sorted((q for q in qs if eligible(q)), key=lambda q: hashlib.sha256(q["id"].encode()).hexdigest())
    chosen, per = [], Counter()
    for q in qs:
        if per[q["task"]] < PER_TASK:
            chosen.append(q)
            per[q["task"]] += 1
        if len(chosen) == N_QUESTIONS:
            break
    spec, runs = {}, {}
    for t in sorted(per):
        n = RUN0
        for q in (q for q in chosen if q["task"] == t):
            for j, opt in enumerate(q["options"]):
                for k in range(1 if opt.startswith("FINISH") else 2):
                    spec[f"{t}|run{n}"] = {"id": q["id"], "step": q["step"], "option_index": j, "rollout": k, "option": opt}
                    runs.setdefault(n, []).append(t)
                    n += 1
    FD.mkdir(parents=True, exist_ok=True)
    with open(FD / "sel_questions.jsonl", "w") as f:
        for q in chosen:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    (FD / "branch_spec.json").write_text(json.dumps(spec, ensure_ascii=False))
    for n, ts in runs.items():
        (FD / f"branch_tasks_run{n}.txt").write_text("\n".join(ts) + "\n")
    print(f"{len(qs)} eligible questions; {len(chosen)} chosen in {len(per)} tasks; {len(spec)} rollouts in runs "
          f"{min(runs)}-{max(runs)}; options per question {sum(len(q['options']) for q in chosen) / len(chosen):.2f}")


_TOK, _STATE = [], {}


def request(q):
    """(prompt text without the ending, letters) from the exam request, fitted as the System One proxy fits it."""
    if not _TOK:
        _TOK.append(Tokenizer.from_file(QWEN_TOK))

    def count(req):
        x = req["questions"]["next"]
        text = req["state"] + "\n" + x["instructions"] + "\n" + "\n".join(f"{k}: {v}" for k, v in x["criteria"].items())
        return len(_TOK[0].encode(text, add_special_tokens=False).ids) + TEMPLATE_ALLOWANCE

    if q["task"] not in _STATE:
        _STATE[q["task"]] = replay(JOBS / SLUG / q["task"] / "logs/agent/decisions.messages.jsonl")
    instruction, steps = history_steps(_STATE[q["task"]][q["msg_epoch"]][:q["n_messages"]])
    letters = _letters(len(q["options"]), hashlib.sha256(q["id"].encode()).hexdigest())  # the exam's letters
    opts = {letters[i]: o for i, o in enumerate(q["options"])}
    req, _, _ = fit_request(instruction, steps, DECISION_QUESTION, opts, count, MAX_CONTEXT)
    text = req["state"] + "\n\n---\n" + DECISION_QUESTION + "\n\n" + "\n\n".join(f"{k}. {opts[k]}" for k in sorted(opts))
    return text, letters


def judge_one(cfg, url, q):
    """(index of the picked option, info)."""
    text, letters = request(q)
    model = MODEL["gemma" if cfg == "AG" else "qwen"]
    if cfg in ("A", "AG", "D"):
        body = {"model": model, "max_tokens": 1, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": text + "\n\nAnswer with the letter of one option only."}]}
        r = requests.post(f"{url}/chat/completions", timeout=3600, json=body | (
            {"n": 5} if cfg == "D" else {"temperature": 0, "logprobs": True, "top_logprobs": 20}))
        r.raise_for_status()
        m = r.json()
        if cfg == "D":
            votes = [c["message"]["content"].strip()[:1] for c in m["choices"]]
            c = Counter(v for v in votes if v in letters)
            best = max(c.values(), default=0)
            top = [v for v in votes if c.get(v) == best]
            return (letters.index(top[0]) if top else q["first"]), {"tokens": m["usage"]["completion_tokens"]}
        lp = {k: max((x["logprob"] for x in m["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                      if x["token"].strip() == k), default=-50.0) for k in letters}
        return letters.index(max(lp, key=lp.get)), {"tokens": 1, "dist": {k: round(math.exp(v), 4) for k, v in lp.items()}}
    r = requests.post(f"{url}/chat/completions", timeout=3600, json={
        "model": model, "max_tokens": BUDGET[cfg], "messages": [{"role": "user", "content": text + (
            "\n\nThink it through, then end your reply with one line 'ANSWER=<letter>'.")}]})
    r.raise_for_status()
    m = r.json()
    v = re.findall(r"ANSWER\s*=\s*([A-Z])", m["choices"][0]["message"]["content"] or "")
    ok = bool(v) and v[-1] in letters
    return (letters.index(v[-1]) if ok else q["first"]), {"tokens": m["usage"]["completion_tokens"],
                                                         **({} if ok else {"no_answer": True})}


def questions():
    return [json.loads(l) for l in (FD / "sel_questions.jsonl").read_text().splitlines()]


def judge(cfg, url, workers):
    out, lock = FD / f"sel_{cfg}.jsonl", threading.Lock()
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [q for q in questions() if q["id"] not in done]
    print(f"{out.name}: {len(todo)} to answer", flush=True)

    def run(q):
        t0 = time.time()
        try:
            p, info = judge_one(cfg, url, q)
        except Exception as e:
            p, info = None, {"error": repr(e)[:300]}
        with lock, open(out, "a") as f:
            f.write(json.dumps({"id": q["id"], "pick": p, "s": round(time.time() - t0, 3)} | info) + "\n")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, todo))


def timing(name, url):
    qs = questions()[:50]

    def one(q):
        if name in S1:
            instruction, steps = history_steps(_STATE.setdefault(q["task"], replay(
                JOBS / SLUG / q["task"] / "logs/agent/decisions.messages.jsonl"))[q["msg_epoch"]][:q["n_messages"]])
            letters = _letters(len(q["options"]), hashlib.sha256(q["id"].encode()).hexdigest())
            r = requests.post(DECIDER_URLS[name], timeout=900, json={
                "instruction": instruction, "steps": steps, "question": DECISION_QUESTION,
                "options": {letters[i]: o for i, o in enumerate(q["options"])}})
            r.raise_for_status()
            return 0
        return judge_one(name, url, q)[1].get("tokens", 0)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, qs[:5]))  # warm-up
        t0 = time.time()
        toks = list(ex.map(one, qs[5:]))
        wall = time.time() - t0
    rec = {"judge": name, "items": len(toks), "wall_s": round(wall, 2), "gpu_s_per_judgment": round(wall / len(toks), 4),
           "tokens_per_judgment": round(sum(toks) / len(toks), 1), "concurrency": 8}
    with open(FD / "sel_timing.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(rec)


def boot(per, n=2000):
    ts, rng = sorted(per), random.Random(1)
    if not ts:
        return None
    b = sorted(sum(per[t] for t in s) / len(s) for s in ([rng.choice(ts) for _ in ts] for _ in range(n)))
    return [round(b[int(0.025 * n)], 4), round(b[int(0.975 * n) - 1], 4)]


def report(path):
    jobs = JOBS / "screen_branch-p1_b0m8_k1"
    spec = json.loads((FD / "branch_spec.json").read_text())
    vals = {}  # (id, option index) -> [checkpoint pass share]
    for key, b in spec.items():
        task, run = key.split("|")
        p = jobs / run / task / "logs/verifier/pytest_output.txt"
        if p.exists():
            cps = CP.findall(p.read_text())
            if cps:
                vals.setdefault((b["id"], b["option_index"]), []).append(sum(s == "PASSED" for s, _ in cps) / len(cps))
    qs = questions()
    picks = {}
    for name in S1:
        a = {x["id"]: x["pick"] for x in map(json.loads, (EXAM_OD / f"answers_{name}.jsonl").read_text().splitlines())
             if x["pick"] is not None}
        picks[name] = a
    for cfg in ("A", "AG", "C2", "C8", "D"):
        f = FD / f"sel_{cfg}.jsonl"
        if f.exists():
            picks[cfg] = {x["id"]: x["pick"] for x in map(json.loads, f.read_text().splitlines()) if x["pick"] is not None}
    picks["own"] = {q["id"]: q["first"] for q in qs}
    picks["most_sampled"] = {q["id"]: max(range(len(q["counts"])), key=lambda i: (q["counts"][i], -i)) for q in qs}
    gain, regret, info_q = {j: {} for j in picks}, {j: {} for j in picks}, 0
    for q in qs:
        v = [sum(vals[(q["id"], j)]) / len(vals[(q["id"], j)]) if vals.get((q["id"], j)) else None
             for j in range(len(q["options"]))]
        if None in v or len(set(v)) == 1:
            continue
        info_q += 1
        mean_v = sum(v) / len(v)
        for j, a in picks.items():
            if q["id"] in a:
                gain[j].setdefault(q["task"], []).append(v[a[q["id"]]] - mean_v)
                regret[j].setdefault(q["task"], []).append(max(v) - v[a[q["id"]]])
    per = {j: {t: sum(x) / len(x) for t, x in d.items()} for j, d in gain.items()}
    per["S1mean"] = {t: sum(per[s][t] for s in S1) / len(S1) for t in set.intersection(*(set(per[s]) for s in S1))}
    per["S1gemma"] = {t: (per["linne"][t] + per["decider31b"][t]) / 2 for t in set(per["linne"]) & set(per["decider31b"])}
    res = {"questions": len(qs), "informative": info_q, "rollouts_graded": sum(map(len, vals.values())),
           "rollouts_planned": len(spec), "judges": {}, "diff": {}}
    for j, d in per.items():
        res["judges"][j] = {"gain": round(sum(d.values()) / len(d), 4) if d else None, "ci95": boot(d), "tasks": len(d)}
        if j in regret and regret[j]:
            r = [x for v in regret[j].values() for x in v]
            res["judges"][j]["regret"] = round(sum(r) / len(r), 4)
    for x, y in (("S1mean", "A"), ("S1mean", "AG"), ("S1mean", "C8"), ("S1mean", "C2"), ("S1mean", "D"),
                 ("S1mean", "own"), ("S1mean", "most_sampled"), ("A", "own"), ("S1gemma", "AG")) + tuple((s, "A") for s in S1):
        if x in per and y in per:
            d = {t: per[x][t] - per[y][t] for t in set(per[x]) & set(per[y])}
            if d:
                res["diff"][f"{x}_minus_{y}"] = {"tasks": len(d), "mean": round(sum(d.values()) / len(d), 4), "ci95": boot(d)}
    tf = FD / "sel_timing.jsonl"
    if tf.exists():
        res["timing"] = {x["judge"]: x for x in map(json.loads, tf.read_text().splitlines())}
        # as F3: on the frontier unless another judge at <= GPU-seconds has higher gain, paired CI excluding 0
        cost = {j: v["gpu_s_per_judgment"] for j, v in res["timing"].items() if j in per}
        res["frontier"] = {}
        for j in sorted(cost, key=cost.get):
            by = []
            for k in cost:
                ks = set(per[k]) & set(per[j])
                if k == j or cost[k] > cost[j] or not ks:
                    continue
                if boot({t: per[k][t] - per[j][t] for t in ks})[0] > 0:
                    by.append(k)
            res["frontier"][j] = {"gpu_s": cost[j], "gain": res["judges"][j]["gain"], "dominated_by": by}
    Path(path).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pick", "judge", "timing", "report"])
    ap.add_argument("--cfg", choices=["A", "AG", "C2", "C8", "D"])
    ap.add_argument("--judge")
    ap.add_argument("--url", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(REPO / "results/pb_f_selector.json"))
    a = ap.parse_args()
    os.umask(0o077)
    {"pick": pick, "judge": lambda: judge(a.cfg, a.url, a.workers), "timing": lambda: timing(a.judge, a.url),
     "report": lambda: report(a.out)}[a.cmd]()


if __name__ == "__main__":
    main()
