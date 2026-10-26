"""F, role 1 (docs/research_loop.md): the checker. Every judge answers "is this order indicated?" for every item of the
task-prior pool from the task text alone; System One models (scores from R3 / R3-d3 / M, unchanged) against the
same-scale LLM in every reasonable configuration:
  A   Qwen, no thinking, lettered choice, P from the first token's log-probabilities (L's scores, unchanged)
  AG  the same with Gemma 4 31B-it (FP8)
  B   Qwen, no thinking, one line 'P=<0-100>'
  C2  Qwen, thinking, max 2,048 generated tokens, then 'P=<0-100>' (the G1 comparator prompt); C8 with 8,192
  D   Qwen, no thinking, 5 sampled letters at the model's default temperature; P = share of Yes
C2 and C8 run on a seeded 20 of the 54 tasks. Timing: the same 200 seeded items of those tasks per judge, 8 concurrent
requests, 10 warm-up items, one H100 per server. Aggregates only in results/pb_f_checker.json.

  .venv/bin/python scripts/pb_f.py judge --cfg B [--url http://127.0.0.1:8011/v1*1,...] [--workers 32]
  .venv/bin/python scripts/pb_f.py timing --judge pplx12|A|AG|B|C2|C8|D [--url ...]
  .venv/bin/python scripts/pb_f.py report
Patient content stays under data/offline/pb_f/.
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from pb_offline_orders import matches, required_orders  # noqa: E402
from pb_prior import NO, OD as PRIOR_OD, QUESTION, YES, pools, render_order, tasks  # noqa: E402
from pb_replay import cases  # noqa: E402
from src.s1clin.pb_decider import DECIDER_URLS  # noqa: E402

FD = Path("/ocean/projects/cis260032p/yliy/data/offline/pb_f")
S1 = ("pplx12", "d3", "clef", "linne", "amos", "decider31b")
RUNS = {"qwen": ("llm_log_k5/run1", "llm_log_k5/run3"), "gemma4": ("llm_log_k1__gemma4/run1", "llm_log_k1__gemma4/run2")}
MODEL = {"qwen": "qwen3.8-27b", "gemma": "gemma-4-31b-it"}
P_LINE = re.compile(r"P\s*=\s*([0-9]+(?:\.[0-9]+)?)")
ASK_B = " Reply with one line 'P=<number from 0 to 100>', your probability that it is indicated, and nothing else."
ASK_C = (" Think it through, then end your reply with one line 'P=<number from 0 to 100>', your probability that it is "
         "indicated.")  # the G1 comparator's ending (pb_screen.LLM_PROMPT)
BUDGET = {"C2": 2048, "C8": 8192}


def c_tasks():
    ts = sorted(pools(drop=False))
    random.Random("F-C").shuffle(ts)
    return set(ts[:20])


def items(cfg=None):
    full = pools(drop=False)
    keep = c_tasks() if cfg in BUDGET else set(full)
    return [(t, o) for t in sorted(full) if t in keep for o in full[t]]


def timing_items():
    its = items("C8")
    random.Random("F-timing").shuffle(its)
    return its[:210]  # 10 warm-up + 200 timed


def letters(t, o):
    ls = ["A", "B"]
    random.Random(f"{t}|{render_order(o)}").shuffle(ls)  # the same letters as the System One request (pb_prior.score)
    return ls


def post(url, body):
    r = requests.post(f"{url}/chat/completions", timeout=3600, json=body)
    r.raise_for_status()
    return r.json()


def judge_one(cfg, url, instruction, t, o):
    """(P(indicated) or None, info) for one item."""
    model = MODEL["gemma" if cfg == "AG" else "qwen"]
    question = QUESTION.format(order=render_order(o))
    if cfg in ("A", "AG", "D"):
        ls = letters(t, o)
        opts = {ls[0]: YES, ls[1]: NO}
        prompt = (f"{instruction}\n\n---\n{question}\n\n" + "\n".join(f"{k}. {v}" for k, v in sorted(opts.items()))
                  + "\n\nAnswer with the letter of one option only.")
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1,
                "chat_template_kwargs": {"enable_thinking": False}}
        if cfg == "D":
            m = post(url, body | {"n": 5})
            votes = [c["message"]["content"].strip()[:1] for c in m["choices"]]
            return sum(v == ls[0] for v in votes) / len(votes), {"tokens": m["usage"]["completion_tokens"]}
        m = post(url, body | {"temperature": 0, "logprobs": True, "top_logprobs": 20})
        top = m["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        lp = {k: max((x["logprob"] for x in top if x["token"].strip() == k), default=-50.0) for k in opts}
        z = sum(math.exp(v) for v in lp.values())
        return math.exp(lp[ls[0]]) / z, {"tokens": 1}
    prompt = f"TASK:\n{instruction}\n\n---\n{question}" + (ASK_B if cfg == "B" else ASK_C)
    body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if cfg == "B":
        body |= {"max_tokens": 16, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    else:
        body |= {"max_tokens": BUDGET[cfg]}  # thinking on, the model's default sampling (as the G1 comparator)
    m = post(url, body)
    v = P_LINE.findall(m["choices"][0]["message"]["content"] or "")
    return (min(float(v[-1]), 100) / 100 if v else None), {"tokens": m["usage"]["completion_tokens"],
                                                           **({} if v else {"no_answer": True})}


def route(urls, key):
    """A replica per key (here one item), spread by hash."""
    us = [u.split("*")[0] for u in urls.split(",")]
    return us[int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(us)]


def judge(cfg, urls, workers):
    FD.mkdir(parents=True, exist_ok=True)
    out, lock = FD / f"checker_{cfg}.jsonl", threading.Lock()
    done = {(x["task"], x["order"]) for x in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    info = tasks()
    todo = [(t, o) for t, o in items(cfg) if (t, render_order(o)) not in done]
    random.Random(f"F-{cfg}").shuffle(todo)  # spread the load over the replicas (items are routed one by one)
    print(f"{out.name}: {len(todo)} to judge", flush=True)

    def run(item):
        t, o = item
        t0 = time.time()
        try:
            p, extra = judge_one(cfg, route(urls, f"{t}|{render_order(o)}"), info[t][0], t, o)
        except Exception as e:
            p, extra = None, {"error": repr(e)[:300]}
        with lock, open(out, "a") as f:
            f.write(json.dumps({"task": t, "order": render_order(o), "p": p, "s": round(time.time() - t0, 3)} | extra,
                               ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, todo))


def timing(name, url):
    """Wall time for 200 items at 8 concurrent requests (after 10 warm-up), on the judge's own server."""
    FD.mkdir(parents=True, exist_ok=True)
    info, its = tasks(), timing_items()

    def one(item):
        t, o = item
        if name in S1:
            ls = letters(t, o)
            r = requests.post(DECIDER_URLS[name], timeout=600, json={
                "instruction": info[t][0], "steps": [], "question": QUESTION.format(order=render_order(o)),
                "options": {ls[0]: YES, ls[1]: NO}})
            r.raise_for_status()
            return 0
        return judge_one(name, url, info[t][0], t, o)[1].get("tokens", 0)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, its[:10]))
        t0 = time.time()
        toks = list(ex.map(one, its[10:]))
        wall = time.time() - t0
    rec = {"judge": name, "items": len(toks), "wall_s": round(wall, 2), "gpu_s_per_judgment": round(wall / len(toks), 4),
           "tokens_per_judgment": round(sum(toks) / len(toks), 1), "concurrency": 8}
    with open(FD / "timing.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(rec)


def auroc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def boot(per, n=2000):
    ts, rng = sorted(per), random.Random(1)
    if not ts:
        return None
    b = sorted(sum(per[t] for t in s) / len(s) for s in ([rng.choice(ts) for _ in ts] for _ in range(n)))
    return [round(b[int(0.025 * n)], 4), round(b[int(0.975 * n) - 1], 4)]


def load_scores():
    sc = {}
    for name in S1 + ("qwen",):
        sc["A" if name == "qwen" else name] = {(x["task"], x["order"]): x["p"] for x in
                                               map(json.loads, (PRIOR_OD / f"prior_scores_{name}.jsonl").read_text().splitlines())}
    for cfg in ("AG", "B", "C2", "C8", "D"):
        f = FD / f"checker_{cfg}.jsonl"
        if f.exists():
            rows = [json.loads(l) for l in f.read_text().splitlines()]
            sc[cfg] = {(x["task"], x["order"]): (0.5 if x["p"] is None else x["p"]) for x in rows}
            sc[cfg + "__noanswer"] = sum(x["p"] is None for x in rows) / max(1, len(rows))
    return sc


def report(path):
    sc = load_scores()
    judges = [j for j in sc if not j.endswith("__noanswer")]
    full = pools(drop=False)
    cs = [c for agent in RUNS for c in cases(RUNS[agent]) if c["task"] in full and c["open"]]
    per = {j: {} for j in judges}  # judge -> task -> [case AUROC]
    sec = {j: {} for j in judges}  # task-level: all required vs not
    brier = {j: [] for j in judges}
    for c in cs:
        t, specs = c["task"], [s for s in required_orders(c["task"]) if not s["negative"]]
        items_ = full[t]
        pos = [o for o in items_ if any(matches(specs[i], *o) for i in c["open"])]
        neg = [o for o in items_ if not any(matches(s, *o) for s in specs)]
        if not pos or not neg:
            continue
        for j in judges:
            p = [sc[j].get((t, render_order(o))) for o in pos]
            q = [sc[j].get((t, render_order(o))) for o in neg]
            if None in p or None in q:
                continue
            per[j].setdefault(t, []).append(auroc(p, q))
    for t, its in full.items():
        specs = [s for s in required_orders(t) if not s["negative"]]
        lab = [any(matches(s, *o) for s in specs) for o in its]
        for j in judges:
            v = [sc[j].get((t, render_order(o))) for o in its]
            if None in v or not any(lab) or all(lab):
                continue
            sec[j][t] = auroc([x for x, y in zip(v, lab) if y], [x for x, y in zip(v, lab) if not y])
            brier[j] += [(x - y) ** 2 for x, y in zip(v, lab)]
    task_mean = {j: {t: sum(v) / len(v) for t, v in per[j].items()} for j in judges}
    sec_ = {j: sec[j] for j in judges}
    sc_mean = lambda d: {t: sum(d[j][t] for j in S1) / len(S1) for t in set.intersection(*(set(d[j]) for j in S1))}  # noqa: E731
    task_mean["S1mean"], sec_["S1mean"] = sc_mean(task_mean), sc_mean(sec_)
    task_mean["S1gemma"] = {t: (task_mean["linne"][t] + task_mean["decider31b"][t]) / 2
                            for t in set(task_mean["linne"]) & set(task_mean["decider31b"])}
    res = {"cases": len(cs), "c_tasks": len(c_tasks()), "judges": {}}
    for j, d in task_mean.items():
        res["judges"][j] = {"auroc_open": round(sum(d.values()) / len(d), 4) if d else None, "ci95": boot(d), "tasks": len(d)}
        if j in sec_ and sec_[j]:
            res["judges"][j]["auroc_required"] = round(sum(sec_[j].values()) / len(sec_[j]), 4)
        if j in brier and brier[j]:
            res["judges"][j]["brier"] = round(sum(brier[j]) / len(brier[j]), 4)
        if j + "__noanswer" in sc:
            res["judges"][j]["no_answer"] = round(sc[j + "__noanswer"], 4)
    res["diff"] = {}
    for x, y in (("S1mean", "A"), ("S1gemma", "AG"), ("S1mean", "B"), ("S1mean", "C2"), ("S1mean", "C8"),
                 ("S1mean", "D"), ("A", "B"), ("A", "C8"), ("A", "D"), ("C8", "C2")) + tuple((s, "A") for s in S1):
        if x not in task_mean or y not in task_mean:
            continue
        ks = set(task_mean[x]) & set(task_mean[y])
        d = {t: task_mean[x][t] - task_mean[y][t] for t in ks}
        if d:
            res["diff"][f"{x}_minus_{y}"] = {"tasks": len(d), "mean": round(sum(d.values()) / len(d), 4), "ci95": boot(d)}
    tf = FD / "timing.jsonl"
    if tf.exists():
        res["timing"] = {x["judge"]: x for x in map(json.loads, tf.read_text().splitlines())}
    Path(path).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["judge", "timing", "report"])
    ap.add_argument("--cfg", choices=["AG", "B", "C2", "C8", "D"])
    ap.add_argument("--judge")
    ap.add_argument("--url", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=str(REPO / "results/pb_f_checker.json"))
    a = ap.parse_args()
    os.umask(0o077)
    {"judge": lambda: judge(a.cfg, a.url, a.workers), "timing": lambda: timing(a.judge, a.url),
     "report": lambda: report(a.out)}[a.cmd]()


if __name__ == "__main__":
    main()
