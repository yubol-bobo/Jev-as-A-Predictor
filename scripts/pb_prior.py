"""P8 (docs/research_loop.md): a pool of candidate orders built from the task text alone, ranked by a System One model.

The P6/P7' screens draw their pool from the run's own proposals, which rarely hold the required orders the run
misses. Here the pool comes from outside the run, from short inputs only: Qwen without thinking reads the task
instruction (no chart) and lists candidate orders, 8 samples, union by normalized name; items matching the base
run's orders placed before its first FINISH are dropped (by name, as an online screen would); pplx12 ranks the rest
with the task-only question of P6c. Target per task: the required-order specs the base run had not met at its first
FINISH. Aggregates only in results/.

  .venv/bin/python scripts/pb_prior.py gen [--url http://127.0.0.1:8013/v1] [--workers 4]
  S1CLIN_DECIDER=pplx12 .venv/bin/python scripts/pb_prior.py score [--placed] [--workers 4]
  .venv/bin/python scripts/pb_prior.py report [--out results/pb_prior.json]
  .venv/bin/python scripts/pb_prior.py export [--name d3]   (-> prior_ranked_<name>.json, read by --ranker prior-<name>)
  S1CLIN_DECIDER=d3 .venv/bin/python scripts/pb_prior.py score --full   (the whole pool, for export; R3-d3)
  S1CLIN_DECIDER=pplx12 .venv/bin/python scripts/pb_prior.py dedup [--workers 4]   (P10)
  .venv/bin/python scripts/pb_prior.py factorized   (-> results/pb_p10.json)
  S1CLIN_DECIDER=qwen .venv/bin/python scripts/pb_prior.py score --full   (L: the LLM, thinking off, as the ranker:
      the same input, question and options, P(Yes) from the answer letter's log-probabilities; QWEN_URL)
Patient content stays under data/offline/pb_screen/.
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from pb_offline_orders import matches, orders_in, required_orders  # noqa: E402
from pb_screen import JOBS, NO, OD, QUESTION, STATE_RUN, YES, norm, render_order  # noqa: E402
from src.s1clin.pb_decider import (DECIDER_URLS, DEDUP_INSTRUCTION, DEDUP_NO, DEDUP_QUESTION, DEDUP_YES,  # noqa: E402
                                   parse_turn, turn_kind)

N_SAMPLES = 8
GEN_PROMPT = (
    "A clinical task assigned to an AI agent that works in a patient's electronic health record:\n\n{task}\n\n---\n"
    "Without seeing the chart, list the orders (laboratory tests, imaging, referrals or consults, procedures, "
    "medications) that could be indicated as part of completing this task. Up to 15, one per line, each in the form\n"
    "ServiceRequest: <name> (code <code>)\nor\nMedicationRequest: <name> (code <code>)\n"
    "leaving out the code part when unsure. Nothing else.")
PLACED_QUESTION = (
    "The state shows a clinical task assigned to an AI agent that works in an electronic health record through FHIR "
    "tools, and the orders the agent has placed so far. Consider this order for the patient: {order}. Is this order "
    "indicated as part of completing the task and still missing, that is, not the same as or equivalent to an order "
    "already placed?")
PLACED_YES, PLACED_NO = "Yes, indicated and still missing", "No, not indicated or already placed"
QWEN_URL = os.environ.get("QWEN_URL", "http://127.0.0.1:8011/v1")


def llm_choice(instruction, question, options):
    """L: the LLM (Qwen, thinking off) answers the System One model's request as a lettered choice in one token; the
    distribution over the option letters comes from the first token's log-probabilities, renormalized."""
    import math
    prompt = (f"{instruction}\n\n---\n{question}\n\n" + "\n".join(f"{k}. {v}" for k, v in options.items())
              + "\n\nAnswer with the letter of one option only.")
    r = requests.post(f"{QWEN_URL}/chat/completions", timeout=600, json={
        "model": "qwen3.8-27b", "messages": [{"role": "user", "content": prompt}], "max_tokens": 1,
        "temperature": 0, "logprobs": True, "top_logprobs": 20,
        "chat_template_kwargs": {"enable_thinking": False}})
    r.raise_for_status()
    top = r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    lp = {k: max((t["logprob"] for t in top if t["token"].strip() == k), default=-50.0) for k in options}
    z = sum(math.exp(v) for v in lp.values())
    return {k: math.exp(v) / z for k, v in lp.items()}


LINE = re.compile(r"(ServiceRequest|MedicationRequest)\s*:\s*(.+?)(?:\s*\(code\s*([^)]*)\))?\s*$")


def tasks():
    """task -> (instruction, required specs, indices of specs open at the base run's first FINISH, placed names,
    placed orders)."""
    states = json.loads((OD / "states.json").read_text())
    out = {}
    for t, (instruction, _) in states.items():
        specs = [s for s in required_orders(t) if not s["negative"]]
        rows = [json.loads(line) for line in (JOBS / STATE_RUN / t / "logs/agent/decisions.jsonl").read_text().splitlines()]
        last = next((r for r in rows if turn_kind(r["candidates"][r["pick"]]) == "finish"), rows[-1])
        placed = [o for r in rows[:rows.index(last)] for o in orders_in(parse_turn(r["candidates"][r["pick"]]))]
        met = {i for i, s in enumerate(specs) if any(matches(s, *o) for o in placed)}
        out[t] = (instruction, specs, [i for i in range(len(specs)) if i not in met], {norm(o) for o in placed}, placed)
    return out


def gen(url, workers):
    path, lock = OD / "prior_gen.jsonl", threading.Lock()
    done = {json.loads(line)["task"] for line in path.read_text().splitlines()} if path.exists() else set()
    todo = [(t, v[0]) for t, v in tasks().items() if t not in done]
    print(f"{path.name}: {len(todo)} tasks to generate", flush=True)

    def run(item):
        t, instruction = item
        t0 = time.time()
        r = requests.post(f"{url}/chat/completions", timeout=1800, json={
            "model": "qwen3.8-27b", "n": N_SAMPLES, "temperature": 1.0, "max_tokens": 1500,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": GEN_PROMPT.format(task=instruction)}]})
        r.raise_for_status()
        samples = [c["message"]["content"] or "" for c in r.json()["choices"]]
        with lock, open(path, "a") as f:
            f.write(json.dumps({"task": t, "samples": samples, "s": round(time.time() - t0, 2)}, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, todo))


def pools(drop=True):
    """task -> list of (kind, display, code) after dedup and (drop) dropping what the base run placed."""
    info = tasks()
    out = {}
    for g in map(json.loads, (OD / "prior_gen.jsonl").read_text().splitlines()):
        seen, pool = set(info[g["task"]][3]) if drop else set(), []
        for text in g["samples"]:
            for line in text.splitlines():
                m = LINE.search(line.strip().lstrip("-*0123456789. "))
                if not m:
                    continue
                o = ("medication" if m[1] == "MedicationRequest" else "service", m[2].strip(), (m[3] or "").strip())
                if norm(o) not in seen:
                    seen.add(norm(o))
                    pool.append(o)
        out[g["task"]] = pool
    return out


def score(workers, placed=False, full=False):
    """P(Yes) per pool order on the task text; with `placed` (P8b) also the run's placed orders, asking for an order
    that is indicated and not yet placed in any equivalent form. `full` scores the whole pool (nothing dropped), as
    `export` needs (R3-d3)."""
    name = os.environ.get("S1CLIN_DECIDER", "pplx12")
    path, lock = OD / f"prior_scores_{name}{'_placed' if placed else ''}.jsonl", threading.Lock()
    done = {(x["task"], x["order"]) for x in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
    info = tasks()
    todo = [(t, o) for t, pool in pools(drop=not (placed or full)).items() for o in pool if (t, render_order(o)) not in done]
    print(f"{path.name}: {len(todo)} to score", flush=True)

    def run(item):
        t, o = item
        letters = ["A", "B"]
        random.Random(f"{t}|{render_order(o)}").shuffle(letters)
        t0 = time.time()
        try:
            steps = [{"action": "Orders the agent has placed so far",
                      "results": ["\n".join(map(render_order, info[t][4])) or "(none)"]}] if placed else []
            question = (PLACED_QUESTION if placed else QUESTION).format(order=render_order(o))
            options = {letters[0]: PLACED_YES if placed else YES, letters[1]: PLACED_NO if placed else NO}
            if name == "qwen":  # L: the LLM as the ranker, same input (no steps: --placed is not supported)
                p = llm_choice(info[t][0], question, options)[letters[0]]
            else:
                r = requests.post(DECIDER_URLS[name], timeout=600, json={
                    "instruction": info[t][0], "steps": steps, "question": question, "options": options})
                r.raise_for_status()
                p = r.json()["distribution"][letters[0]]
        except Exception as e:
            p = None
            print(repr(e)[:200], flush=True)
        rec = {"task": t, "order": render_order(o), "p": p} | ({"s": round(time.time() - t0, 3)} if name == "qwen" else {})
        with lock, open(path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, todo))


def dedup(workers):
    """P10: P(the candidate repeats a placed order) for every item of the whole pool, given only the placed orders."""
    name = os.environ.get("S1CLIN_DECIDER", "pplx12")
    path, lock = OD / f"prior_dedup_{name}.jsonl", threading.Lock()
    done = {(x["task"], x["order"]) for x in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
    info = tasks()
    todo = [(t, o) for t, pool in pools(drop=False).items() if info[t][2] for o in pool if (t, render_order(o)) not in done]
    print(f"{path.name}: {len(todo)} to score", flush=True)

    def run(item):
        t, o = item
        letters = ["A", "B"]
        random.Random(f"dedup|{t}|{render_order(o)}").shuffle(letters)
        try:
            r = requests.post(DECIDER_URLS[name], timeout=600, json={
                "instruction": DEDUP_INSTRUCTION,
                "steps": [{"action": "Orders already placed", "results": ["\n".join(map(render_order, info[t][4])) or "(none)"]}],
                "question": DEDUP_QUESTION.format(order=render_order(o)),
                "options": {letters[0]: DEDUP_YES, letters[1]: DEDUP_NO}})
            r.raise_for_status()
            p = r.json()["distribution"][letters[0]]
        except Exception as e:
            p = None
            print(repr(e)[:200], flush=True)
        with lock, open(path, "a") as f:
            f.write(json.dumps({"task": t, "order": render_order(o), "p": p}, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, todo))


def factorized(path, name="pplx12"):
    """P10 report: top-m coverage of the open specs when the whole pool is ranked by P(indicated | task) x
    (1 - P(repeats a placed order)), vs the name-drop ranking of P8, a random m, and an oracle that drops every item
    matching an already-met spec; plus the share of top-m slots that repeat a met spec. `filtered` is the online
    arm's ranking (O1, R): pplx12's repeat judge drops items with P >= 0.5, the rest in `name`'s order. For a decider
    other than pplx12 (R3-d3), the repeat judge stays pplx12's and `name`'s filtered ranking is compared with
    pplx12's on the same tasks."""
    info, full = tasks(), pools(drop=False)
    load = lambda f: {(x["task"], x["order"]): x["p"] for x in map(json.loads, (OD / f).read_text().splitlines())}
    ind, dup = load(f"prior_scores_{name}.jsonl"), load("prior_dedup_pplx12.jsonl")
    ind12 = load("prior_scores_pplx12.jsonl") if name != "pplx12" else ind
    filtered = lambda sc: lambda t, items, placed: sorted((o for o in items if dup[(t, render_order(o))] < 0.5),
                                                          key=lambda o: -sc[(t, render_order(o))])
    rankers = {
        "name_drop": lambda t, items, placed: sorted((o for o in items if norm(o) not in placed), key=lambda o: -ind[(t, render_order(o))]),
        "factorized": lambda t, items, placed: sorted(items, key=lambda o: -ind[(t, render_order(o))] * (1 - dup[(t, render_order(o))])),
        "dedup_only": lambda t, items, placed: sorted(items, key=lambda o: dup[(t, render_order(o))]),
        "filtered": filtered(ind),
    }
    if name != "pplx12":
        rankers["filtered_pplx12"] = filtered(ind12)
    res = {}
    for m in (3, 5, 8):
        out = {}
        for rk, f in rankers.items():
            cov, var, ts = [], [], []
            for t, (_, specs, open_, placed_names, _) in info.items():
                if not open_ or t not in full:
                    continue
                items = full[t]
                if any(sc.get((t, render_order(o))) is None for o in items for sc in (ind, ind12, dup)):
                    continue
                met = [i for i in range(len(specs)) if i not in open_]
                top = f(t, items, placed_names)[:m]
                cov.append(len({i for o in top for i in open_ if matches(specs[i], *o)}) / len(open_))
                var.append(sum(any(matches(specs[i], *o) for i in met) for o in top) / max(1, len(top)))
                ts.append(t)
            out[rk] = {"coverage": round(sum(cov) / len(cov), 4), "ci95": boot(cov), "variant_share": round(sum(var) / len(var), 4),
                       "tasks": len(ts)}
            out[rk]["_cov"] = dict(zip(ts, cov))
        d = [out["factorized"]["_cov"][t] - out["name_drop"]["_cov"][t] for t in out["factorized"]["_cov"]]
        out["factorized_minus_name_drop"] = {"mean": round(sum(d) / len(d), 4), "ci95": boot(d)}
        if name != "pplx12":
            d = [out["filtered"]["_cov"][t] - out["filtered_pplx12"]["_cov"][t] for t in out["filtered"]["_cov"]]
            out[f"filtered_{name}_minus_pplx12"] = {"mean": round(sum(d) / len(d), 4), "ci95": boot(d)}
        for v in out.values():
            v.pop("_cov", None)
        res[f"top{m}"] = out
    Path(path).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def export(name="pplx12"):
    """The online arm's input (P9): per task, the whole pool (nothing dropped) ranked by the decider, best first."""
    sc = {(x["task"], x["order"]): x["p"] for x in map(json.loads, (OD / f"prior_scores_{name}.jsonl").read_text().splitlines())}
    out = {t: sorted(([*o, sc[(t, render_order(o))]] for o in pool), key=lambda x: -x[3])
           for t, pool in pools(drop=False).items()}
    (OD / f"prior_ranked_{name}.json").write_text(json.dumps(out, ensure_ascii=False))
    print(f"prior_ranked_{name}.json: {len(out)} tasks, {sum(map(len, out.values()))} orders")


def boot(vals, n=2000):
    rng = random.Random(0)
    if not vals:
        return None
    b = sorted(sum(s) / len(s) for s in ([rng.choice(vals) for _ in vals] for _ in range(n)))
    return [round(b[int(0.025 * n)], 4), round(b[int(0.975 * n) - 1], 4)]


def report(path, name="pplx12"):
    info, pool = tasks(), pools()
    sp = OD / f"prior_scores_{name}.jsonl"
    sc = {(x["task"], x["order"]): x["p"] for x in map(json.loads, sp.read_text().splitlines())} if sp.exists() else {}
    gens = {g["task"]: g for g in map(json.loads, (OD / "prior_gen.jsonl").read_text().splitlines())}
    rows = []
    for t, (_, specs, open_, _, _) in info.items():
        if not open_ or t not in pool:
            continue
        items = pool[t]
        hits = [{i for i in open_ if matches(specs[i], *o)} for o in items]  # open specs each item matches
        row = {"task": t, "open": len(open_), "pool": len(items),
               "coverage": len(set().union(*hits) if hits else set()) / len(open_), "gen_s": gens[t]["s"]}
        if all((t, render_order(o)) in sc and sc[(t, render_order(o))] is not None for o in items):
            ranked = sorted(range(len(items)), key=lambda j: -sc[(t, render_order(items[j]))])
            for m in (3, 5):
                row[f"top{m}"] = len(set().union(*[hits[j] for j in ranked[:m]]) if items else set()) / len(open_)
                n = len(items)  # random m of the pool: P(an open spec is covered) = 1 - C(n - k, m) / C(n, m)
                row[f"random{m}"] = sum(1 - (comb(n - k, m) / comb(n, m) if n >= m else 0)
                                        for k in (sum(i in h for h in hits) for i in open_)) / len(open_)
        rows.append(row)
    res = {"tasks_with_open": len(rows), "open_specs": sum(r["open"] for r in rows),
           "pool_mean": round(sum(r["pool"] for r in rows) / len(rows), 2),
           "gen_s_median": sorted(r["gen_s"] for r in rows)[len(rows) // 2]}
    for k in ("coverage", "top3", "random3", "top5", "random5"):
        v = [r[k] for r in rows if k in r]
        res[k] = {"mean": round(sum(v) / len(v), 4) if v else None, "ci95": boot(v), "tasks": len(v)}
    v = [r["top5"] - r["random5"] for r in rows if "top5" in r]
    res["top5_minus_random5"] = {"mean": round(sum(v) / len(v), 4) if v else None, "ci95": boot(v)}
    res["specs_covered_pooled"] = round(sum(r["coverage"] * r["open"] for r in rows) / res["open_specs"], 4)
    Path(path).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "score", "report", "export", "dedup", "factorized"])
    ap.add_argument("--url", default="http://127.0.0.1:8013/v1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(REPO / "results/pb_prior.json"))
    ap.add_argument("--placed", action="store_true", help="score/report: P8b, the placed orders in the input")
    ap.add_argument("--full", action="store_true", help="score: the whole pool, nothing dropped (for export)")
    ap.add_argument("--name", default="pplx12", help="export, factorized: the decider whose scores rank the pool")
    a = ap.parse_args()
    os.umask(0o077)
    {"gen": lambda: gen(a.url, a.workers), "score": lambda: score(a.workers, a.placed, a.full),
     "report": lambda: report(a.out, "pplx12_placed" if a.placed else "pplx12"), "export": lambda: export(a.name),
     "dedup": lambda: dedup(a.workers), "factorized": lambda: factorized(REPO / f"results/pb_p10{'' if a.name == 'pplx12' else '_' + a.name}.json", a.name)}[a.cmd]()


if __name__ == "__main__":
    main()
