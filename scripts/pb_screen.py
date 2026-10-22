"""P6 (docs/research_loop.md, cycle 8): a System One model as a fast screener over a wide pool of orders.

For each task with code-graded required orders, the pool is every order any candidate proposed in five logged runs of
the task (pb_pool.py); variants are merged by rubric spec where they match one, the rest by normalized name or code.
The state is the llm_log_k5/run1 state just before its FINISH (or its last step), shortened to the instruction plus
the actions and their results, each result cut so the whole fits about 8k tokens. Each order is one pointwise
question ("is this order indicated for this patient?", Yes / No): a System One decider gives P(Yes) with the state
and blind (order only); the LLM gives a probability after thinking. Aggregates only in results/.

  .venv/bin/python scripts/pb_screen.py build
  S1CLIN_DECIDER=pplx12 .venv/bin/python scripts/pb_screen.py score [--blind | --rich | --task] [--workers 4]
  .venv/bin/python scripts/pb_screen.py llm [--url http://127.0.0.1:8011/v1] [--workers 16]
  .venv/bin/python scripts/pb_screen.py report [--out results/pb_screen.json]
Patient content stays under data/offline/pb_screen/.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from pb_offline_orders import BLIND, CREATE, matches, orders_in, replay, required_orders  # noqa: E402
from src.s1clin.pb_decider import DECIDER_URLS, history_steps, parse_turn, turn_kind  # noqa: E402

JOBS = Path("/ocean/projects/cis260032p/yliy/data/physicianbench/jobs")
OD = Path("/ocean/projects/cis260032p/yliy/data/offline/pb_screen")
STATE_RUN = "llm_log_k5/run1"
POOL_RUNS = ["llm_log_k5/run1", "clef_k5/run1", "clef_k5__pplx12/run1",
             "gate0_orders_k5__pplx12/run1", "gate0_orders_k5__pplx12/run2"]
STATE_CHARS = 28_000  # about 8k tokens for instruction, actions and results together
QUESTION = (
    "The state shows a clinical task assigned to an AI agent that works in an electronic health record through FHIR "
    "tools, with the agent's actions and the information it has gathered so far. Consider this order for the "
    "patient: {order}. Is this order indicated for this patient as part of completing the task (whether or not the "
    "agent has already placed it)?")
YES, NO = "Yes, the order is indicated", "No, the order is not indicated"
WARN = ("Agents like this one often overlook orders that the task requires, so the orders it has placed may be "
        "incomplete. Judge the order from the task and the patient's data, not from whether it resembles what the agent "
        "has done. ")  # G7: the debiasing instruction placed before the question
LLM_PROMPT = (
    "{state}\n\n---\nConsider this order for the patient: {order}.\nIs this order indicated for this patient as part "
    "of completing the task (whether or not the agent has already placed it)? Think it through, then end your reply "
    "with one line 'P=<number from 0 to 100>', your probability that it is indicated.")


def render_order(o):
    kind, display, code = o
    return f"{'MedicationRequest' if kind == 'medication' else 'ServiceRequest'}: {display or '?'}" + (
        f" (code {code})" if code else "")


def norm(o):
    return (o[0], re.sub(r"\W+", " ", (o[1] or "").lower()).strip() or o[2])


def short_state(messages):
    instruction, steps = history_steps(messages)
    n = sum(len(s["results"]) for s in steps) or 1
    fixed = len(instruction) + sum(len(s["action"]) for s in steps)
    cap = max(300, (STATE_CHARS - fixed) // n)
    steps = [{"action": s["action"][:2000], "results": [r if len(r) <= cap else r[:cap] + " [...]" for r in s["results"]]}
             for s in steps]
    return instruction, steps


def build():
    OD.mkdir(parents=True, exist_ok=True, mode=0o700)
    states, qs = {}, []
    for jd in sorted((JOBS / STATE_RUN).iterdir()):
        specs = [s for s in required_orders(jd.name)] if (jd / "metadata.json").exists() else []
        req = [i for i, s in enumerate(specs) if not s["negative"]]
        if not req:
            continue
        rows = [json.loads(l) for l in (jd / "logs/agent/decisions.jsonl").read_text().splitlines()]
        last = next((r for r in rows if turn_kind(r["candidates"][r["pick"]]) == "finish"), rows[-1])
        msgs = replay(jd / "logs/agent/decisions.messages.jsonl")[last["msg_epoch"]][:last["n_messages"]]
        states[jd.name] = short_state(msgs)
        placed = set()
        for r in rows[:rows.index(last)]:
            placed |= {norm(o) for o in orders_in(parse_turn(r["candidates"][r["pick"]]))}
        groups = {}  # group key -> {"variants": Counter, "runs": set, "spec": i or None}
        for run in POOL_RUNS:
            dec = JOBS / run / jd.name / "logs/agent/decisions.jsonl"
            if not dec.exists():
                continue
            for r in map(json.loads, dec.read_text().splitlines()):
                for c in r["candidates"]:
                    for o in orders_in(parse_turn(c)):
                        hit = [i for i, s in enumerate(specs) if matches(s, *o)]
                        key = ("spec", hit[0]) if hit else (("code", o[0], o[2]) if o[2] else ("name",) + norm(o))
                        g = groups.setdefault(key, {"variants": Counter(), "runs": set(), "spec": hit[0] if hit else None})
                        g["variants"][o] += 1
                        g["runs"].add(run)
        for key, g in groups.items():
            o = g["variants"].most_common(1)[0][0]
            label = None if g["spec"] is None else ("forbidden" if specs[g["spec"]]["negative"] else "required")
            qs.append({"id": hashlib.sha256(f"{jd.name}|{key}".encode()).hexdigest()[:16], "task": jd.name,
                       "order": render_order(o), "label": label, "runs": len(g["runs"]),
                       "proposals": sum(g["variants"].values()),
                       "placed": any(norm(v) in placed for v in g["variants"])})
    (OD / "states.json").write_text(json.dumps(states, ensure_ascii=False))
    with open(OD / "questions.jsonl", "w") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    lab = Counter(q["label"] for q in qs)
    print(f"screen: {len(states)} tasks, {len(qs)} orders, labels {dict(lab)}; "
          f"chars per state median {sorted(len(i) + sum(len(s['action']) + sum(map(len, s['results'])) for s in st) for i, st in states.values())[len(states) // 2]}")


RICH_CHARS = 90_000  # about 26k tokens: the System One models read 32k


def build_rich():
    """P6b: the same questions with more of the chart. The run's own state plus the retrieval steps the other four
    pool runs took before their first order (so no other run's orders leak in), identical results kept once, each
    result cut so the whole fits about RICH_CHARS."""
    states = json.loads((OD / "states.json").read_text())
    rich, extra = {}, []
    for task, (instruction, own) in states.items():
        seen = {r for st in own for r in st["results"]}
        more = []
        for run in POOL_RUNS[1:]:
            ml = JOBS / run / task / "logs/agent/decisions.messages.jsonl"
            if not ml.exists():
                continue
            _, steps = history_steps(max(replay(ml), key=len))
            for st in steps:
                if any(n in CREATE for n, _ in parse_turn(st["action"])) or st["action"].startswith(("write_file", "FINISH")):
                    break
                res = [r for r in st["results"] if r not in seen]
                seen |= set(res)
                if res:
                    more.append({"action": st["action"][:2000], "results": res})
        steps = own + more
        n = sum(len(st["results"]) for st in steps) or 1
        cap = max(300, (RICH_CHARS - len(instruction) - sum(len(st["action"]) for st in steps)) // n)
        rich[task] = (instruction, [{"action": st["action"],
                                     "results": [r if len(r) <= cap else r[:cap] + " [...]" for r in st["results"]]}
                                    for st in steps])
        extra.append(len(more))
    (OD / "states_rich.json").write_text(json.dumps(rich, ensure_ascii=False))
    print(f"rich states: {len(rich)} tasks, extra retrieval steps per task median {sorted(extra)[len(extra) // 2]}")


def missed_tasks():
    """Tasks where the state run left a required order unplaced (from all questions, so resumes keep the subset)."""
    return {q["task"] for q in map(json.loads, (OD / "questions.jsonl").read_text().splitlines())
            if q["label"] == "required" and not q["placed"]}


def _answers(name):
    p = OD / f"answers_{name}.jsonl"
    return p, ({x["id"]: x for x in map(json.loads, p.read_text().splitlines()) if x.get("p") is not None}
               if p.exists() else {})


ORDER_TEXT = re.compile(r"^(MedicationRequest|ServiceRequest): (.*?)(?: \(code (.*)\))?$")


def placed_step(instruction, steps, order):
    """G4: a fabricated final step in which the agent places `order`; nothing about the patient changes."""
    m = ORDER_TEXT.match(order)
    kind = "medication" if m[1] == "MedicationRequest" else "service"
    tool = next(t for t, (k, _, _) in CREATE.items() if k == kind)
    _, display_arg, code_arg = CREATE[tool]
    pid = re.search(r'patient(?:_reference)?="(?:Patient/)?([^"]+)"', " ".join(st["action"] for st in steps) + instruction)
    args = {"patient_reference": f"Patient/{pid[1] if pid else 'unknown'}", code_arg: m[3] or "", display_arg: m[2]}
    action = f"{tool}(" + ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items()) + ")"
    result = json.dumps(dict(resourceType=m[1], id="new", status="active", intent="order"))  # a constant stub
    return {"action": action, "results": [result]}


DECISION_TOOL = re.compile(r"^(\w+_create|write_file)\(", re.M)


NOTE_TOOL = re.compile(r"^write_file\(", re.M)
ORDER_TOOL = re.compile(r"^\w+_create\(", re.M)


def without(steps, pattern):
    """G6: the steps without those whose action matches `pattern` (agent notes, or agent orders and messages)."""
    return [st for st in steps if not pattern.search(st["action"])]


def retrieval_only(steps):
    """G5: the steps without any agent decision (orders, appointments, messages, notes): the chart data retrieved."""
    return [st for st in steps if not DECISION_TOOL.search(st["action"])]


def score(blind, workers, rich=False, task=False, subset=False, cf=False, retrieval=False, drop=None, warn=False):
    name = os.environ.get("S1CLIN_DECIDER", "clef")
    url = os.environ.get("S1CLIN_DECIDER_URL") or DECIDER_URLS[name]
    out, done = _answers(name + ("_blind" if blind else "") + ("_rich" if rich else "") + ("_task" if task else "") +
                         ("_cf" if cf else "") + ("_retr" if retrieval else "") + (f"_no{drop}" if drop else "") +
                         ("_warn" if warn else ""))
    states = json.loads((OD / ("states_rich.json" if rich else "states.json")).read_text())
    todo = [q for q in map(json.loads, (OD / "questions.jsonl").read_text().splitlines()) if q["id"] not in done]
    if cf or warn:  # G4, G7: the unplaced orders of the tasks with a missed required order
        missed = missed_tasks()
        todo = [q for q in todo if q["task"] in missed and not q["placed"]]
    if subset:  # G1: only the orders the placed / unplaced AUROCs are computed on
        missed = missed_tasks()
        todo = [q for q in todo if q["placed"] or q["task"] in missed]
    if rich:  # first the subset P6b is judged on: unplaced orders of tasks where the run missed a required order
        missed = missed_tasks()
        todo.sort(key=lambda q: not (q["task"] in missed and not q["placed"]))
    lock = threading.Lock()
    print(f"{out.name}: {len(todo)} to score", flush=True)

    def run(q):
        instruction, steps = (BLIND, []) if blind else (states[q["task"]][0], []) if task else states[q["task"]]
        if cf:
            steps = steps + [placed_step(instruction, steps, q["order"])]
        if retrieval:
            steps = retrieval_only(steps)
        if drop:
            steps = without(steps, {"notes": NOTE_TOOL, "orders": ORDER_TOOL}[drop])
        letters = ["A", "B"]
        random.Random(q["id"]).shuffle(letters)  # same letters for every decider
        payload = {"instruction": instruction, "steps": steps,
                   "question": (WARN if warn else "") + QUESTION.format(order=q["order"]),
                   "options": {letters[0]: YES, letters[1]: NO}}
        t = time.time()
        try:
            r = requests.post(url, json=payload, timeout=600)
            r.raise_for_status()
            o = r.json()
            p, info = o["distribution"][letters[0]], {"input_tokens": o.get("input_tokens")}
        except Exception as e:
            p, info = None, {"error": repr(e)[:300]}
        with lock, open(out, "a") as f:
            f.write(json.dumps({"id": q["id"], "p": p, "s": round(time.time() - t, 3), **info}) + "\n")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, todo))


def llm(url, workers, task=False, retrieval=False, drop=None):
    out, done = _answers("llm_task" if task else "llm_retr" if retrieval else f"llm_no{drop}" if drop else "llm")
    states = json.loads((OD / "states.json").read_text())
    if task:  # the task instruction only (P6c control)
        states = {t: (v[0], []) for t, v in states.items()}
    if retrieval:  # G5 for the LLM: the state without the agent's decisions
        states = {t: (v[0], retrieval_only(v[1])) for t, v in states.items()}
    if drop:  # G6 for the LLM
        states = {t: (v[0], without(v[1], {"notes": NOTE_TOOL, "orders": ORDER_TOOL}[drop])) for t, v in states.items()}
    todo = [q for q in map(json.loads, (OD / "questions.jsonl").read_text().splitlines()) if q["id"] not in done]
    if task or retrieval or drop:  # only the subset the unplaced AUROC is computed on: unplaced orders of tasks with a missed one
        missed = missed_tasks()
        todo = [q for q in todo if q["task"] in missed and not q["placed"]]
    lock = threading.Lock()
    print(f"{out.name}: {len(todo)} to score", flush=True)

    def text(task):
        instruction, steps = states[task]
        return "TASK:\n" + instruction + "".join(
            f"\n\nACTION {k + 1}: {s['action']}" + "".join(f"\nRESULT: {r}" for r in s["results"])
            for k, s in enumerate(steps))

    def run(q):
        t = time.time()
        try:
            r = requests.post(f"{url}/chat/completions", timeout=1800, json={
                "model": "qwen3.8-27b", "max_tokens": 8192,
                "messages": [{"role": "user", "content": LLM_PROMPT.format(state=text(q["task"]), order=q["order"])}]})
            r.raise_for_status()
            m = r.json()
            c = m["choices"][0]["message"]["content"] or ""
            v = re.findall(r"P\s*=\s*([0-9]+(?:\.[0-9]+)?)", c)
            p = min(float(v[-1]), 100) / 100 if v else None
            info = {"completion_tokens": m["usage"]["completion_tokens"], **({} if v else {"error": "no P= line"})}
        except Exception as e:
            p, info = None, {"error": repr(e)[:300]}
        with lock, open(out, "a") as f:
            f.write(json.dumps({"id": q["id"], "p": p, "s": round(time.time() - t, 3), **info}) + "\n")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, todo))


def auroc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def _boot(per, n=2000):
    """Percentile 95% CI of the mean over tasks, resampling tasks."""
    ts, rng = list(per), random.Random(1)
    if not ts:
        return None
    b = sorted(sum(per[t] for t in s) / len(s) for s in ([rng.choice(ts) for _ in ts] for _ in range(n)))
    return [round(b[int(0.025 * n)], 4), round(b[int(0.975 * n) - 1], 4)]


def report(path):
    qs = [json.loads(l) for l in (OD / "questions.jsonl").read_text().splitlines()]
    scorers = {f.stem[len("answers_"):]: _answers(f.stem[len("answers_"):])[1] for f in sorted(OD.glob("answers_*.jsonl"))}
    scorers = {k: {i: a["p"] for i, a in v.items()} for k, v in scorers.items()}
    scorers["frequency_runs"] = {q["id"]: q["runs"] + q["proposals"] / 1e4 for q in qs}
    by_task = defaultdict(list)
    for q in qs:
        by_task[q["task"]].append(q)
    rng = random.Random(0)
    res = {"tasks": len(by_task), "orders": len(qs), "labels": dict(Counter(str(q["label"]) for q in qs)),
           "required_not_placed": sum(q["label"] == "required" and not q["placed"] for q in qs), "scorers": {}}
    for name, sc in scorers.items():
        per, rec, times, unpl, unpl_t, pl_t = {}, [], [], [], {}, {}
        for t, tq in by_task.items():
            tq = [q for q in tq if q["id"] in sc and q["label"] != "forbidden"]
            a = auroc([sc[q["id"]] for q in tq if q["label"] == "required"], [sc[q["id"]] for q in tq if q["label"] is None])
            if a is not None:
                per[t] = a
            # the decision-relevant subset: orders the run has not placed (what a screen could add)
            u = auroc([sc[q["id"]] for q in tq if q["label"] == "required" and not q["placed"]],
                      [sc[q["id"]] for q in tq if q["label"] is None and not q["placed"]])
            if u is not None:
                unpl.append(u)
                unpl_t[t] = u
            pa = auroc([sc[q["id"]] for q in tq if q["label"] == "required" and q["placed"]],
                       [sc[q["id"]] for q in tq if q["label"] is None and q["placed"]])
            if pa is not None:
                pl_t[t] = pa
            un = sorted((q for q in tq if not q["placed"]), key=lambda q: -sc[q["id"]])
            miss = [q for q in un if q["label"] == "required"]
            if miss:
                rec.append(sum(q["label"] == "required" for q in un[:5]) / len(miss))
        ts = list(per)
        boot = sorted(sum(per[t] for t in s) / len(s) for s in ([rng.choice(ts) for _ in ts] for _ in range(2000))) if ts else []
        ans = _answers(name)[1] if name != "frequency_runs" else {}
        res["scorers"][name] = {
            "scored": len(sc), "tasks": len(ts),
            "auroc_mean_per_task": round(sum(per.values()) / len(ts), 4) if ts else None,
            "ci95": [round(boot[50], 4), round(boot[1949], 4)] if boot else None,
            "auroc_unplaced_mean_per_task": round(sum(unpl) / len(unpl), 4) if unpl else None, "tasks_unplaced": len(unpl),
            "unplaced_ci95": _boot(unpl_t), "unplaced_by_task": {t: round(v, 4) for t, v in sorted(unpl_t.items())},
            "auroc_placed_mean_per_task": round(sum(pl_t.values()) / len(pl_t), 4) if pl_t else None,
            "placed_ci95": _boot(pl_t), "tasks_placed": len(pl_t),
            "recall_at5_required_not_placed": round(sum(rec) / len(rec), 4) if rec else None, "tasks_with_missed": len(rec),
            "median_s_per_question": sorted(a["s"] for a in ans.values())[len(ans) // 2] if ans else None,
            "mean_p": {lab: round(sum(sc[q["id"]] for q in qs if q["id"] in sc and str(q["label"]) == lab) /
                                  max(1, sum(1 for q in qs if q["id"] in sc and str(q["label"]) == lab)), 4)
                       for lab in ("required", "None", "forbidden")} if name != "frequency_runs" else None}
    # state minus blind, paired by task
    for name in [n for n in scorers if n + "_blind" in scorers]:
        a, b = res["scorers"][name], res["scorers"][name + "_blind"]
        if a["auroc_mean_per_task"] is not None and b["auroc_mean_per_task"] is not None:
            res.setdefault("state_minus_blind", {})[name] = round(a["auroc_mean_per_task"] - b["auroc_mean_per_task"], 4)
    Path(path).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def cf_report(path, models=("pplx12", "d3")):
    """G4: P(Yes) with the state vs the state plus a fabricated step placing the order, paired by question."""
    qs = {q["id"]: q for q in map(json.loads, (OD / "questions.jsonl").read_text().splitlines())}
    res = {}
    for m in models:
        a, b = (_answers(m)[1], _answers(m + "_cf")[1])
        ids = [i for i in b if i in a]
        out = {"questions": len(ids)}
        for lab in ("required", "None"):
            d = [b[i]["p"] - a[i]["p"] for i in ids if str(qs[i]["label"]) == lab]
            if d:
                rng = random.Random(0)
                bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
                out[lab] = {"n": len(d), "p_state": round(sum(a[i]["p"] for i in ids if str(qs[i]["label"]) == lab) / len(d), 4),
                            "p_cf": round(sum(b[i]["p"] for i in ids if str(qs[i]["label"]) == lab) / len(d), 4),
                            "delta": round(sum(d) / len(d), 4), "ci95": [round(bs[50], 4), round(bs[1949], 4)]}
        for tag, sc in (("state", a), ("cf", b)):
            u = []
            by = defaultdict(list)
            for i in ids:
                by[qs[i]["task"]].append(i)
            for t, ii in by.items():
                x = auroc([sc[i]["p"] for i in ii if qs[i]["label"] == "required"],
                          [sc[i]["p"] for i in ii if qs[i]["label"] is None])
                if x is not None:
                    u.append(x)
            out[f"auroc_unplaced_{tag}"] = round(sum(u) / len(u), 4) if u else None
        res[m] = out
    Path(path).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "build_rich", "score", "llm", "report", "cf"])
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--task", action="store_true", help="score: the task instruction only, no trajectory")
    ap.add_argument("--drop", choices=["notes", "orders"], help="score/llm: the state without the agent's notes or orders (G6)")
    ap.add_argument("--retrieval", action="store_true", help="score: the state without agent decisions (G5)")
    ap.add_argument("--warn", action="store_true", help="score: the state with a debiasing instruction (G7)")
    ap.add_argument("--cf", action="store_true", help="score: the state plus a fabricated step placing the order (G4)")
    ap.add_argument("--subset", action="store_true", help="score: placed orders and the unplaced orders of tasks "
                    "with a missed required order (G1)")
    ap.add_argument("--rich", action="store_true", help="score: the rich states (P6b)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--url", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--out", default=str(REPO / "results/pb_screen.json"))
    a = ap.parse_args()
    os.umask(0o077)
    {"cf": lambda: cf_report(a.out), "build": build, "build_rich": build_rich, "score": lambda: score(a.blind, a.workers, a.rich, a.task, a.subset, a.cf, a.retrieval, a.drop, a.warn), "llm": lambda: llm(a.url, a.workers, a.task, a.retrieval, a.drop),
     "report": lambda: report(a.out)}[a.cmd]()


if __name__ == "__main__":
    main()
