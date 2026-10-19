"""Offline decision evaluation on PhysicianBench order steps (natural candidate sets; no gold to inject).

Source: arm llm_log runs (K=5 candidates per step, the first executed; message history logged). An *order step* is a
step where at least one candidate creates a ServiceRequest or MedicationRequest. Each task's code-graded order
checkpoints (validate_service_order / validate_medication_order / validate_service_orders in its public test file)
give the required orders as name/code regexes. A candidate is *right* if it creates a required order not yet placed
earlier in the trajectory (by the executed turns), using the graders' own matching rule (regex on the display text,
or on the code). Items whose candidates include both right and other turns are where a decider can matter.

Deciders: clef (state + options), clef_blind (options only), first (the executed sample), majority (most-sampled),
random (expected). Patient content stays under data/offline/pb_<slug>/; only aggregates go to results/.

  .venv/bin/python scripts/pb_offline_orders.py build|select|report [--slug llm_log_k5/run1]
  .venv/bin/python scripts/pb_offline_orders.py build --slug clef_k5/run1; ... report --slug clef_k5/run1 --out ...
      (the online clef arm: Clef's logged picks on its own trajectories, no replay needed)
"""

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO)]
from src.s1clin.pb_decider import DECISION_QUESTION, ClefDecider, parse_turn  # noqa: E402

PB = REPO / "third_party/physicianbench/tasks/v1"
JOBS = Path("/ocean/projects/cis260032p/yliy/data/physicianbench/jobs")
OUT = Path("/ocean/projects/cis260032p/yliy/data/offline")
BLIND = "A clinical task in an electronic health record. No further information is available."
CREATE = {"fhir_service_request_create": ("service", "code_display", "code_code"),
          "fhir_medication_request_create": ("medication", "medication_display", "medication_code")}


def required_orders(task):
    """[(checkpoint, kind, name_patterns, code_patterns)] from the task's code graders (literal arguments only)."""
    tree = ast.parse((PB / task / "tests/test_outputs.py").read_text())
    out = []
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_checkpoint_")]:
        negative = any(isinstance(n, ast.Assert) and isinstance(n.test, ast.UnaryOp) and isinstance(n.test.op, ast.Not)
                       and "found" in ast.unparse(n.test.operand) for n in ast.walk(fn))  # "must not be ordered"
        for call in [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]:
            kw = {k.arg: k.value for k in call.keywords}
            try:
                if call.func.id in ("validate_service_order", "validate_medication_order"):
                    kind = "service" if "service" in call.func.id else "medication"
                    specs = [(ast.literal_eval(kw["name_patterns"]),
                              ast.literal_eval(kw["code_patterns"]) if "code_patterns" in kw else [])]
                elif call.func.id == "validate_service_orders":
                    kind = "service"
                    specs = [(d["name_patterns"], d.get("code_patterns") or [])
                             for d in ast.literal_eval(kw.get("order_specs") or call.args[0])]
                else:
                    continue
            except (ValueError, KeyError, IndexError):
                continue
            for names, codes in specs:
                out.append({"cp": fn.name[len("test_checkpoint_"):], "kind": kind, "names": names, "codes": codes or [],
                            "negative": negative})
    return out


def matches(spec, kind, display, code):
    if spec["kind"] != kind:
        return False
    return (any(re.search(p, display or "", re.IGNORECASE) for p in spec["names"])
            or any(re.search(p, code or "") for p in spec["codes"]))


def orders_in(turn):
    return [(CREATE[n][0], a.get(CREATE[n][1], ""), str(a.get(CREATE[n][2], "") or ""))
            for n, a in turn if n in CREATE]


def build(slug):
    od = OUT / ("pb_" + slug.replace("/", "_"))
    od.mkdir(parents=True, exist_ok=True, mode=0o700)
    items, n_steps, n_tasks = [], 0, 0
    for jd in sorted((JOBS / slug).iterdir()):
        dec = jd / "logs/agent/decisions.jsonl"
        if not jd.is_dir() or not (jd / "metadata.json").exists() or not dec.exists():
            continue
        n_tasks += 1
        specs = [s for s in required_orders(jd.name) if not s["negative"]]
        placed = set()
        for r in map(json.loads, dec.read_text().splitlines()):
            n_steps += 1
            turns = [parse_turn(c) for c in r["candidates"]]
            ords = [orders_in(t) for t in turns]
            if any(ords) and specs:
                open_specs = [i for i in range(len(specs)) if i not in placed]
                right = [any(matches(specs[i], *o) for o in os_ for i in open_specs) for os_ in ords]
                items.append({"id": f"{jd.name}|{r['step']}", "task": jd.name, "step": r["step"],
                              "n_messages": r.get("n_messages"), "msg_epoch": r.get("msg_epoch"),
                              "online_pick": r["pick"], "online_distribution": r.get("distribution"),
                              "options": r["candidates"],
                              "counts": r["counts"], "right": right, "creates": [bool(o) for o in ords],
                              "n_open_required": len(open_specs)})
            for o in orders_in(turns[r["pick"]]):  # the executed turn places its orders
                placed |= {i for i, s in enumerate(specs) if matches(s, *o)}
    with open(od / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{slug}: {n_tasks} tasks, {n_steps} steps, {len(items)} order steps, "
          f"{sum(len(set(i['right'])) > 1 for i in items)} with right and other candidates")


def replay(path):
    epochs, cur = [], []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["op"] == "reset":
            epochs.append(cur)
            cur = cur[:r["keep"]]
        else:
            cur = cur + [r["msg"]]
    return epochs + [cur]


def select(slug, workers):
    od = OUT / ("pb_" + slug.replace("/", "_"))
    items = [json.loads(l) for l in (od / "items.jsonl").read_text().splitlines()]
    out = od / "selections.jsonl"
    done = {(x["id"], x["decider"]) for x in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    clef, lock, cache = ClefDecider(question=DECISION_QUESTION), __import__("threading").Lock(), {}

    def state(it):
        if it["task"] not in cache:
            cache[it["task"]] = replay(JOBS / slug / it["task"] / "logs/agent/decisions.messages.jsonl")
        return cache[it["task"]][it["msg_epoch"]][:it["n_messages"]]

    def run(job):
        it, d = job
        seed = hashlib.sha256(it["id"].encode()).hexdigest()
        try:
            msgs = state(it) if d == "clef" else [{"role": "user", "content": BLIND}]
            pick, dist, info = clef.choose(msgs, it["options"], seed)
        except Exception as e:
            pick, dist, info = None, None, {"error": repr(e)[:300]}
        with lock, open(out, "a") as f:
            f.write(json.dumps({"id": it["id"], "decider": d, "pick": pick, "distribution": dist,
                                "info": {k: info.get(k) for k in ("error", "input_tokens") if k in info}}) + "\n")

    jobs = [(it, d) for it in items for d in ("clef", "clef_blind")
            if len(it["options"]) > 1 and (it["id"], d) not in done]
    print(f"{len(jobs)} selections", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, jobs))


def report(slug, path):
    od = OUT / ("pb_" + slug.replace("/", "_"))
    items = [json.loads(l) for l in (od / "items.jsonl").read_text().splitlines()]
    sel = {}
    if (od / "selections.jsonl").exists():
        for x in map(json.loads, (od / "selections.jsonl").read_text().splitlines()):
            if x["pick"] is not None:
                sel[(x["id"], x["decider"])] = x
    disc = [it for it in items if len(set(it["right"])) > 1]
    res = {"slug": slug, "order_steps": len(items), "tasks": len({i["task"] for i in items}),
           "steps_any_right": sum(any(i["right"]) for i in items),
           "discriminative_steps": len(disc), "deciders": {}}
    res["random_expected"] = round(sum(sum(i["right"]) / len(i["right"]) for i in disc) / max(len(disc), 1), 4)
    res["deciders"]["first"] = round(sum(i["right"][0] for i in disc) / max(len(disc), 1), 4)
    res["deciders"]["majority"] = round(sum(i["right"][max(range(len(i["counts"])), key=lambda k: (i["counts"][k], -k))]
                                            for i in disc) / max(len(disc), 1), 4)
    if slug.startswith("clef"):  # online Clef picks on its own trajectories (natural sets, on-policy for Clef)
        for i in items:
            if i["online_distribution"]:
                sel[(i["id"], "clef_online")] = {"pick": i["online_pick"], "distribution": i["online_distribution"]}
    for d in ("clef", "clef_blind", "clef_online"):
        got = [(i, sel[(i["id"], d)]) for i in disc if (i["id"], d) in sel]
        if got:
            pos = [p for i, x in got for p, ok in zip(x["distribution"], i["right"]) if ok]
            neg = [p for i, x in got for p, ok in zip(x["distribution"], i["right"]) if not ok]
            au = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / max(len(pos) * len(neg), 1)
            res["deciders"][d] = {"n": len(got), "accuracy": round(sum(i["right"][x["pick"]] for i, x in got) / len(got), 4),
                                  "mass_on_right": round(sum(sum(p for p, ok in zip(x["distribution"], i["right"]) if ok)
                                                             for i, x in got) / len(got), 4),
                                  "auroc_candidate": round(au, 4)}
    # whether-to-order: steps mixing order-creating and other candidates. Compare how often each decider picks an
    # order-creating candidate when one of them places a required order vs when none does (orders the graders do
    # not require are not necessarily wrong; they are only not credited).
    mixed = [i for i in items if len(set(i["creates"])) > 1]
    res["whether_to_order"] = {}
    for name, sub in (("required_available", [i for i in mixed if any(i["right"])]),
                      ("none_required", [i for i in mixed if not any(i["right"])])):
        r = {"n": len(sub), "share_creating": round(sum(sum(i["creates"]) / len(i["creates"]) for i in sub)
                                                    / max(len(sub), 1), 4),
             "first": round(sum(i["creates"][0] for i in sub) / max(len(sub), 1), 4)}
        for d in ("clef", "clef_blind", "clef_online"):
            got = [(i, sel[(i["id"], d)]) for i in sub if (i["id"], d) in sel]
            if got:
                r[d] = round(sum(i["creates"][x["pick"]] for i, x in got) / len(got), 4)
        res["whether_to_order"][name] = r
    res["which_order_steps"] = sum(1 for i in items if all(i["creates"]) and len(set(i["right"])) > 1)
    Path(path).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "select", "report"])
    ap.add_argument("--slug", default="llm_log_k5/run1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(REPO / "results/pb_offline_orders.json"))
    a = ap.parse_args()
    os.umask(0o077)
    {"build": lambda: build(a.slug), "select": lambda: select(a.slug, a.workers),
     "report": lambda: report(a.slug, a.out)}[a.cmd]()


if __name__ == "__main__":
    main()
