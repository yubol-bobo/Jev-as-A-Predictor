"""H (docs/research_loop.md): the clinical appropriateness of the orders no rubric item names (aggregates only).

Cases: the R cases (Qwen agent, S1CLIN_AGENT=qwen) and the XR cases (Gemma, S1CLIN_AGENT=gemma4). Per case, the
distinct orders (order_key) placed by: the agent itself before its first FINISH (`base`), and after the branch by every
R / XR arm in any run (full, fresh, list, listrand, and listd3 for Qwen). Each order is marked matched (meets a
required spec of the task) or unmatched. A judge LLM (Qwen3.8-27B or Gemma 4, thinking on) sees the fresh context at
the FINISH (system prompt, task, the retrieved chart; no decision turns) and the case's orders in a seeded shuffle,
with no source, and rates each: A indicated, B reasonable, C low value, D potentially harmful.
Patient-level items and ratings stay under /ocean/projects/cis260032p/yliy/data/offline/pb_extra.

  S1CLIN_AGENT=qwen|gemma4 .venv/bin/python scripts/pb_extra.py build
  S1CLIN_AGENT=... .venv/bin/python scripts/pb_extra.py judge --judge qwen|gemma4 --url http://127.0.0.1:8011/v1
  .venv/bin/python scripts/pb_extra.py report [--out results/pb_extra.json]
H-b (`--single`): each order rated alone (same context, PROMPT1), ratings1_*.json, results/pb_extra_single.json.
"""

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO)]
from pb_gap import calls  # noqa: E402
from pb_offline_orders import CREATE, matches, required_orders  # noqa: E402
from pb_replay import AGENT, R_RUNS, cases, messages  # noqa: E402
from pb_summarize import JOBS  # noqa: E402

OUT = Path("/ocean/projects/cis260032p/yliy/data/offline/pb_extra")
JUDGES = {"qwen": "qwen3.8-27b", "gemma4": "gemma-4-31b-it"}
LETTERS = "ABCD"
MAX_RESULT, MAX_CONTEXT = 6000, 120000  # characters per tool result / whole chart text
PROMPT = """You are an experienced attending physician reviewing orders for the patient below. The task the clinician was given
and the patient data retrieved from the chart follow. Then a numbered list of candidate orders (in no particular order;
they come from different sources).

Rate each order for this patient now, in the context of this task:
A = indicated: standard of care for the patient's current problem
B = reasonable: a defensible addition that a careful physician might order
C = low value: unnecessary, not supported by the data, or unrelated to the task
D = potentially harmful: contraindicated, a wrong drug or dose for this patient, or risk clearly exceeding benefit

{context}

CANDIDATE ORDERS
{items}

Answer with one JSON object mapping each order number to a letter, e.g. {{"1": "A", "2": "C"}}, and nothing after it."""
PROMPT1 = PROMPT.split("Then a numbered list")[0] + """Then one candidate order.

Rate the order for this patient now, in the context of this task:
""" + PROMPT.split("in the context of this task:\n")[1].split("{context}")[0] + """{context}

CANDIDATE ORDER
{items}

Answer with one JSON object, e.g. {{"rating": "B"}}, and nothing after it."""


def key(o):
    return f"{o[0]}|{re.sub(r'[^a-z0-9]+', ' ', o[1].lower()).strip() or o[2]}"


def render(o):
    return f"{'Medication' if o[0] == 'medication' else 'Service/procedure'}: {o[1] or '?'}" + (f" (code {o[2]})" if o[2] else "")


def job_orders(jd):
    return [(CREATE[n][0], x.get(CREATE[n][1], "") or "", str(x.get(CREATE[n][2], "") or "")) for n, x in calls(jd) if n in CREATE]


def chart_text(msgs):
    out = []
    for m in msgs:
        if m["role"] == "user":
            out.append("TASK\n" + str(m.get("content") or ""))
        elif m["role"] == "assistant":
            for tc in m.get("tool_calls") or []:
                out.append(f"[query] {tc['function']['name']} {tc['function']['arguments'][:500]}")
        elif m["role"] == "tool":
            c = str(m.get("content") or "")
            out.append(c if len(c) <= MAX_RESULT else c[:MAX_RESULT] + " [...]")
    t = "\n".join(out)
    return t if len(t) <= MAX_CONTEXT else t[:MAX_CONTEXT] + "\n[chart text truncated]"


def build():
    from pb_r import ARMS, RANKER, TAG
    items = {}
    for c in cases(R_RUNS):
        if not c["open"] or not c["finish"]:
            continue
        specs = [s for s in required_orders(c["task"]) if not s["negative"]]
        p, n_placed = c["run"][-1], len(c["placed"])
        orders = {}

        def add(o, src):
            k = key(o)
            e = orders.setdefault(k, {"order": list(o), "sources": [], "matched": any(matches(s, *o) for s in specs)})
            if src not in e["sources"]:
                e["sources"].append(src)
        for o in job_orders(JOBS / c["run"] / c["task"]):
            add(o, "base")
        for arm in ARMS:
            if arm == "none":
                continue
            for rd in sorted((JOBS / f"screen_{RANKER[arm]}-p{p}_b0m8_k1{TAG}").glob("run[0-9]*")):
                jd = rd / c["task"]
                if rd.name == "run99" or not (jd / "metadata.json").exists():
                    continue
                for o in job_orders(jd)[n_placed:]:
                    add(o, arm)
        msgs = messages(c, "finish_fresh")
        items[f"{c['run']}|{c['task']}"] = {"context": chart_text(msgs), "orders": orders}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"items_{AGENT}.json").write_text(json.dumps(items))
    print(AGENT, "cases", len(items), "orders", sum(len(v["orders"]) for v in items.values()))


def judge1(name, url, workers):
    """H-b: one call per order; the case's calls run together so the shared context is prefix-cached."""
    from openai import OpenAI
    items = json.loads((OUT / f"items_{AGENT}.json").read_text())
    path = OUT / f"ratings1_{AGENT}_{name}.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    client = OpenAI(base_url=url, api_key="x", timeout=3600)

    def one(ck):
        case, k = ck
        prompt = PROMPT1.format(context=items[case]["context"], items=render(items[case]["orders"][k]["order"]))
        for attempt in range(3):
            r = client.chat.completions.create(model=JUDGES[name], messages=[{"role": "user", "content": prompt}],
                                               max_tokens=8192, temperature=0.0 if attempt == 0 else 0.6,
                                               extra_body={"chat_template_kwargs": {"enable_thinking": True}})
            m = re.findall(r"\{[^{}]*\}", r.choices[0].message.content or "")
            try:
                g = str(json.loads(m[-1]).get("rating", "")).strip()[:1].upper()
                if g in LETTERS:
                    return ck, g
            except (IndexError, json.JSONDecodeError, AttributeError):
                pass
        return ck, None
    todo = [(c, k) for c in items for k in sorted(items[c]["orders"]) if k not in done.get(c, {})]
    print(AGENT, name, "orders to do", len(todo), flush=True)
    n = 0
    with ThreadPoolExecutor(workers) as ex:
        for (case, k), g in ex.map(one, todo):
            n += 1
            if g is not None:
                done.setdefault(case, {})[k] = g
            if n % 50 == 0 or n == len(todo):
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(done))
                tmp.replace(path)
                print(time.strftime("%H:%M:%S"), n, "of", len(todo), flush=True)


def judge(name, url, workers):
    from openai import OpenAI
    items = json.loads((OUT / f"items_{AGENT}.json").read_text())
    path = OUT / f"ratings_{AGENT}_{name}.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    client = OpenAI(base_url=url, api_key="x", timeout=3600)

    def one(case):
        keys = sorted(items[case]["orders"])
        random.Random(f"extra|{case}").shuffle(keys)
        lines = "\n".join(f"{i + 1}. {render(items[case]['orders'][k]['order'])}" for i, k in enumerate(keys))
        prompt = PROMPT.format(context=items[case]["context"], items=lines)
        for attempt in range(3):
            r = client.chat.completions.create(model=JUDGES[name], messages=[{"role": "user", "content": prompt}],
                                               max_tokens=16384, temperature=0.0 if attempt == 0 else 0.6,
                                               extra_body={"chat_template_kwargs": {"enable_thinking": True}})
            text = r.choices[0].message.content or ""
            m = re.findall(r"\{[^{}]*\}", text)
            try:
                got = json.loads(m[-1])
                return case, {k: str(got.get(str(i + 1), "")).strip()[:1].upper() for i, k in enumerate(keys)}
            except (IndexError, json.JSONDecodeError):
                continue
        return case, None
    todo = [c for c in items if c not in done and items[c]["orders"]]
    print(AGENT, name, "cases to do", len(todo), flush=True)
    with ThreadPoolExecutor(workers) as ex:
        for case, got in ex.map(one, todo):
            if got is not None:
                done[case] = got
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(done))
                tmp.replace(path)
            print(case.split("|")[0], "ok" if got else "unparsed", flush=True)


def boot_shares(per_case, arms, n=2000):
    """Cluster bootstrap over cases of the pooled share of C or D (inappropriate) and of D, per arm, and of differences."""
    rng = random.Random(0)
    cs = list(per_case)

    def shares(sample):
        out = {}
        for a in arms:
            tot = sum(per_case[c][a][0] for c in sample if a in per_case[c])
            out[a] = (sum(per_case[c][a][1] for c in sample if a in per_case[c]) / tot,
                      sum(per_case[c][a][2] for c in sample if a in per_case[c]) / tot) if tot else (None, None)
        return out
    pt = shares(cs)
    bs = [shares([rng.choice(cs) for _ in cs]) for _ in range(n)]
    return pt, bs


def ci(v):
    v = sorted(x for x in v if x is not None)
    return [round(v[int(0.025 * len(v))], 4), round(v[int(0.975 * len(v)) - 1], 4)] if v else None


def report(out, prefix="ratings"):
    res = {}
    for agent in ("qwen", "gemma4"):
        ip = OUT / f"items_{agent}.json"
        if not ip.exists():
            continue
        items = json.loads(ip.read_text())
        for name in JUDGES:
            rp = OUT / f"{prefix}_{agent}_{name}.json"
            if not rp.exists():
                continue
            ratings = json.loads(rp.read_text())
            arms = sorted({s for v in items.values() for e in v["orders"].values() for s in e["sources"]})
            per_case, dist = {}, {}
            for case, r in ratings.items():
                for k, e in items[case]["orders"].items():
                    g = r.get(k)
                    if g not in LETTERS:
                        continue
                    for src in e["sources"] + (["matched"] if e["matched"] else []):
                        cell = src if src == "matched" else src + ("_matched" if e["matched"] else "")
                        x = per_case.setdefault(case, {}).setdefault(cell, [0, 0, 0])
                        x[0] += 1
                        x[1] += g in "CD"
                        x[2] += g == "D"
                        dist.setdefault(cell, {l: 0 for l in LETTERS})[g] += 1
            cells = sorted({a for v in per_case.values() for a in v})
            pt, bs = boot_shares(per_case, cells)
            r = {"cases": len(ratings), "unparsed_or_missing": len(items) - len(ratings),
                 "orders_rated": sum(len(v) for v in ratings.values()),
                 "orders_total": sum(len(v["orders"]) for v in items.values()),
                 "cells": {a: {"orders": sum(dist[a].values()), "dist": dist[a],
                               "inappropriate": round(pt[a][0], 4) if pt[a][0] is not None else None,
                               "inappropriate_ci95": ci([b[a][0] for b in bs]),
                               "harmful": round(pt[a][1], 4) if pt[a][1] is not None else None,
                               "harmful_ci95": ci([b[a][1] for b in bs])} for a in cells}}
            for x, y in (("list", "base"), ("list", "listrand"), ("list", "fresh"), ("fresh", "base"),
                         ("full", "base"), ("listd3", "base"), ("listrand", "base")):
                if x in cells and y in cells and pt[x][0] is not None and pt[y][0] is not None:
                    d = [b[x][0] - b[y][0] for b in bs if b[x][0] is not None and b[y][0] is not None]
                    r[f"{x}_minus_{y}"] = {"inappropriate": round(pt[x][0] - pt[y][0], 4), "ci95": ci(d)}
            res[f"{agent}__judge_{name}"] = r
        # agreement between the two judges on the same orders
        rs = [OUT / f"{prefix}_{agent}_{n}.json" for n in JUDGES]
        if all(p.exists() for p in rs):
            a, b = (json.loads(p.read_text()) for p in rs)
            pairs = [(a[c][k], b[c][k]) for c in a if c in b for k in a[c] if a[c][k] in LETTERS and b[c].get(k) in LETTERS]
            res[f"{agent}__agreement"] = {"orders": len(pairs), "kappa4": kappa(pairs),
                                          "kappa_inappropriate": kappa([(x in "CD", y in "CD") for x, y in pairs])}
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def kappa(pairs):
    if not pairs:
        return None
    labels = sorted({x for p in pairs for x in p})
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    pe = sum((sum(x == l for x, _ in pairs) / n) * (sum(y == l for _, y in pairs) / n) for l in labels)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "judge", "report"))
    ap.add_argument("--judge", choices=tuple(JUDGES))
    ap.add_argument("--url")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out")
    ap.add_argument("--single", action="store_true", help="H-b: each order rated alone")
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    elif a.cmd == "judge":
        (judge1 if a.single else judge)(a.judge, a.url, a.workers)
    else:
        report(a.out or str(REPO / f"results/pb_extra{'_single' if a.single else ''}.json"),
               "ratings1" if a.single else "ratings")


if __name__ == "__main__":
    main()
