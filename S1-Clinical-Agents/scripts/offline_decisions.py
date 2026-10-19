"""Offline decision evaluation on ClinEnv (docs/full_experiment.md, "offline"): the same states and the same
candidate sets, different deciders.

Source: arm llm_log runs (the LLM runs normally, executing the first of K samples; every step logs the K
candidates and the message history). Items are the *submit steps*: steps whose candidates are all submissions of one
type (medication / procedure / diagnosis). Each candidate is scored against the stage gold that is still open at that
step (gold of that type not yet matched by an earlier executed submission in the stage) with ClinEnv's own pair
scores (ICD retrieval + rerank + hierarchical F1; ATC hierarchy with the action gate). Correct = score >= 0.5.

Candidates that differ only in wording or in the `reasoning` argument map to one canonical option (ClinEnv's own
mapping), so `canon` sets are usually smaller than the raw K.
Three versions of each candidate set:
  raw   the candidates exactly as the online deciders see them (tool call with the agent's `reasoning` argument)
  canon the same set, every candidate rendered as action + canonical name (ICD title of its mapped code; ATC
        preprocessed drug name), no reasoning: removes surface-form cues
  inj   canon, and when no candidate is correct, one wrong candidate (seeded) is replaced by an open gold item
        rendered the same canonical way, so every set has a right answer (a single distinct wrong option gets
        the gold one added instead, making a 2-way choice)
Deciders: clef (state + options), clef_blind (options only: detects cues that do not need the patient),
llm_select (the LLM with its message history, thinking on), majority (most-sampled, raw only), first (the executed
sample, raw only).

Patient content (states, candidates) stays under data/offline/ (mode 700); only aggregates go to results/.

  .venv/bin/python scripts/offline_decisions.py build  --slug qwen3.8-27b__llm_log_k5__run1
  .venv/bin/python scripts/offline_decisions.py select --slug ... --deciders clef,clef_blind,llm_select
  .venv/bin/python scripts/offline_decisions.py report --slug ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CE = REPO / "third_party" / "clinenv"
DATA = Path("/ocean/projects/cis260032p/yliy/data")
TREE, HF, OUT = DATA / "clinenv" / "tree", DATA / "clinenv" / "hf", DATA / "offline"
sys.path[:0] = [str(CE), str(REPO)]
os.chdir(CE)  # ClinEnv modules read their data files relative to the repo root
os.environ.setdefault("BENCHMARK_EVAL_CONFIG", str(TREE / "_configs" / "eval_8011.json"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.s1clin.pb_decider import (CE_DECISION_QUESTION, SELECT_PROMPT, ClefDecider, _letters,  # noqa: E402
                                   history_steps, parse_turn)

TAU = 0.5
SUBMIT = {"submit_medication": "medication", "submit_diagnosis": "diagnosis", "submit_procedure": "procedure"}
ARG = {"diagnosis": "diagnosis", "procedure": "procedure"}
BLIND = "A hospitalized patient is being assessed. No further information about the patient is available."


# ---------------------------------------------------------------- parsing and replay

parse_rendering = parse_turn


def replay(path: Path) -> list[list[dict]]:
    epochs, cur = [], []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["op"] == "reset":
            epochs.append(cur)
            cur = cur[:r["keep"]]
        else:
            cur = cur + [r["msg"]]
    return epochs + [cur]


# ---------------------------------------------------------------- ClinEnv scoring with a mapping cache

class Scorer:
    def __init__(self, cache_path: Path):
        from evaluation import scorer as S
        from evaluation.scorers import atc_scoring as A, icd_scoring as I
        self.S, self.A, self.I = S, A, I
        self.path = cache_path
        self.cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def save(self):
        self.path.write_text(json.dumps(self.cache))

    def code(self, text, vocab):  # text -> ICD code, the scorer's own retrieval + rerank
        key = f"{vocab}|{text}"
        if key not in self.cache:
            cands = self.I._retrieve(text, vocab)
            self.cache[key] = self.I._rerank(text, cands) if cands else ""
        return self.cache[key]

    def gold_code(self, g):
        c = g.get("icd_code")
        if not c:
            m = re.match(r"^\[([^\]]+)\]", g.get("description", ""))
            c = m.group(1) if m else None
        return c

    def pair(self, g, sub):
        if g["type"] != sub["type"]:
            return 0.0
        if g["type"] == "medication":
            return self.S._score_medication_pair(g, sub)
        c = self.gold_code(g)
        if not c:
            return 0.0
        vocab = self.S._icd_vocab({**g, "icd_code": c})
        pred = self.code(sub["value"], vocab)
        return self.I.hierarchical_f1(pred, self.I._normalize(c, vocab), vocab) if pred else 0.0

    def canon(self, sub, vocab_of=None):
        """Canonical name: ICD title of the mapped code; ATC-preprocessed drug name."""
        if sub["type"] == "medication":
            return self.A._preprocess(self.A._resolve_alias(sub["value"])).lower()
        code = self.code(sub["value"], vocab_of)
        return self._title(code, vocab_of) or sub["value"]

    def _title(self, code, vocab):
        try:
            return self.I._vocab(vocab).lookup(code) if code else None
        except Exception:  # code missing from the vocabulary graph
            return None

    def gold_canon(self, g):
        if g["type"] == "medication":
            return self.A._preprocess(self.A._resolve_alias(g.get("drug_name", ""))).lower()
        c = self.gold_code(g)
        vocab = self.S._icd_vocab({**g, "icd_code": c})
        return (self._title(self.I._normalize(c, vocab), vocab)
                or re.sub(r"^\[[^\]]+\]\s*", "", g.get("description", "")))


def submission_of(name, args):
    t = SUBMIT[name]
    if t == "medication":
        return {"type": t, "value": args.get("drug_name", ""), "action": args.get("action"),
                "direction": args.get("direction")}
    return {"type": t, "value": args.get(ARG[t], "") or next((v for k, v in args.items() if k != "reasoning"), "")}


def render_canon(sub, name):
    if sub["type"] == "medication":
        extra = f', direction="{sub["direction"]}"' if sub.get("action") == "adjust" and sub.get("direction") else ""
        return f'submit_medication(action="{sub.get("action") or "start"}", drug_name="{name}"{extra})'
    return f'submit_{sub["type"]}({ARG[sub["type"]]}="{name}")'


# ---------------------------------------------------------------- build

def build(slug):
    od = OUT / slug
    od.mkdir(parents=True, exist_ok=True, mode=0o700)
    sc = Scorer(od / "code_cache.json")
    items, n_rows, kinds = [], 0, Counter()
    for run_dir in sorted(TREE.glob(f"*/*/runs/{slug}")):
        s, h = run_dir.parts[-4], run_dir.parts[-3]
        dec = run_dir / "decisions.jsonl"  # the message log is needed only by `select`
        if not (run_dir / "output.json").exists() or not dec.exists():
            continue
        stages = json.loads((HF / s / h / "case" / "case.json").read_text())["stages"]
        rows = [json.loads(l) for l in dec.read_text().splitlines()]
        if rows and rows[0].get("stage") is None:  # logs before the stage field: stage = finalizes executed so far
            n_fin = 0
            for r in rows:
                r["stage"] = n_fin
                n_fin += any(n == "finalize_decision" for n, _ in parse_rendering(r["candidates"][r["pick"]]))
            if n_fin != len(stages):
                kinds["case_skipped_stage_mismatch"] += 1
                continue
        executed = defaultdict(list)  # stage -> executed submissions so far
        for r in rows:
            n_rows += 1
            st = r["stage"]
            r["_n_stages"] = len(stages)
            parsed = [parse_rendering(c) for c in r["candidates"]]
            names = {c[0][0] if len(c) == 1 else "multi" for c in parsed}
            kind = (SUBMIT[next(iter(names))] if len(names) == 1 and next(iter(names)) in SUBMIT else
                    "mixed" if names & set(SUBMIT) else "other")
            kinds[kind] += 1
            ex = parsed[r["pick"]]
            if kind in SUBMIT.values() and st < len(stages):
                gold = [g for g in stages[st]["gts"] if g["type"] == kind and not g.get("unscoreable")]
                open_gold = list(gold)
                for sub in executed[st]:  # remove gold already matched by earlier executed submissions
                    if open_gold:
                        best = max(range(len(open_gold)), key=lambda i: sc.pair(open_gold[i], sub))
                        if sc.pair(open_gold[best], sub) >= TAU:
                            open_gold.pop(best)
                if open_gold and r["n_unique"] >= 1:
                    items.append(make_item(sc, s, h, st, r, parsed, open_gold, kind))
            for name, args in ex:
                if name in SUBMIT:
                    executed[st].append(submission_of(name, args))
        sc.save()
    with open(od / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    os.chmod(od / "items.jsonl", 0o600)
    print(f"{slug}: {n_rows} steps, kinds {dict(kinds)}, {len(items)} submit items")


def make_item(sc, s, h, st, r, parsed, open_gold, kind):
    subs = [submission_of(*p[0]) for p in parsed]
    scores = [max(sc.pair(g, sub) for g in open_gold) for sub in subs]
    vocab = None
    if kind != "medication":
        g0 = open_gold[0]
        vocab = sc.S._icd_vocab({**g0, "icd_code": sc.gold_code(g0)})
    canon_names = [sc.canon(sub, vocab) for sub in subs]
    item_id = f"{s}|{h}|{r['step']}"
    rng = random.Random(hashlib.sha256(item_id.encode()).hexdigest())
    canon = merge([render_canon(sub, n) for sub, n in zip(subs, canon_names)], r["counts"], scores)
    inj = {"options": list(canon["options"]), "counts": list(canon["counts"]), "scores": list(canon["scores"]),
           "injected": None}
    if max(inj["scores"]) < TAU:
        g = rng.choice(open_gold)
        code = sc.gold_code(g) or ""
        inj["gold"] = {"chapter": (code[:1] if code[:1].isalpha() else code[:1] + "xx") if kind != "medication" else "med",
                       "primary": g.get("primary"), "action": g.get("action")}
        gsub = {"type": kind, "value": g.get("drug_name", g.get("description", "")), "action": g.get("action"),
                "direction": g.get("direction")}
        if len(inj["options"]) > 1:  # replace one wrong option: same set size, one right answer
            slot = rng.randrange(len(inj["options"]))
            inj["options"][slot] = render_canon(gsub, sc.gold_canon(g))
            inj["scores"][slot], inj["counts"][slot], inj["injected"] = 1.0, 0, slot
        else:  # a single distinct (wrong) option: add the gold one, in a random position
            slot = rng.randrange(2)
            for k, val in (("options", render_canon(gsub, sc.gold_canon(g))), ("scores", 1.0), ("counts", 0)):
                inj[k].insert(slot, val)
            inj["injected"] = slot
        if inj["options"].count(inj["options"][slot]) > 1:  # canonical gold equals another option: keep one
            keep = [i for i, o in enumerate(inj["options"]) if o != inj["options"][slot] or i == slot]
            inj = {k: ([v[i] for i in keep] if isinstance(v, list) else v) for k, v in inj.items()}
            inj["injected"] = keep.index(slot)
    return {"id": item_id, "subject": s, "hadm": h, "stage": st, "step": r["step"], "type": kind,
            "last_stage": st == r.get("_n_stages", 0) - 1,
            "n_messages": r.get("n_messages"), "msg_epoch": r.get("msg_epoch"), "online_pick": r["pick"],
            "online_distribution": r.get("distribution"), "q": r.get("q", 1),
            "raw": {"options": r["candidates"], "counts": r["counts"], "scores": scores},
            "canon": canon, "inj": inj, "n_open_gold": len(open_gold)}


def merge(options, counts, scores):
    out = {"options": [], "counts": [], "scores": []}
    for o, c, s in zip(options, counts, scores):
        if o in out["options"]:
            i = out["options"].index(o)
            out["counts"][i] += c
            out["scores"][i] = max(out["scores"][i], s)
        else:
            out["options"].append(o); out["counts"].append(c); out["scores"].append(s)
    return out


# ---------------------------------------------------------------- select

class _Chat:
    def __init__(self, url):
        import openai
        self.c = openai.OpenAI(api_key="EMPTY", base_url=url, timeout=3600)

    def chat(self, messages, tools=None, reasoning_effort=None):
        r = self.c.chat.completions.create(model="qwen3.8-27b", messages=messages, max_completion_tokens=16384,
                                           extra_body={"chat_template_kwargs": {"enable_thinking": True}})
        from types import SimpleNamespace
        return SimpleNamespace(content=r.choices[0].message.content, prompt_tokens=r.usage.prompt_tokens,
                               completion_tokens=r.usage.completion_tokens)


def select(slug, deciders, workers, llm_url):
    from src.s1clin.claim import route
    from src.s1clin.pb_decider import LLMSelectDecider
    od = OUT / slug
    items = [json.loads(l) for l in (od / "items.jsonl").read_text().splitlines()]
    out_path = od / "selections.jsonl"
    done = set()
    if out_path.exists():
        done = {(x["id"], x["decider"], x["version"]) for x in map(json.loads, out_path.read_text().splitlines())}
    states = {}

    def state(it):
        key = (it["subject"], it["hadm"])
        if key not in states:
            states[key] = replay(TREE / it["subject"] / it["hadm"] / "runs" / slug / "decisions.messages.jsonl")
        return states[key][it["msg_epoch"]][:it["n_messages"]]

    jobs = []
    for it in items:
        for d in deciders:
            for v in (("raw", "canon", "inj") if d != "llm_select" else ("canon", "inj")):
                if len(it[v]["options"]) > 1 and (it["id"], d, v) not in done:
                    jobs.append((it, d, v))
    print(f"{len(jobs)} selections to run", flush=True)
    clef = ClefDecider(question=CE_DECISION_QUESTION)
    lock = __import__("threading").Lock()

    def run(job):
        it, d, v = job
        opts = it[v]["options"]
        seed = hashlib.sha256(f"{it['id']}|{v}".encode()).hexdigest()  # same letters for every decider
        try:
            if d == "clef":
                pick, dist, info = clef.choose(state(it), opts, seed)
            elif d == "clef_blind":
                pick, dist, info = clef.choose([{"role": "user", "content": BLIND}], opts, seed)
            else:
                dec = LLMSelectDecider(_Chat(route(llm_url, it["id"])), None)
                pick, dist, info = dec.choose(state(it), opts, seed)
        except Exception as e:
            pick, dist, info = None, None, {"error": repr(e)[:300]}
        rec = {"id": it["id"], "decider": d, "version": v, "pick": pick, "distribution": dist,
               "info": {k: info.get(k) for k in ("error", "parsed", "completion_tokens", "input_tokens") if k in info}}
        with lock, open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, jobs))
    os.chmod(out_path, 0o600)


# ---------------------------------------------------------------- report (aggregates only)

def auroc(pos, neg):
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def ece(conf, correct, bins=10):
    if not conf:
        return None
    tot, err = len(conf), 0.0
    for b in range(bins):
        sel = [i for i, c in enumerate(conf) if b / bins <= c < (b + 1) / bins or (b == bins - 1 and c == 1.0)]
        if sel:
            err += len(sel) / tot * abs(sum(conf[i] for i in sel) / len(sel) - sum(correct[i] for i in sel) / len(sel))
    return round(err, 4)


def report(slug, out):
    od = OUT / slug
    items = {it["id"]: it for it in map(json.loads, (od / "items.jsonl").read_text().splitlines())}
    sel = defaultdict(dict)
    for x in map(json.loads, (od / "selections.jsonl").read_text().splitlines()):
        if x["pick"] is not None:
            sel[(x["decider"], x["version"])][x["id"]] = x
    res = {"slug": slug, "items": len(items), "by_type": dict(Counter(it["type"] for it in items.values())),
           "tau": TAU, "versions": {}}
    for v in ("raw", "canon", "inj"):
        multi = [it for it in items.values() if len(it[v]["options"]) > 1]
        ok = lambda it, i: it[v]["scores"][i] >= TAU
        r = {"items_with_choice": len(multi),
             "oracle_any_correct": round(sum(any(ok(it, i) for i in range(len(it[v]["options"]))) for it in multi)
                                         / max(len(multi), 1), 4),
             "random_expected": round(sum(sum(ok(it, i) for i in range(len(it[v]["options"])))
                                          / len(it[v]["options"]) for it in multi) / max(len(multi), 1), 4),
             "mean_options": round(sum(len(it[v]["options"]) for it in multi) / max(len(multi), 1), 2),
             "deciders": {}}
        if v == "raw":
            for name, fn in (("first", lambda it: 0),
                             ("majority", lambda it: max(range(len(it[v]["counts"])), key=lambda i: (
                                 it[v]["counts"][i], -i)))):
                r["deciders"][name] = {"n": len(multi), "accuracy": round(sum(ok(it, fn(it)) for it in multi)
                                                                          / max(len(multi), 1), 4)}
        for (d, vv), picks in sorted(sel.items()):
            if vv != v:
                continue
            its = [it for it in multi if it["id"] in picks]
            acc = [ok(it, picks[it["id"]]["pick"]) for it in its]
            dd = {"n": len(its), "accuracy": round(sum(acc) / max(len(its), 1), 4),
                  "mean_pick_score": round(sum(it[v]["scores"][picks[it["id"]]["pick"]] for it in its)
                                           / max(len(its), 1), 4)}
            if d.startswith("clef"):
                conf = [max(picks[it["id"]]["distribution"]) for it in its]
                pos = [p for it in its for i, p in enumerate(picks[it["id"]]["distribution"]) if ok(it, i)]
                neg = [p for it in its for i, p in enumerate(picks[it["id"]]["distribution"]) if not ok(it, i)]
                anyc = [any(ok(it, i) for i in range(len(it[v]["options"]))) for it in its]
                dd.update({"ece_top1": ece(conf, [int(a) for a in acc]),
                           "brier_top1": round(sum((c - a) ** 2 for c, a in zip(conf, acc)) / max(len(its), 1), 4),
                           "auroc_candidate": auroc(pos, neg),
                           "mean_max_p_when_some_correct": round(sum(c for c, a in zip(conf, anyc) if a)
                                                                 / max(sum(anyc), 1), 4),
                           "mean_max_p_when_none_correct": round(sum(c for c, a in zip(conf, anyc) if not a)
                                                                 / max(len(anyc) - sum(anyc), 1), 4)})
            r["deciders"][d] = dd
        res["versions"][v] = r
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "select", "report"])
    ap.add_argument("--slug", required=True)
    ap.add_argument("--deciders", default="clef,clef_blind,llm_select")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--llm-url", default="http://127.0.0.1:8011/v1*5,http://127.0.0.1:8013/v1*1")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.umask(0o077)
    if a.cmd == "build":
        build(a.slug)
    elif a.cmd == "select":
        select(a.slug, a.deciders.split(","), a.workers, a.llm_url)
    else:
        report(a.slug, a.out or REPO / "results" / f"offline_{a.slug}.json")


if __name__ == "__main__":
    main()
