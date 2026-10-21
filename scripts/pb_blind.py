"""P4 (docs/research_loop.md): does the decider use the patient state? Compares a decider's exam answers with the state
(answers_D) and blind, options only (answers_D_blind, pb_exam.py select --blind), on the questions both answered:
agreement overall, on keyed and unkeyed questions, and by option kind; key errors and type preferences for both, and
the agreement between different deciders with the state as a yardstick. Aggregates only.

  .venv/bin/python scripts/pb_blind.py [--decider pplx12] [--out results/pb_blind.json]
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from pb_combine import argmax, load  # noqa: E402
from pb_exam import OD  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decider", default="pplx12")
    ap.add_argument("--others", default="clef,d3,amos,decider31b")
    ap.add_argument("--out", default=str(REPO / "results/pb_blind.json"))
    a = ap.parse_args()
    d, b = a.decider, a.decider + "_blind"
    others = [o for o in a.others.split(",") if (OD / f"answers_{o}.jsonl").exists()]
    qs = load([d, b] + others)
    pk = {n: {q["id"]: argmax(q["p"][n]) for q in qs} for n in [d, b] + others}
    share = lambda xs, f: round(sum(map(f, xs)) / len(xs), 4) if xs else None  # noqa: E731
    keyed = [q for q in qs if q["bad"]]
    res = {"decider": d, "questions": len(qs), "keyed": len(keyed),
           "agreement_state_vs_blind": {
               "all": share(qs, lambda q: pk[d][q["id"]] == pk[b][q["id"]]),
               "keyed": share(keyed, lambda q: pk[d][q["id"]] == pk[b][q["id"]]),
               "unkeyed": share([q for q in qs if not q["bad"]], lambda q: pk[d][q["id"]] == pk[b][q["id"]]),
               "same_kind_pick": share(qs, lambda q: q["kinds"][pk[d][q["id"]]] == q["kinds"][pk[b][q["id"]]])},
           "agreement_with_other_deciders_with_state": {o: share(qs, lambda q, o=o: pk[d][q["id"]] == pk[o][q["id"]])
                                                        for o in others},
           "random_agreement": share(qs, lambda q: 1 / len(q["options"])),
           "agree_first_sample": {n: share(qs, lambda q, n=n: pk[n][q["id"]] == q["first"]) for n in (d, b)}}
    for n in (d, b, "first"):
        f = (lambda q: q["first"]) if n == "first" else (lambda q, n=n: pk[n][q["id"]])
        res.setdefault("key_error", {})[n] = {
            key: share([q for q in qs if key in q["bad"]], lambda q, key=key: q["bad"][key][f(q)])
            for key in ("order", "finish", "redundant")}
        first_kind = Counter(q["kinds"][q["first"]] for q in qs)
        res.setdefault("kind_preference", {})[n] = {
            k: round(sum(q["kinds"][f(q)] == k for q in qs) / c, 3) for k, c in first_kind.items()}
    res["key_error"]["random_option"] = {
        key: share([q for q in qs if key in q["bad"]], lambda q, key=key: sum(q["bad"][key]) / len(q["bad"][key]))
        for key in ("order", "finish", "redundant")}
    # yardsticks for same_kind_pick: a random pick, and other deciders with the state
    res["same_kind_baseline"] = {
        "random": share(qs, lambda q: sum(q["kinds"][pk[d][q["id"]]] == k for k in q["kinds"]) / len(q["kinds"])),
        **{o: share(qs, lambda q, o=o: q["kinds"][pk[d][q["id"]]] == q["kinds"][pk[o][q["id"]]]) for o in others}}
    # blind minus state, per key: questions, tasks, difference with a task-bootstrap 95% CI
    rng = random.Random(0)
    for key in ("order", "finish", "redundant"):
        by_task = defaultdict(list)
        for q in qs:
            if key in q["bad"]:
                by_task[q["task"]].append(q["bad"][key][pk[b][q["id"]]] - q["bad"][key][pk[d][q["id"]]])
        tasks = list(by_task)
        diff = lambda ts: sum(sum(by_task[t]) for t in ts) / sum(len(by_task[t]) for t in ts)  # noqa: E731
        boot = sorted(diff(rng.choices(tasks, k=len(tasks))) for _ in range(2000)) if tasks else []
        res.setdefault("blind_minus_state", {})[key] = {
            "questions": sum(map(len, by_task.values())), "tasks": len(tasks),
            "difference": round(diff(tasks), 4) if tasks else None,
            "ci95": [round(boot[50], 4), round(boot[1949], 4)] if boot else None}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
