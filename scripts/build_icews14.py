"""Build frozen ICEWS14 event-forecasting items: (s, r, ?, t) -> object entity.

  python scripts/build_icews14.py

Queries are object-prediction facts sampled (seeded) from test.txt (1,000), valid.txt (300 dev)
and train.txt (10,000, stored in seeded random order so the first N are nested subsets), with at
most 3 queries per (subject, day). History = every fact with day < t from train+valid+test
(ground-truth history, extrapolation setting). Candidates = gold + 19 entities sampled uniformly
from all entities, excluding every o' with (s, r, o', t) true in any split (time-aware filter).
Writes data/items/icews14/{train,dev,test}.jsonl and vocab.json.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from seqpred import items as I  # noqa: E402
from seqpred.icews14 import ICEWS14  # noqa: E402

TASK = "icews14"
SEED = 20261009
SIZES = {"test": ("test", 1000), "dev": ("valid", 300), "train": ("train", max(I.TRAIN_SIZES))}
MAX_PER_SUBJECT_DAY = 3
N_NEG = 19
N_RECENT = 20
N_SAME_SR = 10


def sample_queries(ds, src_split, n, rng):
    picked, per = [], Counter()
    for i in rng.permutation(ds.split_idx(src_split)):
        key = (ds.s[i], ds.day[i])
        if per[key] < MAX_PER_SUBJECT_DAY:
            per[key] += 1
            picked.append(int(i))
            if len(picked) == n:
                return picked
    raise RuntimeError(f"only {len(picked)} queries available in {src_split}")


def history(ds, s, r, day):
    """Shown history: (earlier same-(s, r) facts not among the recent ones, recent facts involving s)."""
    recent = ds.before(ds.by_ent.get(s), day)[-N_RECENT:]
    shown = set(recent.tolist())
    same = [i for i in ds.before(ds.by_sr.get((s, r)), day).tolist() if i not in shown][-N_SAME_SR:]
    return list(same), recent.tolist()


def render_state(ds, s, r, day, same, recent):
    S, R = ds.ent[s], ds.rel[r]
    while True:
        lines = []
        if same:
            lines.append(f"Earlier events with the same subject and relation ({S} — {R}), oldest to newest:")
            lines += [ds.fact_text(i) for i in same]
            lines.append("")
        if recent:
            lines.append(f"Most recent events involving {S}, oldest to newest:")
            lines += [ds.fact_text(i) for i in recent]
        else:
            lines.append(f"No earlier events involving {S} are recorded.")
        lines += ["", f"Query date: {ds.date(day)}."]
        state = "\n".join(lines)
        if len(state) <= I.MAX_STATE_CHARS:
            return state, same, recent
        if same:  # drop oldest facts, same-(s, r) section first
            same = same[1:]
        else:
            recent = recent[1:]


def build_split(ds, split, rng):
    src, n = SIZES[split]
    out = []
    for i in sample_queries(ds, src, n, rng):
        s, r, o, day = (int(x) for x in ds.facts[i])
        excluded = ds.answers[(s, r, day)]
        negs = []
        while len(negs) < N_NEG:
            e = int(rng.integers(ds.n_ent))
            if e not in excluded and e not in negs:
                negs.append(e)
        cand = [o] + negs
        rng.shuffle(cand)
        same, recent = history(ds, s, r, day)
        state, same, recent = render_state(ds, s, r, day, same, recent)
        shown = same + recent  # chronological: same-(s, r) facts are older than the recent window
        date = ds.date(day)
        out.append(I.make_item(
            id=f"{TASK}-{split}-{i}", task=TASK, split=split, group=s,
            state=state,
            instructions=f"Which entity is the most likely object of this event: "
                         f"{ds.ent[s]} — {ds.rel[r]} — ? on {date}?",
            options=[ds.ent[e] for e in cand], gold=cand.index(o),
            meta={"option_ids": [e + 1 for e in cand],
                  "history_ids": [int(ds.o[j]) + 1 for j in shown],
                  "query": {"s": s, "r": r, "t": day * 24, "day": day, "date": date},
                  "fact_index": i, "n_history_shown": len(shown), "n_same_sr_shown": len(same)}))
    return out


def main():
    ds = ICEWS14()
    stats = {}
    for k, split in enumerate(("test", "dev", "train")):
        rng = np.random.default_rng([SEED, k])
        its = build_split(ds, split, rng)
        I.write_items(TASK, split, its)
        L = np.array([len(it["request"]["state"]) for it in its])
        H = np.array([it["meta"]["n_history_shown"] for it in its])
        stats[split] = {"n": len(its), "subjects": len({it["group"] for it in its}),
                        "days": len({it["meta"]["query"]["day"] for it in its}),
                        "state_chars": {"mean": round(float(L.mean()), 1), "median": float(np.median(L)),
                                        "p95": float(np.percentile(L, 95)), "max": int(L.max())},
                        "facts_shown_mean": round(float(H.mean()), 2),
                        "no_history": int((H == 0).sum()),
                        "with_same_sr_section": int(sum(it["meta"]["n_same_sr_shown"] > 0 for it in its)),
                        "gold_in_shown_history": round(float(np.mean(
                            [it["meta"]["option_ids"][it["gold"]] in it["meta"]["history_ids"] for it in its])), 3)}
        print(split, json.dumps(stats[split]))
    vocab = {"n_items": ds.n_ent, "index_to_name": {str(e + 1): n for e, n in enumerate(ds.ent)},
             "n_relations": ds.n_rel, "relation_names": ds.rel,
             "note": "item index = ICEWS14 entity id + 1 (entity2id.txt); 0 is unused"}
    (I.ITEMS / TASK / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=0))
    (I.ITEMS / TASK / "build_stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
