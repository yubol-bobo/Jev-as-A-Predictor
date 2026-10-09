"""Build frozen bpic12 (next-activity) items, SASRec inputs and the majority / n-gram baselines.

Source: BPI Challenge 2012 event log, 4TU.ResearchData, DOI 10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f
  (article 12689204, file BPI_Challenge_2012.xes.gz, md5 74c7ba9aba85bfcb181a22c9d565e5b5)
saved to data/external/bpic12/BPI_Challenge_2012.xes.gz and parsed with xml.etree (data only).

Protocol (see results/data_cards/bpic12.md):
  * only lifecycle:transition == COMPLETE events; events in a case sorted by time (stable on file order);
  * cases sorted by first-event time; first 80% = train part, last 20% = test part;
    the last 10% (by time) of the train part = dev cases; the rest = train cases;
  * items are prefixes: given events 1..k (k >= 1) predict event k+1, or [END] if k = case length;
  * candidates = every activity seen in train-part cases + [END] (no sampling);
  * prefixes are sampled with proportional allocation over prefix-length strata (so the prefix-length
    distribution matches the population of all prefixes), at most 3 prefixes per case;
    test 1,000 (test cases), dev 300 (dev cases), train 10,000 (train cases, seeded random order).

  python scripts/build_bpic12.py
"""

import gzip
import json
import math
import random
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from seqpred import items as I, metrics  # noqa: E402

TASK = "bpic12"
SEED = 20261009
HIST = 20
N_TEST, N_DEV, N_TRAIN = 1000, 300, 10000
MAX_PER_CASE = 3
END = "[END]"
RAW = I.EXTERNAL / TASK / "BPI_Challenge_2012.xes.gz"
INSTRUCTIONS = "Which activity will happen next in this case? Choose [END] if the case is finished."
NGRAM_ORDER = 4
ALPHAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def rng(*parts):
    return random.Random("-".join(map(str, (SEED, TASK) + parts)))


def local(tag):
    return tag.rsplit("}", 1)[-1]


def parse_xes(path):
    """Return {case_id: [(timestamp, file_order, activity)]} for COMPLETE events."""
    cases, order = {}, 0
    with gzip.open(path, "rb") as f:
        for _, el in ET.iterparse(f, events=("end",)):
            if local(el.tag) != "trace":
                continue
            cid, evs = None, []
            for ch in el:
                t = local(ch.tag)
                if t == "string" and ch.get("key") == "concept:name":
                    cid = ch.get("value")
                elif t == "event":
                    a = {c.get("key"): c.get("value") for c in ch}
                    if a.get("lifecycle:transition", "").upper() == "COMPLETE":
                        evs.append((datetime.fromisoformat(a["time:timestamp"]), order, a["concept:name"]))
                    order += 1
            assert cid is not None and cid not in cases
            if evs:
                cases[cid] = sorted(evs, key=lambda e: (e[0], e[1]))
            el.clear()
    return cases


def fmt_elapsed(sec):
    m = int(sec // 60)
    d, h, mm = m // 1440, (m % 1440) // 60, m % 60
    if d:
        return f"{d}d {h}h"
    if h:
        return f"{h}h {mm}m"
    return f"{mm}m"


def stratum(k):
    for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 10), (11, 12), (13, 15),
                   (16, 20), (21, 25), (26, 30), (31, 40), (41, 60)):
        if lo <= k <= hi:
            return f"{lo}-{hi}"
    return "61+"


def sample_prefixes(case_ids, lengths, n, tag):
    """Proportional allocation over prefix-length strata, <= MAX_PER_CASE prefixes per case."""
    pop = defaultdict(list)
    for c in case_ids:
        for k in range(1, lengths[c] + 1):
            pop[stratum(k)].append((c, k))
    total = sum(map(len, pop.values()))
    keys = sorted(pop)
    raw = {s: n * len(pop[s]) / total for s in keys}
    quota = {s: int(math.floor(raw[s])) for s in keys}
    for s in sorted(keys, key=lambda s: (-(raw[s] - quota[s]), s))[:n - sum(quota.values())]:
        quota[s] += 1
    for s in keys:
        rng("strata", tag, s).shuffle(pop[s])
    per_case, picked, ptr = Counter(), [], {s: 0 for s in keys}
    # interleave strata so that the per-case cap does not systematically favour one stratum
    active = [s for s in keys if quota[s] > 0]
    got = Counter()
    while active:
        nxt = []
        for s in active:
            while ptr[s] < len(pop[s]):
                c, k = pop[s][ptr[s]]
                ptr[s] += 1
                if per_case[c] < MAX_PER_CASE:
                    per_case[c] += 1
                    picked.append((c, k))
                    got[s] += 1
                    break
            if got[s] < quota[s] and ptr[s] < len(pop[s]):
                nxt.append(s)
        active = nxt
    assert len(picked) == n, f"{tag}: sampled {len(picked)} of {n} (quota {quota}, got {got})"
    return picked, {s: (len(pop[s]), quota[s]) for s in keys}


def main():
    cases = parse_xes(RAW)
    start = {c: evs[0][0] for c, evs in cases.items()}
    ordered = sorted(cases, key=lambda c: (start[c], int(c) if c.isdigit() else c))
    n_train_part = int(0.8 * len(ordered))
    train_part, test_cases = ordered[:n_train_part], ordered[n_train_part:]
    n_fit = int(0.9 * len(train_part))
    train_cases, dev_cases = train_part[:n_fit], train_part[n_fit:]

    acts = sorted({e[2] for c in train_part for e in cases[c]})
    vocab = acts + [END]
    aidx = {a: i + 1 for i, a in enumerate(vocab)}
    n_items = len(vocab)
    all_ids = list(range(1, n_items + 1))
    unseen = Counter(e[2] for c in test_cases for e in cases[c] if e[2] not in aidx)
    lengths = {c: len(cases[c]) for c in cases}

    def build(c, k, split):
        evs = cases[c]
        prefix = evs[:k]
        gold = END if k == len(evs) else evs[k][2]
        t0 = evs[0][0]
        shown_from = max(0, k - HIST)
        lines = ["Events of this case so far, oldest to newest:"]
        if shown_from:
            lines.append(f"(earlier events omitted: {shown_from})")
        for j in range(shown_from, k):
            ts, _, a = prefix[j]
            lines.append(f"{j + 1}. {a} (t+{fmt_elapsed((ts - t0).total_seconds())})")
        item_id = f"{TASK}-{split}-c{c}-k{k}"
        return I.make_item(
            id=item_id, task=TASK, split=split, group=c, state="\n".join(lines), instructions=INSTRUCTIONS,
            options=vocab, gold=vocab.index(gold), descriptions=None,
            meta={"option_ids": all_ids, "history_ids": [aidx[a] for _, _, a in prefix[shown_from:]],
                  "case": c, "prefix_len": k, "case_len": len(evs),
                  "prefix_ids": [aidx[a] for _, _, a in prefix]})

    out, strata = {}, {}
    for split, cids, n in (("test", test_cases, N_TEST), ("dev", dev_cases, N_DEV), ("train", train_cases, N_TRAIN)):
        # drop prefixes whose gold or history contains an activity unseen in the train part
        ok = [c for c in cids if all(e[2] in aidx for e in cases[c])]
        picked, strata[split] = sample_prefixes(ok, lengths, n, split)
        if split == "train":
            rng("train_order").shuffle(picked)
        else:
            pos = {c: i for i, c in enumerate(ordered)}
            picked.sort(key=lambda ck: (pos[ck[0]], ck[1]))
        out[split] = [build(c, k, split) for c, k in picked]
        I.write_items(TASK, split, out[split])

    base = I.ITEMS / TASK
    train_seqs = [[aidx[e[2]] for e in cases[c]] + [aidx[END]] for c in train_cases]
    with gzip.GzipFile(base / "train_sequences.jsonl.gz", "wb", mtime=0) as f:
        f.write("".join(json.dumps(s) + "\n" for s in train_seqs).encode())
    (base / "vocab.json").write_text(json.dumps(
        {"n_items": n_items, "index_to_raw": {str(i): a for i, a in enumerate(vocab, 1)},
         "index_to_name": {str(i): a for i, a in enumerate(vocab, 1)}, "end_index": aidx[END],
         "note": "activities sorted by name, then [END] as the last index; 0 is padding"}))

    # ---- baselines (fit on train cases only) ----
    BOS = 0
    nxt = Counter()
    ngram = defaultdict(Counter)  # context tuple (len 0..ORDER-1, BOS-padded) -> next-token counts
    for s in train_seqs:
        padded = [BOS] * (NGRAM_ORDER - 1) + s
        for j in range(NGRAM_ORDER - 1, len(padded)):
            y = padded[j]
            nxt[y] += 1
            for n in range(0, NGRAM_ORDER):
                ngram[tuple(padded[j - n:j])][y] += 1

    def majority_p(it):
        w = np.array([nxt[i] + 1 for i in it["meta"]["option_ids"]], dtype=float)
        return w / w.sum()

    def ngram_p(it, alpha):
        padded = [BOS] * (NGRAM_ORDER - 1) + it["meta"]["prefix_ids"]
        for n in range(NGRAM_ORDER - 1, -1, -1):  # back off to the longest seen context
            ctx = tuple(padded[len(padded) - n:]) if n else ()
            if ctx in ngram:
                cnt = ngram[ctx]
                break
        w = np.array([cnt[i] + alpha for i in it["meta"]["option_ids"]], dtype=float)
        return w / w.sum()

    dev_nll = {a: np.mean([metrics.item_metrics(ngram_p(it, a), it["gold"])["nll"] for it in out["dev"]]) for a in ALPHAS}
    alpha = min(ALPHAS, key=lambda a: dev_nll[a])
    print("ngram alpha (dev NLL):", alpha, {a: round(v, 4) for a, v in dev_nll.items()})
    for system, fn in (("majority", majority_p), ("ngram", lambda it: ngram_p(it, alpha))):
        report = {"fit": "train cases (train part minus dev cases)"}
        if system == "ngram":
            report.update(order=NGRAM_ORDER, alpha=alpha, alpha_grid_dev_nll={str(a): round(v, 4) for a, v in dev_nll.items()})
        for split in ("dev", "test"):
            its = out[split]
            I.write_predictions(TASK, system, split, [{"id": it["id"], "p": fn(it).round(6).tolist()} for it in its])
            s = metrics.summarize(its, I.load_predictions(TASK, system, split))
            report[split] = {k: (v if not isinstance(v, tuple) else [round(x, 4) for x in v]) for k, v in s.items()}
            print(system, split, {k: report[split][k] for k in ("hr1", "ndcg10", "mrr", "nll", "ece")})
        (I.PREDS / TASK / system / "report.json").write_text(json.dumps(report, indent=1))

    # ---- stats for the data card ----
    stats = {"cases": len(cases), "complete_events": sum(lengths.values()), "activities": len(acts),
             "activities_all": len({e[2] for evs in cases.values() for e in evs}),
             "unseen_test_activities": dict(unseen),
             "train_part_cases": len(train_part), "train_cases": len(train_cases), "dev_cases": len(dev_cases),
             "test_cases": len(test_cases),
             "time_ranges": {k: [str(start[v[0]]), str(start[v[-1]])] for k, v in
                             (("train", train_cases), ("dev", dev_cases), ("test", test_cases))},
             "case_len": {"mean": round(float(np.mean(list(lengths.values()))), 2), "max": max(lengths.values()),
                          "min": min(lengths.values())},
             "train_seq_tokens": sum(map(len, train_seqs)), "strata_population_quota": strata,
             "majority_next_train": vocab[max(nxt, key=nxt.get) - 1]}
    for split, its in out.items():
        L = [len(it["request"]["state"]) for it in its]
        stats[split] = {"n": len(its), "cases": len({it["group"] for it in its}),
                        "state_chars_mean": round(float(np.mean(L)), 1), "state_chars_max": int(max(L)),
                        "prefix_len_mean": round(float(np.mean([it["meta"]["prefix_len"] for it in its])), 2),
                        "truncated_share": round(float(np.mean([it["meta"]["prefix_len"] > HIST for it in its])), 4),
                        "gold_end_share": round(float(np.mean([it["options"][it["gold"]] == END for it in its])), 4)}
    (base / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "strata_population_quota"}, indent=1))


if __name__ == "__main__":
    main()
