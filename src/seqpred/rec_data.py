"""Shared builder for the leave-one-out sequential-recommendation tasks (ml1m, beauty).

Protocol (identical for every rec task):
  * per user, interactions sorted by timestamp (stable: ties keep original file order)
  * test target = last item, dev target = second-to-last; 1,000 test users and 300 dev
    users, disjoint, sampled with the base seed
  * train items: targets at 0-based positions 3..len-3 of any user (never a dev/test
    target, >= 3 history items); 10,000 sampled with at most 3 per user, stored in the
    seeded sampling order so the first N lines are nested subsets
  * state = the most recent <= 20 history items, oldest to newest
  * candidates = gold + 19 negatives drawn uniformly from items the user never interacted
    with (seeded per item); canonical option order is a seeded shuffle
  * item indices are 1..n_items (raw ids sorted); train_sequences = every user's index
    sequence without its last two items
"""

import gzip
import json
import random
from collections import Counter

import numpy as np

from . import items as I

SEED = 20261009
N_TEST, N_DEV, N_TRAIN = 1000, 300, 10000
TRAIN_PER_USER = 3
MIN_TRAIN_HISTORY = 3
HISTORY = 20
N_NEG = 19
MAX_NAME = 100


def clip_name(s, n=MAX_NAME):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 3].rstrip() + "..."


def user_sequences(records):
    """records: iterable of (user, item, timestamp) in original file order.
    Returns {user: [item, ...]} sorted by timestamp, ties in file order (stable sort)."""
    per = {}
    for k, (u, it, ts) in enumerate(records):
        per.setdefault(u, []).append((ts, k, it))
    return {u: [it for _, _, it in sorted(v)] for u, v in per.items()}


def build(task, seqs, names, descs, *, header, instructions, user_key=str, item_key=str, seed=SEED,
          log=print):
    """seqs: {raw user: [raw item, ...]} chronological. names/descs: {raw item: str / None}.
    Writes items, vocab.json and train_sequences.jsonl.gz; returns stats for the data card."""
    raw_items = sorted({i for s in seqs.values() for i in s}, key=item_key)
    idx = {r: k + 1 for k, r in enumerate(raw_items)}
    n_items = len(raw_items)
    name_of = {idx[r]: clip_name(names[r]) for r in raw_items}
    desc_of = {idx[r]: descs.get(r) for r in raw_items}
    users = sorted(seqs, key=user_key)
    iseq = {u: [idx[r] for r in seqs[u]] for u in users}
    seen = {u: set(s) for u, s in iseq.items()}

    def make(split, u, pos):
        """Item predicting iseq[u][pos] from the history before it."""
        s = iseq[u]
        hist = s[:pos][-HISTORY:]
        gold_idx = s[pos]
        item_id = f"{task}-{split}-u{u}" if split != "train" else f"{task}-train-u{u}-p{pos}"
        rng = random.Random(f"{seed}|{item_id}")
        negs, chosen = [], {gold_idx}
        while len(negs) < N_NEG:
            c = rng.randint(1, n_items)
            if c not in seen[u] and c not in chosen:
                chosen.add(c)
                negs.append(c)
        opt_ids = [gold_idx] + negs
        rng.shuffle(opt_ids)
        gold = opt_ids.index(gold_idx)
        opts = [name_of[c] for c in opt_ids]
        cnt = Counter(opts)
        opts = [o if cnt[o] == 1 else f"{o} (#{c})" for o, c in zip(opts, opt_ids)]
        state = header + "\n" + "\n".join(f"{k}. {name_of[h]}" for k, h in enumerate(hist, 1))
        return I.make_item(
            id=item_id, task=task, split=split, group=u, state=state, instructions=instructions,
            options=opts, gold=gold, descriptions={o: desc_of[c] for o, c in zip(opts, opt_ids)},
            meta={"user": u, "target_pos": pos, "seq_len": len(s),
                  "option_ids": opt_ids, "history_ids": hist,
                  "option_raw_ids": [raw_items[c - 1] for c in opt_ids],
                  "history_raw_ids": [raw_items[h - 1] for h in hist]})

    rng = random.Random(seed)
    eligible = [u for u in users if len(iseq[u]) >= 3]  # dev target needs >= 1 history item
    picked = rng.sample(eligible, N_TEST + N_DEV)
    test_users, dev_users = sorted(picked[:N_TEST], key=user_key), sorted(picked[N_TEST:], key=user_key)
    test = [make("test", u, len(iseq[u]) - 1) for u in test_users]
    dev = [make("dev", u, len(iseq[u]) - 2) for u in dev_users]

    pool = [(u, p) for u in users for p in range(MIN_TRAIN_HISTORY, len(iseq[u]) - 2)]
    random.Random(seed + 1).shuffle(pool)
    taken, train_keys = Counter(), []
    for u, p in pool:
        if taken[u] < TRAIN_PER_USER:
            taken[u] += 1
            train_keys.append((u, p))
            if len(train_keys) == N_TRAIN:
                break
    assert len(train_keys) == N_TRAIN, f"only {len(train_keys)} train items available"
    train = [make("train", u, p) for u, p in train_keys]

    for split, its in (("test", test), ("dev", dev), ("train", train)):
        log(f"wrote {I.write_items(task, split, its)} ({len(its)})")
    base = I.ITEMS / task
    (base / "vocab.json").write_text(json.dumps(
        {"n_items": n_items, "index_to_raw": {str(k): raw_items[k - 1] for k in range(1, n_items + 1)},
         "index_to_name": {str(k): name_of[k] for k in range(1, n_items + 1)}}, ensure_ascii=False))
    with gzip.open(base / "train_sequences.jsonl.gz", "wt") as f:
        for u in users:
            f.write(json.dumps(iseq[u][:-2]) + "\n")

    def state_stats(its):
        L = np.array([len(it["request"]["state"]) for it in its])
        H = np.array([len(it["meta"]["history_ids"]) for it in its])
        return {"n": len(its), "state_chars_mean": round(float(L.mean()), 1), "state_chars_max": int(L.max()),
                "history_items_mean": round(float(H.mean()), 2), "history_items_min": int(H.min()),
                "desc_null_frac": round(float(np.mean([d is None for it in its for d in
                                                       it["request"]["questions"]["next"]["criteria"].values()])), 4),
                "disambiguated_items": sum(any(o != name_of[c] for o, c in zip(it["options"], it["meta"]["option_ids"]))
                                          for it in its)}

    lens = np.array([len(s) for s in iseq.values()])
    return {"n_users": len(users), "n_items": n_items, "n_interactions": int(lens.sum()),
            "seq_len_mean": round(float(lens.mean()), 2), "seq_len_min": int(lens.min()),
            "seq_len_max": int(lens.max()), "n_train_pool": len(pool),
            "n_train_users": len(taken), "train_seq_interactions": int(sum(max(len(s) - 2, 0) for s in iseq.values())),
            "splits": {"test": state_stats(test), "dev": state_stats(dev), "train": state_stats(train)},
            "example_state": test[0]["request"]["state"][:600]}


def write_card(task, title, sources, stats, preprocessing, deviations):
    """results/data_cards/<task>.md"""
    s = stats
    lines = [f"# Data card — {task} ({title})", "", "## Sources", "",
             "| file | URL | sha256 |", "|---|---|---|"]
    lines += [f"| `{f}` | {u} | `{h}` |" for f, u, h in sources]
    lines += ["", "## Preprocessing", ""] + [f"- {p}" for p in preprocessing]
    lines += ["", "## Counts", "",
              f"- users: {s['n_users']:,}; items: {s['n_items']:,}; interactions: {s['n_interactions']:,}",
              f"- sequence length: mean {s['seq_len_mean']}, min {s['seq_len_min']}, max {s['seq_len_max']}",
              f"- SASRec train_sequences: {s['n_users']:,} sequences, {s['train_seq_interactions']:,} interactions "
              "(last two items of every user removed)",
              f"- train-item pool (eligible targets): {s['n_train_pool']:,}; sampled 10,000 from "
              f"{s['n_train_users']:,} users (≤3 per user)", "",
              "| split | items | state chars mean | state chars max | history items mean | history items min "
              "| null descriptions | items with disambiguated names |", "|---|---|---|---|---|---|---|---|"]
    for sp, v in s["splits"].items():
        lines.append(f"| {sp} | {v['n']:,} | {v['state_chars_mean']} | {v['state_chars_max']} | "
                     f"{v['history_items_mean']} | {v['history_items_min']} | {v['desc_null_frac']:.2%} | "
                     f"{v['disambiguated_items']} |")
    lines += ["", "## Example state (test item 1, truncated)", "", "```", s["example_state"], "```", "",
              "## Deviations / notes", ""] + [f"- {d}" for d in deviations] + [""]
    path = I.ROOT / "results" / "data_cards" / f"{task}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path
