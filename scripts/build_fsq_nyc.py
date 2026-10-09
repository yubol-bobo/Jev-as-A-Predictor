"""Build frozen fsq_nyc (next-POI) items, SASRec inputs and the popularity baseline.

Source: Foursquare NYC check-ins (Yang et al., TSMC 2014), original zip
  http://www-public.imtbs-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip
extracted under data/external/fsq_nyc/ (file is latin-1 encoded, e.g. "Café").

Protocol (see results/data_cards/fsq_nyc.md):
  * iterative 10-core filtering on venues and users;
  * per user, check-ins sorted by local time (UTC + timezone offset), stable on file order;
  * leave-one-out: test target = last check-in, dev target = second-to-last;
    NYC has 1,083 users (< 1,300), so every user is a test item and dev uses 300 sampled users;
  * train targets: positions <= len-3 with >= 3 earlier check-ins, 10,000 sampled round-robin
    across users (seeded), stored in seeded random order (first N lines = nested subsets);
  * candidates: gold + 19 venues drawn uniformly from all filtered venues except the gold
    (previously visited venues are NOT excluded: revisits are common in POI data);
  * state: the 20 most recent check-ins before the target, oldest to newest, local weekday/time;
    the time of the target check-in is not shown.

  python scripts/build_fsq_nyc.py
"""

import gzip
import io
import json
import math
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from seqpred import items as I, metrics  # noqa: E402

TASK = "fsq_nyc"
SEED = 20261009
HIST = 20
N_NEG = 19
N_TEST, N_DEV, N_TRAIN = 1000, 300, 10000
MIN_COUNT = 10
RAW_ZIP = I.EXTERNAL / TASK / "dataset_tsmc2014.zip"
RAW_MEMBER = "dataset_tsmc2014/dataset_TSMC2014_NYC.txt"
INSTRUCTIONS = "Which venue will the user check in at next?"
COLS = ["user", "venue", "cat_id", "cat", "lat", "lon", "tz", "utc"]


def rng(*parts):
    return random.Random("-".join(map(str, (SEED, TASK) + parts)))


def load():
    with zipfile.ZipFile(RAW_ZIP) as z:
        raw = z.read(RAW_MEMBER)
    df = pd.read_csv(io.BytesIO(raw), sep="\t", header=None, names=COLS, encoding="latin-1",
                     dtype={"user": int, "venue": str, "cat_id": str, "cat": str, "lat": float, "lon": float,
                            "tz": int, "utc": str}, quoting=3)
    df["row"] = np.arange(len(df))
    utc = pd.to_datetime(df["utc"], format="%a %b %d %H:%M:%S %z %Y", utc=True)
    df["local"] = utc.dt.tz_localize(None) + pd.to_timedelta(df["tz"], unit="m")
    return df


def kcore(df):
    while True:
        n = len(df)
        df = df[df.venue.map(df.venue.value_counts()) >= MIN_COUNT]
        df = df[df.user.map(df.user.value_counts()) >= MIN_COUNT]
        if len(df) == n:
            return df


def haversine_km(a, b):
    (la1, lo1), (la2, lo2) = a, b
    p1, p2 = math.radians(la1), math.radians(la2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(min(1.0, h)))


def fmt_km(d):
    return f"{d:.2f} km" if d < 1 else f"{d:.1f} km"


def main():
    df = kcore(load())
    # venue attributes: modal category (ties -> earliest check-in), median coordinates
    venues = sorted(df.venue.unique())  # raw Foursquare ids are hex hashes -> index order is uninformative
    vidx = {v: i + 1 for i, v in enumerate(venues)}
    first = df.sort_values("row")
    cat = {}
    for v, g in first.groupby("venue", sort=False):
        c = Counter(g.cat)
        best = max(c.values())
        cat[v] = next(x for x in g.cat if c[x] == best)
    coord = {vidx[v]: (g.lat.median(), g.lon.median()) for v, g in df.groupby("venue")}
    n_items = len(venues)
    vname = {vidx[v]: f"V{vidx[v]:04d} · {cat[v]}" for v in venues}
    vcat = {vidx[v]: cat[v] for v in venues}
    assert len(set(vname.values())) == n_items

    df = df.sort_values(["user", "local", "row"], kind="stable")
    seqs = {}  # user -> list of (venue index, local timestamp)
    for u, g in df.groupby("user", sort=True):
        seqs[int(u)] = list(zip(g.venue.map(vidx).tolist(), g.local.tolist()))
    users = sorted(seqs)

    def build(u, t, split, item_id):
        seq = seqs[u]
        prior = seq[:t]
        shown = prior[-HIST:]
        gold_v = seq[t][0]
        r = rng("cands", item_id)
        negs = set()
        while len(negs) < N_NEG:
            c = r.randint(1, n_items)
            if c != gold_v:
                negs.add(c)
        opt_ids = sorted(negs | {gold_v})
        omitted = len(prior) - len(shown)
        lines = ["Check-ins of this user in New York City, oldest to newest (local time):"]
        if omitted:
            lines.append(f"(earlier check-ins omitted: {omitted})")
        for k, (v, ts) in enumerate(shown, 1):
            lines.append(f"{k}. {ts.strftime('%a %H:%M')} — {vcat[v]} (venue V{v:04d})")
        state = "\n".join(lines)
        visits = Counter(v for v, _ in prior)  # full prior history before the target
        last = coord[shown[-1][0]]
        desc = {}
        for v in opt_ids:
            k = visits[v]
            desc[vname[v]] = (f"{vcat[v]}; {fmt_km(haversine_km(last, coord[v]))} from the last check-in; "
                              f"visited {k} time{'s' if k != 1 else ''} before by this user")
        return I.make_item(
            id=item_id, task=TASK, split=split, group=u, state=state, instructions=INSTRUCTIONS,
            options=[vname[v] for v in opt_ids], gold=opt_ids.index(gold_v), descriptions=desc,
            meta={"option_ids": opt_ids, "history_ids": [v for v, _ in shown], "user": u,
                  "target_pos": t, "n_prior": len(prior), "gold_prior_visits": visits[gold_v],
                  "option_raw": [venues[v - 1] for v in opt_ids]})

    # test: all users (1,083 < 1,300); dev: 300 sampled users' second-to-last check-in
    test_users = users if len(users) < 1300 else sorted(rng("test_users").sample(users, N_TEST))
    dev_users = sorted(rng("dev_users").sample(test_users, N_DEV))
    test = [build(u, len(seqs[u]) - 1, "test", f"{TASK}-test-u{u}") for u in test_users]
    dev = [build(u, len(seqs[u]) - 2, "dev", f"{TASK}-dev-u{u}") for u in dev_users]

    # train: positions 3..len-3, round-robin over users in seeded order until 10,000
    pools = {u: list(range(3, len(seqs[u]) - 2)) for u in users}
    for u in users:
        rng("train_pos", u).shuffle(pools[u])
    order = list(users)
    rng("train_users").shuffle(order)
    picked, rnd = [], 0
    while len(picked) < N_TRAIN:
        progressed = False
        for u in order:
            if rnd < len(pools[u]) and len(picked) < N_TRAIN:
                picked.append((u, pools[u][rnd]))
                progressed = True
        rnd += 1
        if not progressed:
            break
    rng("train_order").shuffle(picked)
    train = [build(u, t, "train", f"{TASK}-train-u{u}-t{t}") for u, t in picked]

    for split, its in (("test", test), ("dev", dev), ("train", train)):
        I.write_items(TASK, split, its)

    base = I.ITEMS / TASK
    train_seqs = [[v for v, _ in seqs[u][:-2]] for u in users]
    with gzip.GzipFile(base / "train_sequences.jsonl.gz", "wb", mtime=0) as f:  # byte-reproducible
        f.write("".join(json.dumps(s) + "\n" for s in train_seqs).encode())
    (base / "vocab.json").write_text(json.dumps(
        {"n_items": n_items, "index_to_raw": {str(i): venues[i - 1] for i in range(1, n_items + 1)},
         "index_to_name": {str(i): vname[i] for i in range(1, n_items + 1)},
         "index_to_category": {str(i): vcat[i] for i in range(1, n_items + 1)},
         "note": "item index = rank of the raw Foursquare venue id in sorted order; 0 is padding"}, ensure_ascii=False))

    # popularity baseline: training check-in counts (+1), normalized over the candidates
    cnt = Counter(v for s in train_seqs for v in s)
    report = {}
    for split, its in (("dev", dev), ("test", test)):
        rows = []
        for it in its:
            w = np.array([cnt[v] + 1 for v in it["meta"]["option_ids"]], dtype=float)
            rows.append({"id": it["id"], "p": (w / w.sum()).round(6).tolist()})
        I.write_predictions(TASK, "popularity", split, rows)
        s = metrics.summarize(its, I.load_predictions(TASK, "popularity", split))
        report[split] = {k: (v if not isinstance(v, tuple) else [round(x, 4) for x in v]) for k, v in s.items()}
        print("popularity", split, {k: report[split][k] for k in ("hr1", "ndcg10", "mrr", "nll", "ece")})
    (I.PREDS / TASK / "popularity" / "report.json").write_text(json.dumps(report, indent=1))

    # stats for the data card
    stats = {"checkins": int(len(df)), "users": len(users), "venues": n_items,
             "categories": len(set(vcat.values())), "seq_len": pd.Series([len(s) for s in seqs.values()]).describe().round(1).to_dict(),
             "train_seq_tokens": sum(map(len, train_seqs)),
             "multi_category_venues": int(sum(1 for v, g in first.groupby("venue") if g.cat.nunique() > 1)),
             "train_users_covered": len({u for u, _ in picked})}
    for split, its in (("test", test), ("dev", dev), ("train", train)):
        L = [len(it["request"]["state"]) for it in its]
        gold_visited = np.mean([it["meta"]["gold_prior_visits"] > 0 for it in its])
        stats[split] = {"n": len(its), "state_chars_mean": round(float(np.mean(L)), 1), "state_chars_max": int(max(L)),
                        "hist_len_mean": round(float(np.mean([len(it["meta"]["history_ids"]) for it in its])), 2),
                        "gold_previously_visited": round(float(gold_visited), 4),
                        "gold_in_shown_history": round(float(np.mean([it["meta"]["option_ids"][it["gold"]] in it["meta"]["history_ids"] for it in its])), 4)}
    (base / "stats.json").write_text(json.dumps(stats, indent=1, default=str))
    print(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
