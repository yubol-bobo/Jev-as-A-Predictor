"""Build frozen Amazon Beauty (2014, 5-core) items (see src/seqpred/rec_data.py for the protocol).

  python scripts/build_beauty.py    # expects data/external/beauty/{reviews_Beauty_5,meta_Beauty}.json.gz

The metadata file is one Python literal per line; it is parsed with ast.literal_eval (data only,
never executed). Only lines whose asin occurs in the 5-core reviews are parsed.
"""

import ast
import gzip
import hashlib
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from seqpred import items as I, rec_data  # noqa: E402

BASE_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
FILES = ("reviews_Beauty_5.json.gz", "meta_Beauty.json.gz")
TASK = "beauty"
ASIN_RE = re.compile(r"""['"]asin['"]:\s*['"]([^'"]+)['"]""")


def clean(s):
    return " ".join(html.unescape(str(s)).split()) if s else ""


def describe(meta):
    paths = [p for p in meta.get("categories") or [] if p]
    path = next((p for p in paths if p[0] == "Beauty"), paths[0] if paths else [])
    path = [clean(x) for x in path if clean(x)]
    if len(path) > 1 and path[0] == "Beauty":
        path = path[1:]
    cat = " > ".join(path)
    brand = clean(meta.get("brand"))
    s = cat + (f" (brand: {brand})" if brand and cat else f"Brand: {brand}" if brand else "")
    return rec_data.clip_name(s, 150) if s else None


def main():
    ext = I.EXTERNAL / TASK
    records = []
    with gzip.open(ext / FILES[0], "rt") as f:
        for line in f:
            r = json.loads(line)
            records.append((r["reviewerID"], r["asin"], int(r["unixReviewTime"])))
    asins = {a for _, a, _ in records}
    meta = {}
    with gzip.open(ext / FILES[1], "rt") as f:
        for line in f:
            m = ASIN_RE.search(line)
            if m and m.group(1) in asins:
                d = ast.literal_eval(line)
                if isinstance(d, dict) and d.get("asin") in asins:
                    meta[d["asin"]] = d
    names, descs, no_title, no_meta = {}, {}, [], 0
    for a in asins:
        d = meta.get(a)
        no_meta += d is None
        t = clean((d or {}).get("title"))
        if not t:
            no_title.append(a)
            t = f"Product {a}"
        names[a], descs[a] = t, describe(d or {})
    n_no_desc = sum(v is None for v in descs.values())
    print(f"items {len(asins)}; no meta {no_meta}; no title {len(no_title)}; no description {n_no_desc}")
    seqs = rec_data.user_sequences(records)
    stats = rec_data.build(TASK, seqs, names, descs,
                           header="Products the user reviewed, oldest to newest:",
                           instructions="Which product will the user review next?")
    ties = sum(sum(1 for a, b in zip(s, s[1:]) if a == b) for s in [sorted(
        ts for _, _, ts in grp) for grp in _by_user(records).values()])
    sources = [(f, BASE_URL + f, hashlib.sha256((ext / f).read_bytes()).hexdigest()) for f in FILES]
    print(rec_data.write_card(
        TASK, "Amazon Beauty 2014, 5-core", sources, stats,
        preprocessing=[
            "Interactions = all reviews in the official 5-core file (no further filtering; no duplicate "
            "(user, item) pairs present).",
            "Per user, sorted by `unixReviewTime` (day resolution); ties keep file order (stable sort).",
            "Item indices 1..n_items assigned in ascending (string) asin order.",
            "Metadata parsed with `ast.literal_eval` for the asins in the 5-core file only.",
            "Name = HTML-unescaped, whitespace-normalised `title`, clipped to 100 chars; items without a "
            "title are named `Product <asin>`.",
            "Description = first category path under `Beauty` without the root (e.g. "
            "`Skin Care > Body > Moisturizers`), plus ` (brand: X)` when a brand exists, clipped to 150 "
            "chars; null if neither exists.",
            f"Base seed {rec_data.SEED}; 1,000 test + 300 dev users (disjoint), leave-one-out; "
            "10,000 train items (targets at positions 3..len-3, ≤3 per user) in seeded order.",
            "State: header + the most recent ≤20 history items, numbered oldest → newest.",
            "Candidates: gold + 19 negatives uniform over items the user never reviewed (seeded per item id); "
            "option order seeded shuffle; duplicate names disambiguated with ` (#<item index>)`.",
        ],
        deviations=[
            f"{len(no_title)} of {len(asins):,} items have no title in the metadata "
            f"({no_meta} have no metadata line at all); they stay in all sequences and are named "
            "`Product <asin>`.",
            f"{n_no_desc} items have neither category nor brand (description null).",
            f"Timestamps are day-resolution: {ties:,} consecutive same-day pairs within users, ordered by "
            "file order (which is grouped by asin, not true time).",
        ]))
    print({k: v for k, v in stats.items() if k != "example_state"})


def _by_user(records):
    out = {}
    for u, a, ts in records:
        out.setdefault(u, []).append((u, a, ts))
    return out


if __name__ == "__main__":
    main()
