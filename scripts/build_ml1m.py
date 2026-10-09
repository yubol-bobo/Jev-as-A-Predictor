"""Build frozen MovieLens-1M items (see src/seqpred/rec_data.py for the shared protocol).

  python scripts/build_ml1m.py      # expects data/external/ml1m/ml-1m.zip
"""

import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from seqpred import items as I, rec_data  # noqa: E402

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
TASK = "ml1m"


def main():
    raw = I.EXTERNAL / TASK / "ml-1m.zip"
    with zipfile.ZipFile(raw) as z:
        movies = z.read("ml-1m/movies.dat").decode("latin-1").splitlines()
        ratings = z.read("ml-1m/ratings.dat").decode("latin-1").splitlines()
    names, descs = {}, {}
    for line in movies:
        mid, title, genres = line.split("::")
        names[int(mid)] = title.strip()
        g = [x for x in genres.split("|") if x and x != "(no genres listed)"]
        descs[int(mid)] = ", ".join(g) if g else None
    records = []
    for line in ratings:
        u, m, _, ts = line.split("::")
        records.append((int(u), int(m), int(ts)))
    seqs = rec_data.user_sequences(records)
    stats = rec_data.build(TASK, seqs, names, descs,
                           header="Movies the user watched, oldest to newest:",
                           instructions="Which movie will the user watch next?",
                           user_key=int, item_key=int)
    sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    print(rec_data.write_card(
        TASK, "MovieLens-1M", [("ml-1m.zip", URL, sha)], stats,
        preprocessing=[
            "`ratings.dat` and `movies.dat` decoded as latin-1; every rating (any score) is an interaction.",
            "Per user, interactions sorted by timestamp; ties keep `ratings.dat` file order (stable sort).",
            "Item indices 1..n_items assigned in ascending MovieID order over movies with ≥1 rating.",
            "Option/history name = `movies.dat` title verbatim (e.g. `American President, The (1995)`), "
            "clipped to 100 chars; description = genres joined by `, `.",
            f"Base seed {rec_data.SEED}; 1,000 test + 300 dev users (disjoint), leave-one-out; "
            "10,000 train items (targets at positions 3..len-3, ≤3 per user) in seeded order.",
            "State: header + the most recent ≤20 history items, numbered oldest → newest.",
            "Candidates: gold + 19 negatives uniform over movies the user never rated (seeded per item id); "
            "option order seeded shuffle; duplicate names disambiguated with ` (#<item index>)`.",
        ],
        deviations=["No k-core filtering (ML-1M users already have ≥20 ratings).",
                    "Movie titles are not re-ordered (`X, The (year)` kept as in the source)."]))
    print({k: v for k, v in stats.items() if k != "example_state"})


if __name__ == "__main__":
    main()
