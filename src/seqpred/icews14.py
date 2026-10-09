"""ICEWS14 temporal knowledge graph: download, loading, names, dates and history lookup.

Source: the ICEWS14 split used by RE-GCN (Li et al., 2021) and the recurrency baseline
(Gastinger et al., 2024), as distributed in nec-research/recurrency_baseline_tkg data/ICEWS14
(identical facts and ids to the RE-GCN data release `data/ICEWS14s`, whose timestamps are days 1..365;
the recurrency copy stores hours: t = (day - 1) * 24). See results/data_cards/icews14.md.

Ids: entities 0..7127 and relations 0..229 as in entity2id.txt / relation2id.txt. `day` is the
0-based day offset from 2014-01-01 (= t_hours / 24). Item-level entity indices are id + 1.
"""

import datetime as dt
import hashlib
from collections import defaultdict

import numpy as np

from .items import EXTERNAL

SRC_DIR = EXTERNAL / "icews14" / "dl" / "recurrency"
REPO = "https://raw.githubusercontent.com/nec-research/recurrency_baseline_tkg/master/data/ICEWS14/"
SHA256 = {
    "train.txt": "09d9ec8df4b779e9793229d240a79c762c5c6f7846b31ee5f2522413a6595d57",
    "valid.txt": "bb9462c02b6fb56f15822692ab9f80d782d6c2cc89e5ed7a83bae2c6022cab66",
    "test.txt": "a72ddd5cc3c86105d48b76620a1c08223fcdc43ef57d1eb9e10b27fd05b471ab",
    "entity2id.txt": "0251b09f1c2e56cdc5506f9d7f83e7a88d5d54c3df25609ae8ce4edb7fb3e573",
    "relation2id.txt": "32fcce0906d05205824f03e980981436a71a85feba72a817f5291e4afda9d04e",
}
SPLITS = ("train", "valid", "test")
BASE_DATE = dt.date(2014, 1, 1)
HOURS_PER_DAY = 24


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download():
    import requests
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    for name, digest in SHA256.items():
        path = SRC_DIR / name
        if not path.exists():
            r = requests.get(REPO + name, timeout=120)
            r.raise_for_status()
            path.write_bytes(r.content)
        assert sha256(path) == digest, f"sha256 mismatch for {path}"


def display_name(raw):
    # entity2id uses '_' for spaces; one Xinjiang entity lost its (non-breaking) space upstream
    return raw.replace("_", " ").replace("XinjiangUyghur", "Xinjiang Uyghur")


class ICEWS14:
    def __init__(self):
        download()
        def names(fn):
            rows = [l.split("\t") for l in (SRC_DIR / fn).read_text(encoding="utf-8").splitlines()]
            assert [int(i) for _, i in rows] == list(range(len(rows)))
            return [n for n, _ in rows]
        self.ent_raw, self.rel_raw = names("entity2id.txt"), names("relation2id.txt")
        self.n_ent, self.n_rel = len(self.ent_raw), len(self.rel_raw)
        ent = [display_name(n) for n in self.ent_raw]
        dup = {n for n in ent if ent.count(n) > 1} if len(set(ent)) < len(ent) else set()
        self.ent = [f"{n} [{i + 1}]" if n in dup else n for i, n in enumerate(ent)]
        self.rel = [display_name(n) for n in self.rel_raw]

        parts, split = [], []
        for k, s in enumerate(SPLITS):
            a = np.loadtxt(SRC_DIR / f"{s}.txt", dtype=np.int64)[:, :4]
            assert (a[:, 3] % HOURS_PER_DAY == 0).all()
            parts.append(a)
            split.append(np.full(len(a), k))
        f = np.concatenate(parts)
        f[:, 3] //= HOURS_PER_DAY
        assert (np.diff(f[:, 3]) >= 0).all(), "facts must be in time order"
        self.facts = f  # (N, 4): s, r, o, day   (global index = chronological order)
        self.split = np.concatenate(split)
        self.s, self.r, self.o, self.day = f.T

        by_ent, by_sr = defaultdict(list), defaultdict(list)
        self.answers = defaultdict(set)  # (s, r, day) -> true objects (time-aware filter)
        for i, (s, r, o, d) in enumerate(f.tolist()):
            by_ent[s].append(i)
            if o != s:
                by_ent[o].append(i)
            by_sr[(s, r)].append(i)
            self.answers[(s, r, d)].add(o)
        self.by_ent = {k: np.array(v) for k, v in by_ent.items()}
        self.by_sr = {k: np.array(v) for k, v in by_sr.items()}
        # first global index with day >= d, for d = 0..max_day+1
        self.day_start = np.searchsorted(self.day, np.arange(self.day.max() + 2))

    def split_idx(self, split):
        return np.flatnonzero(self.split == SPLITS.index(split))

    @staticmethod
    def date(day):
        return (BASE_DATE + dt.timedelta(days=int(day))).isoformat()

    def before(self, idx_array, day):
        """Entries of a chronological fact-index array whose day is < `day`."""
        if idx_array is None or len(idx_array) == 0:
            return np.zeros(0, dtype=np.int64)
        return idx_array[: np.searchsorted(idx_array, self.day_start[day])]

    def fact_text(self, i):
        s, r, o, d = self.facts[i]
        return f"{self.date(d)}: {self.ent[s]} — {self.rel[r]} — {self.ent[o]}"
