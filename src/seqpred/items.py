"""Shared item format for all tasks.

Item files live at data/items/<task>/{dev,test,train}.jsonl. One JSON object per line:

  id        unique string, e.g. "ml1m-test-u123"
  task      ml1m | beauty | fsq_nyc | icews14 | bpic12
  split     dev | test | train
  group     cluster id for bootstrap (user id, case id, query subject)
  request   the System One request sent identically to every system:
            {"state": str, "questions": {"next": {"type": "choice", "instructions": str,
                                                  "criteria": {option name: description or None}}}}
            criteria order is a seeded shuffle (presentation order)
  qid       "next"
  options   option names in canonical order; all predictions are aligned with this list
  gold      index of the correct option in `options`
  meta      task-specific ids used by trained baselines, at least:
            option_ids  raw ids aligned with `options`
            history_ids raw ids of the history shown in the state (oldest → newest)

train.jsonl is stored in a seeded random order: the first N lines are the training subset of
size N, so subsets are nested (500 ⊂ 2,000 ⊂ 10,000).

Predictions live at results/predictions/<task>/<system>/<split>.jsonl with lines
  {"id": str, "p": [probabilities aligned with options]}  (or {"id", "status": "error", ...}).
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ITEMS = ROOT / "data" / "items"
PREDS = ROOT / "results" / "predictions"
EXTERNAL = ROOT / "data" / "external"  # raw downloads (gitignored)

QID = "next"
MAX_OPTIONS = 255
MAX_STATE_CHARS = 6000
TRAIN_SIZES = (500, 2000, 10000)


def choice_request(state, instructions, options, descriptions=None, seed=None):
    """Build the request; option presentation order is shuffled with `seed`."""
    descriptions = descriptions or {}
    order = list(options)
    random.Random(seed).shuffle(order)
    return {"state": state,
            "questions": {QID: {"type": "choice", "instructions": instructions,
                                "criteria": {o: descriptions.get(o) for o in order}}}}


def make_item(*, id, task, split, group, state, instructions, options, gold, meta,
              descriptions=None):
    item = {"id": id, "task": task, "split": split, "group": str(group),
            "request": choice_request(state, instructions, options, descriptions, seed=id),
            "qid": QID, "options": list(options), "gold": int(gold), "meta": meta}
    validate(item)
    return item


def validate(item):
    opts = item["options"]
    assert len(opts) == len(set(opts)), f"{item['id']}: duplicate option names"
    assert 2 <= len(opts) <= MAX_OPTIONS, f"{item['id']}: {len(opts)} options"
    assert 0 <= item["gold"] < len(opts), f"{item['id']}: bad gold"
    assert len(item["request"]["state"]) <= MAX_STATE_CHARS, f"{item['id']}: state too long"
    crit = item["request"]["questions"][QID]["criteria"]
    assert set(crit) == set(opts), f"{item['id']}: criteria/options mismatch"
    meta = item["meta"]
    assert len(meta["option_ids"]) == len(opts), f"{item['id']}: option_ids misaligned"
    assert "history_ids" in meta


def write_items(task, split, items):
    path = ITEMS / task / f"{split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = set()
    with path.open("w") as f:
        for it in items:
            validate(it)
            assert it["id"] not in ids, f"duplicate id {it['id']}"
            ids.add(it["id"])
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return path


def load_items(task, split, n=None):
    out = []
    with (ITEMS / task / f"{split}.jsonl").open() as f:
        for line in f:
            out.append(json.loads(line))
            if n is not None and len(out) >= n:
                break
    return out


def write_predictions(task, system, split, rows):
    path = PREDS / task / system / f"{split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def load_predictions(task, system, split):
    path = PREDS / task / system / f"{split}.jsonl"
    return {r["id"]: r for r in map(json.loads, path.open())}
