"""Train RE-GCN on ICEWS14 train.txt and write candidate-set predictions for the dev/test items.

  CUDA_VISIBLE_DEVICES="" python scripts/run_icews14_regcn.py --max-epochs 30

Settings follow the RE-GCN ICEWS14 config (history length 3 snapshots, hidden 200, 2 uvrgcn layers,
ConvTransE decoder, layer norm, self-loop, dropout 0.2, lr 1e-3; static graph off, entity prediction
only). Model selection: best validation raw MRR (as in the reference main.py), with early stopping
(patience) to keep CPU training feasible. Items are scored with the decoder score of (s, r, ?) given
the ground-truth snapshots of the 3 days before t; softmax temperature fitted on dev NLL.
Also reports the full-ranking (all entities) raw and time-aware filtered MRR on test.txt.
"""

import argparse
import copy
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from seqpred import items as I, metrics, sasrec  # noqa: E402
from seqpred.icews14 import ICEWS14, SPLITS  # noqa: E402
from seqpred.regcn import REGCN, Snapshot, ranks, with_inverse  # noqa: E402

TASK = "icews14"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="regcn")
    ap.add_argument("--history-len", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=200)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    t_start = time.time()

    ds = ICEWS14()
    R = ds.n_rel
    days = np.unique(ds.day)
    snaps, trip = {}, {}
    for d in days.tolist():
        trip[d] = ds.facts[ds.day == d][:, :3]
        snaps[d] = Snapshot(trip[d], ds.n_ent, R)

    def hist(day):
        prev = days[days < day][-a.history_len:]
        return [snaps[d] for d in prev.tolist()]

    split_days = {s: np.unique(ds.day[ds.split_idx(s)]).tolist() for s in SPLITS}

    model = REGCN(ds.n_ent, R, h=a.hidden, n_layers=a.layers, dropout=a.dropout,
                  input_dropout=a.dropout, hidden_dropout=a.dropout, feat_dropout=a.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)

    def evaluate(split):
        """Full-ranking MRR over all facts of a split, both directions (as RE-GCN) and object-only."""
        model.eval()
        out = defaultdict(list)
        with torch.no_grad():
            for d in split_days[split]:
                q = with_inverse(trip[d], R)
                sc = model.score(hist(d), q[:, 0], q[:, 1])
                ans = defaultdict(list)
                for s, r, o in q.tolist():
                    ans[(s, r)].append(o)
                filt = [torch.tensor(ans[(s, r)]) for s, r, _ in q.tolist()]
                n = len(trip[d])
                for kind, rk in (("raw", ranks(sc, q[:, 2])), ("filt", ranks(sc, q[:, 2], filt))):
                    out[kind].append(rk)
                    out[kind + "_obj"].append(rk[:n])
        res = {}
        for k, v in out.items():
            rk = torch.cat(v).float()
            res[k] = {"mrr": round(float((1 / rk).mean()), 4), "h1": round(float((rk <= 1).float().mean()), 4),
                      "h3": round(float((rk <= 3).float().mean()), 4), "h10": round(float((rk <= 10).float().mean()), 4)}
        return res

    best, best_state, bad, log = -1.0, None, 0, []
    train_days = split_days["train"][1:]  # the first snapshot has no history (skipped as in RE-GCN)
    for epoch in range(a.max_epochs):
        model.train()
        t0 = time.time()
        random.shuffle(train_days)
        losses = []
        for d in train_days:
            q = with_inverse(trip[d], R)
            loss = F.cross_entropy(model.score(hist(d), q[:, 0], q[:, 1]), q[:, 2])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
        v = evaluate("valid")
        rec = {"epoch": epoch, "loss": round(float(np.mean(losses)), 4), "valid": v,
               "seconds": round(time.time() - t0, 1)}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if v["raw"]["mrr"] > best:
            best, best_state, bad = v["raw"]["mrr"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    ckpt = ROOT / "results" / "checkpoints" / TASK / f"{a.system}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt)
    train_seconds = time.time() - t_start

    full = {"valid": evaluate("valid"), "test": evaluate("test")}
    print("full ranking:", json.dumps(full), flush=True)

    # candidate-set scores for the frozen items
    model.eval()
    its = {sp: I.load_items(TASK, sp) for sp in ("dev", "test")}
    logits = {}
    with torch.no_grad():
        for sp, v in its.items():
            by_day = defaultdict(list)
            for k, it in enumerate(v):
                by_day[it["meta"]["query"]["day"]].append(k)
            z = [None] * len(v)
            for d, ks in by_day.items():
                s = torch.tensor([v[k]["meta"]["query"]["s"] for k in ks])
                r = torch.tensor([v[k]["meta"]["query"]["r"] for k in ks])
                sc = model.score(hist(d), s, r)
                for row, k in enumerate(ks):
                    z[k] = sc[row, torch.tensor(v[k]["meta"]["option_ids"]) - 1].numpy()
            logits[sp] = z
    T = sasrec.fit_temperature(logits["dev"], [it["gold"] for it in its["dev"]])
    report = {"task": TASK, "system": a.system, "args": vars(a), "temperature": T,
              "best_valid_raw_mrr": best, "epochs_run": len(log), "train_seconds": round(train_seconds, 1),
              "full_ranking": full, "log": log}
    for sp, v in its.items():
        I.write_predictions(TASK, a.system, sp,
                            [{"id": it["id"], "p": sasrec.softmax(z, T).round(6).tolist()} for it, z in zip(v, logits[sp])])
        s = metrics.summarize(v, I.load_predictions(TASK, a.system, sp))
        report[sp] = {k: (x if not isinstance(x, tuple) else [round(y, 4) for y in x]) for k, x in s.items()}
        print(sp, {k: report[sp][k] for k in ("hr1", "ndcg10", "mrr", "nll", "ece")})
    report["total_seconds"] = round(time.time() - t_start, 1)
    (I.PREDS / TASK / a.system / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
