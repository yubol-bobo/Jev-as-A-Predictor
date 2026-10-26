"""M (docs/research_loop.md): the committee of six System One models (listens) against the agent LLM ranker
(listqwen) and pplx12 alone (list), pooled over both agents' cases; aggregates only in results/pb_m.json.

Runs pb_r.py for each agent with --percase into data/offline/pb_m/ (per case, kept out of the repository), then pairs
the arms per case and bootstraps over cases, stratified by agent (each agent's cases resampled within the agent).

  .venv/bin/python scripts/pb_m.py [--out results/pb_m.json]
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PD = Path("/ocean/projects/cis260032p/yliy/data/offline/pb_m")
AGENTS = (("qwen", ""), ("gemma4", "_gemma4"))
PAIRS = (("listens", "listqwen"), ("listens", "list"), ("list", "listqwen"),  # M1, M2, L2 pooled (exploratory)
         ("listens", "none"), ("listens", "listrand"), ("listqwen", "none"), ("list", "none"))


def strat_boot(groups, n=2000):
    rng = random.Random(0)
    tot = sum(len(g) for g in groups)
    b = sorted(sum(sum(rng.choice(g) for _ in g) for g in groups) / tot for _ in range(n))
    return [round(b[50], 4), round(b[1949], 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/pb_m.json"))
    a = ap.parse_args()
    os.umask(0o077)
    PD.mkdir(parents=True, exist_ok=True)
    per = {}
    for agent, suf in AGENTS:
        path = PD / f"percase{suf}.json"
        subprocess.run([sys.executable, str(REPO / "scripts/pb_r.py"), "--percase", str(path)], check=True,
                       stdout=subprocess.DEVNULL, env=os.environ | {"S1CLIN_AGENT": agent})
        per[agent] = json.loads(path.read_text())
    res = {}
    for x, y in PAIRS:
        groups, out = [], {}
        for agent, _ in AGENTS:
            px, py = per[agent].get(x, {}), per[agent].get(y, {})
            d = [px[k]["open"] - py[k]["open"] for k in px if k in py and px[k]["open"] is not None
                 and py[k]["open"] is not None]
            if d:
                groups.append(d)
                out[agent] = {"cases": len(d), "mean": round(sum(d) / len(d), 4)}
        if groups:
            tot = sum(len(g) for g in groups)
            out["pooled"] = {"cases": tot, "mean": round(sum(map(sum, groups)) / tot, 4), "ci95": strat_boot(groups)}
        res[f"{x}_minus_{y}"] = out
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
