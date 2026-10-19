"""Pause part of the work without stopping running drivers: claim the unfinished items for a placeholder process.

Running drivers skip claimed items, finish what they hold, and move on; nothing in flight is lost. The placeholder is
a long-lived `sleep` started here, so the claims stay live (same SLURM job, same host, live pid); killing it makes
them stale and the next driver takes the items again (src/s1clin/claim.py).

  .venv/bin/python scripts/hold_items.py hold --clinenv-slugs qwen3.8-27b__clef_k5__run1,...   -> prints the holder pid
  .venv/bin/python scripts/hold_items.py hold --pb llm_log_k5/run1 --pb-keep configs/pb_pilot20.txt  (all but those)
  .venv/bin/python scripts/hold_items.py release --pid PID      (kills the holder and removes its claim files)
"""

import argparse
import os
import socket
import subprocess
from pathlib import Path

TREE = Path("/ocean/projects/cis260032p/yliy/data/clinenv/tree")
CASES = Path("/ocean/projects/cis260032p/yliy/data/clinenv/eval_subset_paper7.txt")
JOBS = Path("/ocean/projects/cis260032p/yliy/data/physicianbench/jobs")
PB_TASKS = Path(__file__).resolve().parents[1] / "third_party/physicianbench/tasks/v1"


def _claim(path, me):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:  # held by a running driver: it finishes that item
        return 0
    os.write(fd, me.encode())
    os.close(fd)
    return 1


def hold(slugs, pb=None, pb_keep=None):
    p = subprocess.Popen(["sleep", "infinity"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    me = f"{os.environ.get('SLURM_JOB_ID', 'nojob')} {socket.gethostname()} {p.pid}"
    n = 0
    for line in CASES.read_text().splitlines():
        s, h = line.split()
        for slug in slugs:
            if (TREE / s / h / "evals" / slug / "latest.json").exists():
                continue
            n += _claim(TREE / s / h / "runs" / f"{slug}.claim", me)
    if pb:
        keep = set(Path(pb_keep).read_text().split()) if pb_keep else set()
        for t in sorted(d.name for d in PB_TASKS.iterdir() if d.is_dir()):
            if t not in keep and not (JOBS / pb / t / "metadata.json").exists():
                n += _claim(JOBS / pb / f"{t}.claim", me)
    print(f"holder pid {p.pid}: {n} items held")


def release(pid):
    mark = f" {socket.gethostname()} {pid}"
    n = 0
    for claim in list(TREE.glob("*/*/runs/*.claim")) + list(JOBS.glob("*/*/*.claim")):
        try:
            if claim.read_text().endswith(mark):
                claim.unlink()
                n += 1
        except FileNotFoundError:
            pass
    try:
        os.kill(pid, 15)
    except ProcessLookupError:
        pass
    print(f"released {n} items held by pid {pid}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["hold", "release"])
    ap.add_argument("--clinenv-slugs", default="")
    ap.add_argument("--pb", default=None, help="PhysicianBench ARM_TAG/runN to hold")
    ap.add_argument("--pb-keep", default=None, help="file of tasks NOT to hold")
    ap.add_argument("--pid", type=int)
    a = ap.parse_args()
    os.umask(0o077)
    hold([s for s in a.clinenv_slugs.split(",") if s], a.pb, a.pb_keep) if a.cmd == "hold" else release(a.pid)


if __name__ == "__main__":
    main()
