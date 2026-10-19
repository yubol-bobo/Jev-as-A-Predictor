"""Work claiming and replica routing shared by the PhysicianBench and ClinEnv runners.

Claims let several driver processes, on one node or on several SLURM nodes, work through the same (arm, run, item)
list without running an item twice. An item is claimed by creating `<item dir>.claim` atomically (O_EXCL, safe on
Lustre). The claim records the SLURM job, host and pid; it is stale (and may be taken over) when that job is no
longer running, or, on the same host, when that pid is gone. Finished items are recognised by their outputs, never
by claims.

Routing sends each item (a whole task or case, so prefix caching stays on one server) to the least-loaded Qwen
replica when it starts: load = (running + waiting requests, from the replica's vLLM /metrics, plus items this process
sent there in the last minute) / weight. A spec is "URL[*weight],URL[*weight],..."; the replicas are identical apart
from tensor parallelism (scripts/vllm_qwen_up.sh), and the weight is roughly each replica's KV-cache capacity.
Until 2026-10-19 09:00 routing was a weighted hash of the item key; it left the small replica idle at times.
If no replica answers /metrics, the weighted hash is used.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import socket
import subprocess
import time
from pathlib import Path

import requests

_running: tuple[float, set[str]] = (0.0, set())


def _running_jobs() -> set[str]:
    global _running
    if time.time() - _running[0] > 60:
        try:
            out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-t", "R", "-o", "%i"],
                                 capture_output=True, text=True, timeout=60, check=True).stdout
            _running = (time.time(), set(out.split()))
        except Exception:  # squeue unavailable: treat every claim as live (never steal by mistake)
            return {"*"}
    return _running[1]


def _me() -> str:
    return f"{os.environ.get('SLURM_JOB_ID', 'nojob')} {socket.gethostname()} {os.getpid()}"


def _stale(text: str) -> bool:
    try:
        job, host, pid = text.split()[:3]
    except ValueError:
        return True
    jobs = _running_jobs()
    if "*" in jobs:
        return False
    if job != "nojob" and job not in jobs:
        return True
    if host == socket.gethostname():
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass
    return False


def claim(item_dir: Path) -> bool:
    """True if this process now owns item_dir (claim file created, or a stale one taken over)."""
    path = item_dir.parent / (item_dir.name + ".claim")
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.write(fd, _me().encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                text = path.read_text()
            except FileNotFoundError:
                continue
            if not _stale(text):
                return False
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    return False


def release(item_dir: Path) -> None:
    path = item_dir.parent / (item_dir.name + ".claim")
    try:
        if path.read_text() == _me():
            path.unlink()
    except FileNotFoundError:
        pass


def parse_pool(spec: str) -> list[tuple[str, int]]:
    pool = []
    for part in spec.split(","):
        url, _, w = part.partition("*")
        pool.append((url.strip(), int(w) if w else 1))
    return pool


_recent: dict[str, list[float]] = {}
_lock = threading.Lock()


def _load(url: str) -> float | None:
    try:
        text = requests.get(url.rsplit("/v1", 1)[0] + "/metrics", timeout=5).text
    except Exception:
        return None
    vals = re.findall(r"^vllm:num_requests_(?:running|waiting)\{[^}]*\} ([0-9.]+)$", text, re.M)
    return sum(float(v) for v in vals) if vals else None


def route(spec: str, key: str) -> str:
    """The replica URL for an item: least loaded per unit of capacity, else a weighted hash of the key."""
    pool = parse_pool(spec)
    if len(pool) > 1:
        loads = {url: _load(url) for url, _ in pool}
        if any(v is None for v in loads.values()) and any(v is not None for v in loads.values()):
            pool = [(u, w) for u, w in pool if loads[u] is not None]  # a replica that is down gets no new items
        if all(loads[u] is not None for u, _ in pool):
            with _lock:
                now = time.time()
                for url, _ in pool:
                    _recent[url] = [t for t in _recent.get(url, []) if now - t < 60]
                best = min(pool, key=lambda p: (loads[p[0]] + 5 * len(_recent[p[0]])) / p[1])[0]
                _recent[best].append(now)
            return best
    return route_hash(spec, key)


def route_hash(spec: str, key: str) -> str:
    """Weighted, deterministic in the item key."""
    pool = parse_pool(spec)
    x = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % sum(w for _, w in pool)
    for url, w in pool:
        if x < w:
            return url
        x -= w
    return pool[-1][0]
