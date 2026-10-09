"""JEV /v1/systemone client with a content-addressed disk cache.

Every request is keyed by sha256(model + canonical JSON of state/questions); a
cached response is never re-requested, so re-running analyses costs nothing.
Raw responses (with latency/usage) are stored verbatim under results/cache/.
"""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "results" / "cache" / "jev"
ENV_FILE = Path(os.environ.get("JEV_ENV_FILE", "/ocean/projects/cis260032p/yliy/research/jev/.env"))
URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
RETRY_STATUS = (429, 500, 502, 503, 504, 520, 522, 524)


def _api_key():
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() in ("TYPESAFE_API_KEY", "SAFETYPE_API_KEY") and v.strip():
                return v.strip().strip('"').strip("'")
    raise KeyError("TYPESAFE/SAFETYPE API key not found")


def request_key(request, model=MODEL):
    blob = json.dumps({"model": model, "state": request["state"], "questions": request["questions"]},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _cache_path(key):
    return CACHE / key[:2] / f"{key}.json"


def call(request, model=MODEL, timeout=120, retries=5):
    """Return the cached-or-fresh call record: {key, status, response, latency_s, usage, cached}."""
    key = request_key(request, model)
    path = _cache_path(key)
    if path.exists():
        rec = json.loads(path.read_text())
        rec["cached"] = True
        return rec
    payload = {"model": model, "state": request["state"], "questions": request["questions"]}
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + _api_key()}
    resp, err, latency = None, None, None
    for attempt in range(1, retries + 1):
        t0 = time.perf_counter()
        try:
            resp = requests.post(URL, headers=headers, json=payload, timeout=timeout)
            latency = time.perf_counter() - t0
            if resp.status_code in RETRY_STATUS and attempt < retries:
                time.sleep(float(resp.headers.get("retry-after", 2 ** attempt)))
                continue
            break
        except requests.RequestException as exc:  # transport error
            err = repr(exc)
            time.sleep(2 ** attempt)
    if resp is None:
        return {"key": key, "status": "transport_error", "error": err, "cached": False}
    try:
        body = resp.json()
    except ValueError:
        body = {"_non_json_body": resp.text[:5000]}
    rec = {"key": key, "model": model, "http_status": resp.status_code, "latency_s": latency,
           "attempts": attempt, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "status": "ok" if resp.status_code == 200 and "answers" in body else "error",
           "response": body, "usage": body.get("usage") if isinstance(body, dict) else None}
    if rec["status"] == "ok":  # only successful calls are cached; errors are retried next run
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}.{id(rec)}")
        tmp.write_text(json.dumps(rec, ensure_ascii=False))
        tmp.replace(path)  # atomic: concurrent identical requests never read a partial file
    rec["cached"] = False
    return rec


def option_names(question):
    if question["type"] == "noul":
        return ["true", "false"]
    if question["type"] == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return list(question["criteria"])


def distribution(answer, question):
    """Map a JEV answer to {option: prob} in the question's option order (missing → 0)."""
    if question["type"] == "noul":
        p = float(answer["noul"])
        return {"true": p, "false": 1.0 - p}
    probs = answer.get("probabilities") or {}
    return {n: float(probs.get(n, 0.0)) for n in option_names(question)}


def run_items(items, model=MODEL, workers=16, log_every=200):
    """Call JEV for every item (dicts with 'id' and 'request'); returns {id: call record}."""
    out = {}
    n_fresh = 0
    with ThreadPoolExecutor(workers) as pool:
        futs = {it["id"]: pool.submit(call, it["request"], model) for it in items}
        for i, (iid, fut) in enumerate(futs.items(), 1):
            out[iid] = fut.result()
            n_fresh += not out[iid].get("cached")
            if log_every and i % log_every == 0:
                print(f"  {i}/{len(items)} done ({n_fresh} fresh)", flush=True)
    return out
