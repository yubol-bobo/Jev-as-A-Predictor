"""Clients for the System One systems, all fed the identical item["request"].

  jev           hosted API  https://api.typesafe.ai/v1/systemone, model jev-1.13.0
  kev27b        local kev.serve (jaredpalmer/kev-27b) on SEQPRED_URL, concurrent requests allowed
  pplx_decider  local autojev-serve (perplexity-ai/pplx-decider-v1.1-27b) on SEQPRED_URL, one request at a time
  clef27b       in-process native ClefScorer (Cloudflare/clef), see scripts/run_clef.py

Every HTTP response is cached by sha256(system, model, state, questions) under results/cache/<system>/,
so a re-run never re-sends a request that already succeeded.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import requests

from .items import ROOT

CACHE = ROOT / "results" / "cache"
ENV_FILE = Path("/ocean/projects/cis260032p/yliy/research/jev/.env")
RETRY = (429, 500, 502, 503, 504, 520, 522, 524, 529)

SYSTEMS = {
    "jev": {"url": "https://api.typesafe.ai", "model": "jev-1.13.0", "key": True, "workers": 16},
    "kev27b": {"url": None, "model": "kev-latest", "key": False, "workers": 8},
    "pplx_decider": {"url": None, "model": "autojev-qwen3.8-27b", "key": False, "workers": 1},
}


def _api_key():
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() in ("TYPESAFE_API_KEY", "SAFETYPE_API_KEY") and v.strip():
                return v.strip().strip('"').strip("'")
    raise KeyError("TYPESAFE/SAFETYPE API key not found")


def cache_key(system, model, request):
    blob = json.dumps({"system": system, "model": model, "state": request["state"],
                       "questions": request["questions"]}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class HTTPSystemOne:
    def __init__(self, system, url=None, timeout=300, retries=8):
        cfg = SYSTEMS[system]
        self.system, self.model = system, cfg["model"]
        self.url = (url or cfg["url"] or os.environ["SEQPRED_URL"]).rstrip("/") + "/v1/systemone"
        self.headers = {"Content-Type": "application/json"}
        if cfg["key"]:
            self.headers["Authorization"] = "Bearer " + _api_key()
        self.workers, self.timeout, self.retries = cfg["workers"], timeout, retries

    def call(self, request):
        key = cache_key(self.system, self.model, request)
        path = CACHE / self.system / key[:2] / f"{key}.json"
        if path.exists():
            return {**json.loads(path.read_text()), "cached": True}
        payload = {"model": self.model, "state": request["state"], "questions": request["questions"]}
        resp, err, latency, attempt = None, None, None, 0
        for attempt in range(1, self.retries + 1):
            t0 = time.perf_counter()
            try:
                resp = requests.post(self.url, headers=self.headers, json=payload, timeout=self.timeout)
                latency = time.perf_counter() - t0
                if resp.status_code in RETRY and attempt < self.retries:
                    time.sleep(float(resp.headers.get("retry-after", min(2 ** attempt, 30))))
                    continue
                break
            except requests.RequestException as exc:
                err = repr(exc)
                time.sleep(min(2 ** attempt, 30))
        if resp is None:
            return {"key": key, "status": "transport_error", "error": err, "cached": False}
        try:
            body = resp.json()
        except ValueError:
            body = {"_non_json_body": resp.text[:5000]}
        rec = {"key": key, "system": self.system, "model": self.model, "http_status": resp.status_code,
               "latency_s": latency, "attempts": attempt, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "status": "ok" if resp.status_code == 200 and "answers" in body else "error", "response": body}
        if rec["status"] == "ok":
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(rec, ensure_ascii=False))
            tmp.replace(path)
        return {**rec, "cached": False}


def choice_distribution(answer, options):
    """{option: prob} from a System One choice answer, aligned with `options` (missing -> 0)."""
    probs = answer.get("probabilities") or {}
    return [float(probs.get(o, 0.0)) for o in options]
