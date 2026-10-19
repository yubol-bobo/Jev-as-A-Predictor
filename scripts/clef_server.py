"""Local HTTP decider: Clef-27B (pinned native joint head) choosing among an agent's candidate next turns.

POST /decide {"instruction", "steps": [{"action", "results": [str]}], "question", "options": {name: text}}
  -> {"distribution": {name: p}, "input_tokens", "fit"}

The state is the task instruction plus every step so far (action, then its tool results as the LLM saw them),
fitted deterministically into Clef's 32,768-token context:
  - the instruction and every action are always kept in full;
  - tool results are kept in full, then shortened oldest-first to SHORT_CHARS, then replaced oldest-first by
    "[result omitted]", until the complete encoded request fits.
`fit` reports how many results were shortened or omitted. Listens on 127.0.0.1 only.

Run (environment as in Jev-as-A-Predictor/scripts/run_clef.py):
  CUDA_VISIBLE_DEVICES=1 HF_HOME=/ocean/projects/cis260032p/yliy/huggingface_cache HF_HUB_OFFLINE=1 \
  PYTHONPATH=/ocean/projects/cis260032p/yliy/venvs/clef_deps_py312 \
  /ocean/projects/cis260032p/yliy/venvs/s1eval_kev/bin/python scripts/clef_server.py --port 8021
"""

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UCJ = "/ocean/projects/cis260032p/yliy/research/jev/understanding_capability_of_jev/src"
sys.path.insert(0, UCJ)
from s1eval.clef_system import MAX_CONTEXT, ClefScorer, encode_complete  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clef_render import fit_request  # noqa: E402

LOCK = threading.Lock()
SCORER = None


def clef_count(req):
    return len(encode_complete(SCORER.processor.tokenizer, req).input_ids)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        if self.path != "/decide":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        t0 = time.time()
        try:
            with LOCK:
                req, n, fit = fit_request(body["instruction"], body["steps"], body["question"], body["options"],
                                          clef_count, MAX_CONTEXT)
                call, results = SCORER.run_items([dict(req, model="clef")])[0]
            if call.get("status") != "ok":
                raise ValueError(call.get("error"))
            out = {"distribution": results[0]["distribution"], "input_tokens": n, "fit": fit,
                   "wall_s": round(time.time() - t0, 2)}
            code = 200
        except Exception as e:
            out, code = {"error": repr(e)[:1000]}, 500
        data = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    global SCORER
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8021)
    a = ap.parse_args()
    SCORER = ClefScorer()
    print(f"clef decider on 127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
