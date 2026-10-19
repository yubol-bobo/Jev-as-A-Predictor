"""The /decide API of scripts/clef_server.py in front of another System One model's own /v1/systemone server.

POST /decide {"instruction", "steps": [{"action", "results": [str]}], "question", "options": {name: text}}
  -> {"distribution": {name: p}, "input_tokens", "fit", "wall_s"}

Everything the decider sees is built exactly as for Clef: the same state rendering (task, then every action and its
tool results), the same question and options, and the same fitting rule into a 32,768-token budget (results
shortened oldest-first to 800 characters, then omitted oldest-first). Only the token count comes from the decider's
own tokenizer, plus a fixed allowance for its prompt template. The request is the System One wire format with one
choice question; the probabilities come back from the author's own server, unchanged. One request at a time.

  .venv/bin/python scripts/s1_proxy.py --name d3 --port 8023 --backend http://127.0.0.1:27330/v1/systemone \
      --tokenizer <snapshot>/tokenizer.json
"""

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clef_render import fit_request  # noqa: E402  (the rendering and fitting shared with clef_server.py)

MAX_CONTEXT = 32_768
TEMPLATE_ALLOWANCE = 512  # chat template, question header and option codes of the decider's prompt
LOCK = threading.Lock()
A = None
TOK = None


def count(req) -> int:
    q = req["questions"]["next"]
    text = req["state"] + "\n" + q["instructions"] + "\n" + "\n".join(f"{k}: {v}" for k, v in q["criteria"].items())
    return len(TOK.encode(text, add_special_tokens=False).ids) + TEMPLATE_ALLOWANCE


def probabilities(answer: dict) -> dict:
    for key in ("probabilities", "distribution", "probs"):
        if isinstance(answer.get(key), dict):
            return answer[key]
    raise ValueError(f"no probabilities in answer: {json.dumps(answer)[:300]}")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path != "/health":
            self.send_error(404)
            return
        self._send(200, {"name": A.name, "backend": A.backend, "max_context": MAX_CONTEXT})

    def do_POST(self):
        if self.path != "/decide":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        t0 = time.time()
        try:
            req, n, fit = fit_request(body["instruction"], body["steps"], body["question"], body["options"],
                                      count, MAX_CONTEXT)
            if A.model:
                req["model"] = A.model
            with LOCK:
                for attempt in range(60):  # some servers answer 529 while busy
                    r = requests.post(A.backend, json=req, timeout=900)
                    if r.status_code != 529:
                        break
                    time.sleep(1)
            r.raise_for_status()
            dist = probabilities(r.json()["answers"]["next"])
            missing = set(body["options"]) - set(dist)
            if missing:
                raise ValueError(f"options missing from the answer: {sorted(missing)}")
            out = {"distribution": {k: float(dist[k]) for k in body["options"]}, "input_tokens": n, "fit": fit,
                   "wall_s": round(time.time() - t0, 2)}
            code = 200
        except Exception as e:
            out, code = {"error": repr(e)[:1000]}, 500
        self._send(code, out)

    def _send(self, code, out):
        data = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    global A, TOK
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--backend", required=True, help="the decider's own POST /v1/systemone URL")
    ap.add_argument("--tokenizer", required=True, help="the decider's tokenizer.json")
    ap.add_argument("--model", default=None, help="'model' field some servers require")
    A = ap.parse_args()
    TOK = Tokenizer.from_file(A.tokenizer)
    print(f"{A.name} decider proxy on 127.0.0.1:{A.port} -> {A.backend}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", A.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
