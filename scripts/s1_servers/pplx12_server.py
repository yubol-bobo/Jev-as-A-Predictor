"""POST /v1/systemone for perplexity-ai/pplx-decider-v1.2, which ships a library (source/src/pplx_decider) and no server.

The request goes to Decider.decide unchanged, on the reference path (fast=False: the unmodified Transformers model,
which the card says reproduces the evaluated implementation), with the saved calibration temperature. One request at a
time. Listens on 127.0.0.1 only.
  PYTHONPATH=<snapshot>/source/src python pplx12_server.py --checkpoint <snapshot> --port 27340
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pplx_decider import Decider

LOCK = threading.Lock()
DECIDER = None
INFO = {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path != "/health":
            self.send_error(404)
            return
        self._send(200, {"status": "ready", **INFO})

    def do_POST(self):
        if self.path != "/v1/systemone":
            self.send_error(404)
            return
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        req.pop("model", None)
        t0 = time.time()
        try:
            with LOCK:
                out = DECIDER.decide(req)
            out["latency_ms"] = round(1000 * (time.time() - t0), 1)
            self._send(200, out)
        except Exception as e:
            self._send(500, {"error": repr(e)[:1000]})

    def _send(self, code, out):
        data = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    global DECIDER, INFO
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    DECIDER = Decider.from_pretrained(Path(a.checkpoint), device="cuda")
    INFO = {"model": "pplx-decider-v1.2", "checkpoint": a.checkpoint, "temperature": DECIDER.temperature,
            "fast": False, "max_length": DECIDER.max_length}
    print("pplx12 ready", json.dumps(INFO), flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
