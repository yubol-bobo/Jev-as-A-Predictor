"""HTTP shim for PNPM244/linne-oak-1-0: POST /v1/systemone on top of the authors' own Decision Index engine.

The authors ran their Decision Index 0.3 entry in-process with the kit runner and the engine shipped in the model repo
(dataset PNPM244/linne-oak-1-0-decision-index @ 1225e146, runs/linne-oak-1-0/environment.json):
    engine  = linne_engine:LinneEngine
    options = model=<weights>, device=cuda:0, token_budget=16384, noncausal=1,
              chat_template=<Quyet-1.0-Large/chat_template.jinja>
The chat_template.jinja named there is byte-identical (sha256 ae53464b...) to the one in the linne-oak snapshot (and to
google/gemma-4-31B-it@842da379's), so the snapshot's copy is passed. max_tokens keeps the engine default (65,536), as
in the run. The engine needs no HTTP server; the model card ships none.

This shim only (1) builds that engine once with exactly those options (plus the local snapshot path), seeds torch and
runs the engine's warm-up as the kit runner does, and (2) puts `engine(state, questions)` behind HTTP. It does not
touch prompt rendering, labels, read-out, soft-capping, temperature (1.0) or answers: the request's "state" and
"questions" go to the engine unchanged and the engine's response body is returned unchanged plus one field,
latency_ms. The response "model" is the engine's own name ("linne-fern-1-0": the class attribute the authors reuse;
the published run's responses carry the same string). The request "model" field is not checked.
Each response is checked with the kit's validate() as the kit runner does (a failure -> HTTP 500, never altered).
`Unsupported` (a prompt over 65,536 tokens, a choice with more options than distinct labels, a label that is not one
token, an unknown question type) -> HTTP 422 {"error": "unsupported: <engine message>"}; the engine refuses the whole
request (no truncation). Requests are answered one at a time (lock), like the kit's sequential runner.

    python linne_shim.py --snapshot <dir> --repo PNPM244/linne-oak-1-0 --revision <sha> --host 127.0.0.1 --port P
Routes: POST /v1/systemone; GET /health and /v1/systemone/info ({"ok": true, ...}; the port opens only after loading).
--device / --dtype exist only for the CPU check with a tiny random model (servers/validate/linne_cpu_check.py).
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RUN_OPTIONS = {"token_budget": 16384, "noncausal": 1}   # environment.json of the published run (besides model/device)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def versions():
    out = {"python": sys.version.split()[0]}
    for p in ("torch", "transformers", "accelerate", "safetensors", "tokenizers", "huggingface_hub", "decision-index"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True, help="local snapshot dir (weights + linne_engine.py + nc_hook.py)")
    p.add_argument("--engine-dir", default=None, help="dir holding linne_engine.py / nc_hook.py (default: --snapshot)")
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    a = p.parse_args()

    engine_dir = os.path.abspath(a.engine_dir or a.snapshot)
    sys.path.insert(0, engine_dir)                           # as the card: sys.path.insert(0, path); from linne_engine import ...
    import torch
    import linne_engine
    from decision_index import constants as C
    from decision_index.engines.base import Unsupported, validate

    torch.manual_seed(C.RUN_SEED)                            # kit runner does this before loading (the engine never samples)
    t0 = time.time()
    options = dict(model=a.snapshot, device=a.device, dtype=a.dtype, chat_template=os.path.join(a.snapshot, "chat_template.jinja"),
                   **RUN_OPTIONS)
    eng = linne_engine.LinneEngine(**options)
    eng.synchronize()
    eng.warmup()                                             # kit runner: warm=True
    eng.synchronize()
    lock = threading.Lock()
    info = {"ok": True, "model": a.repo, "revision": a.revision, "engine": "linne_engine:LinneEngine",
            "engine_options": options, "provenance": eng.provenance, "max_tokens": eng.max_tokens,
            "pair_labels": len(eng.pair_labels()), "max_choice_options": max(26, len(eng.pair_labels())),
            "engine_sha256": {f: sha256(os.path.join(engine_dir, f)) for f in ("linne_engine.py", "nc_hook.py")},
            "chat_template_sha256": sha256(options["chat_template"]), "versions": versions(),
            "load_seconds": round(time.time() - t0, 1)}
    print("[linne_shim] ready", json.dumps(info), flush=True)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.rstrip("/") in ("/health", "/v1/systemone/info"):
                return self._send(200, info)
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                return self._send(404, {"error": "not found; use POST /v1/systemone"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                questions = body["questions"]
                if not isinstance(questions, dict) or not questions:
                    raise ValueError("questions must be a non-empty object")
            except Exception as e:
                return self._send(400, {"error": f"bad request: {e!r}"})
            try:
                with lock:
                    t = time.perf_counter()
                    resp, _raw = eng(body.get("state"), questions)
                    eng.synchronize()
                    ms = (time.perf_counter() - t) * 1000
                validate(questions, resp)
            except Unsupported as e:
                return self._send(422, {"error": f"unsupported: {e}"})
            except Exception as e:
                traceback.print_exc()
                return self._send(500, {"error": repr(e)[:2000]})
            self._send(200, dict(resp, latency_ms=round(ms, 2)))

    srv = ThreadingHTTPServer((a.host, a.port), H)
    srv.daemon_threads = True
    print(f"[linne_shim] listening on http://{a.host}:{a.port}/v1/systemone", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
