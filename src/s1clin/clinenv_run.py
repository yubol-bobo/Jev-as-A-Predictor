"""Run ClinEnv (interactive mode) for one arm on local models, then score with ClinEnv's own evaluator.

Cases are the authors' prepared cases (HF release, handled as MIMIC data). Our runs live in a separate case tree whose
`case/` entries are symlinks to the pristine HF cases, so `runs/` and `evals/` never mix with the authors' runs:
  data/clinenv/tree/<subject>/<hadm>/case -> data/clinenv/hf/<subject>/<hadm>/case
  data/clinenv/tree/<subject>/<hadm>/runs/qwen3.8-27b__<arm>__run<N>/{output,dialogue}.json, decisions.jsonl
Each case is routed to one Qwen replica (src/s1clin/claim.py: --llm-url "URL*weight,..."), used for the model under
test, the simulators and the scorer; a case meets the same replica in every arm. Cases are claimed before they run, so
several drivers (processes or SLURM nodes) can share one arm and run.

  python -m src.s1clin.clinenv_run --arm clef --cases data/clinenv/eval_subset_paper7.txt --run 1 --workers 8 \
      --llm-url http://127.0.0.1:8012/v1
  python -m src.s1clin.clinenv_run --arm clef --run 1 --score-only ...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.s1clin.claim import claim, parse_pool, release, route

REPO = Path(__file__).resolve().parents[2]
CE = REPO / "third_party" / "clinenv"
DATA = Path("/ocean/projects/cis260032p/yliy/data/clinenv")
HF, TREE = DATA / "hf", DATA / "tree"
MODEL = "qwen3.8-27b"


def write_configs(llm_url: str) -> tuple[Path, Path]:
    """Per-replica copies of ClinEnv's runtime/eval configs (URLs only; MUT thinking on, simulators off)."""
    cdir = TREE / "_configs"
    cdir.mkdir(parents=True, exist_ok=True)
    port = llm_url.rsplit(":", 1)[1].split("/")[0]
    rt = json.loads((CE / "config/runtime.json").read_text())
    rt["api"]["base_url"] = rt["vllm"]["base_url"] = llm_url
    rt["vllm"]["enabled"] = False  # the replicas parse tool calls natively (qwen3_coder); no text-flattening path
    rt["api"]["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}  # same LLM setting as PhysicianBench
    rt["api"]["timeout"] = 3600  # K=5 thinking requests can queue on a busy replica; 600 s + retries failed cases
    ev = json.loads((CE / "config/eval.json").read_text())
    ev["api"]["base_url"] = ev["vllm"]["base_url"] = llm_url
    p_rt, p_ev = cdir / f"runtime_{port}.json", cdir / f"eval_{port}.json"
    p_rt.write_text(json.dumps(rt, indent=1))
    p_ev.write_text(json.dumps(ev, indent=1))
    return p_rt, p_ev


def ensure_tree(cases):
    for s, h in cases:
        d = TREE / s / h
        d.mkdir(parents=True, exist_ok=True)
        try:
            (d / "case").symlink_to(HF / s / h / "case")
        except FileExistsError:  # made earlier or by a concurrent arm
            pass


def env_for(arm, k, run, s, h, p_rt, p_ev, temperature):
    tag = f"__{arm}{'' if arm in ('llm', 's1_tool') else f'_k{k}'}{'' if temperature == 'default' else '_t' + temperature}__run{run}"
    env = {kk: v for kk, v in os.environ.items() if kk not in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
    env.update({
        "PYTHONPATH": os.pathsep.join([str(CE), str(REPO)]),
        "CLINENV_CASES_DIR": str(TREE), "BENCHMARK_RUNTIME_CONFIG": str(p_rt), "BENCHMARK_EVAL_CONFIG": str(p_ev),
        "S1CLIN_ARM": arm, "S1CLIN_K": str(k), "S1CLIN_RUN_TAG": tag, "S1CLIN_MUT_TEMPERATURE": temperature,
        "S1CLIN_SEED": f"{s}|{h}|run{run}", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "S1CLIN_DECISIONS_LOG": str(TREE / s / h / "runs" / (MODEL + tag) / "decisions.jsonl"),
    })
    return env, MODEL + tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["llm", "clef", "llm_select", "llm_log", "s1_tool"])
    ap.add_argument("--cases", default=str(DATA / "eval_subset_paper7.txt"), help="file with 'subject hadm' lines")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temperature", default="default", help="'default' (model generation config) or a number")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--llm-url", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--no-score", action="store_true")
    a = ap.parse_args()
    disabled = REPO / "logs" / "disabled_arms"  # arms dropped from the online design (docs/full_experiment.md)
    if disabled.exists() and a.arm in disabled.read_text().split():
        print(f"arm {a.arm} is disabled (logs/disabled_arms); nothing to do", flush=True)
        return
    os.umask(0o077)

    cases = [tuple(l.split()) for l in Path(a.cases).read_text().splitlines() if l.strip()]
    if a.limit:
        cases = cases[:a.limit]
    ensure_tree(cases)
    configs = {url: write_configs(url) for url, _ in parse_pool(a.llm_url)}
    p_rt, p_ev = next(iter(configs.values()))
    _, slug = env_for(a.arm, a.k, a.run, "x", "x", p_rt, p_ev, a.temperature)
    log_dir = TREE / "_logs" / slug
    log_dir.mkdir(parents=True, exist_ok=True)

    def score(s, h, env, slug):  # ClinEnv's own scorer for one case (outcome + process scorers)
        if (TREE / s / h / "evals" / slug / "latest.json").exists():
            return 0
        with open(log_dir / f"{s}_{h}.eval.log", "w") as log:
            return subprocess.run([sys.executable, "-m", "evaluation.main", "--subject-id", s, "--hadm-id", h,
                                   "--model", slug, "--process", "--workers", "1", "--quiet"], cwd=CE, env=env,
                                  stdout=log, stderr=subprocess.STDOUT).returncode

    def one(sh):
        s, h = sh
        url = route(a.llm_url, f"{s}|{h}")
        env, slug = env_for(a.arm, a.k, a.run, s, h, *configs[url], a.temperature)
        out = TREE / s / h / "runs" / slug
        if not claim(out):  # another driver is on it
            return
        try:
            if (out / "output.json").exists():
                if not a.no_score:
                    score(s, h, env, slug)
                return
            if a.score_only:
                return
            if out.exists():  # unfinished earlier attempt: decisions.jsonl is append-only, start clean
                subprocess.run(["rm", "-rf", str(out)])
            out.mkdir(parents=True)
            (out / "s1clin_meta.json").write_text(json.dumps({
                "llm_url": url, "llm_server": os.environ.get("S1CLIN_LLM_SERVER"),
                "host": f"{os.uname().nodename} job {os.environ.get('SLURM_JOB_ID')}"}))
            t0 = time.time()
            with open(log_dir / f"{s}_{h}.log", "w") as log:
                rc = subprocess.run([sys.executable, "-m", "env.main", "--subject-id", s, "--hadm-id", h, "--model",
                                     MODEL, "--mode", "interactive", "--quiet"], cwd=CE, env=env, stdout=log,
                                    stderr=subprocess.STDOUT).returncode
            erc = score(s, h, env, slug) if rc == 0 and not a.no_score else None
            print(time.strftime("%H:%M:%S"), s, h, "rc", rc, "eval", erc, f"{time.time() - t0:.0f}s",
                  url.split(":")[-1], flush=True)
        finally:
            release(out)

    todo = cases if a.score_only else [c for c in cases if not (TREE / c[0] / c[1] / "evals" / env_for(
        a.arm, a.k, a.run, *c, p_rt, p_ev, a.temperature)[1] / "latest.json").exists()]
    print(f"{slug}: {len(todo)}/{len(cases)} cases to run or score, {a.workers} workers", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, todo))


if __name__ == "__main__":
    main()
