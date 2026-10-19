"""Run PhysicianBench tasks with the leaderboard harness on local models, for one arm.

Arms (docs/physicianbench_decider.md):
  llm         MiniAgent + local Qwen3.8-27B, one sample per step (leaderboard setting)
  clef        each step: Qwen samples K candidate turns, Clef-27B chooses
  llm_select  each step: Qwen samples K candidate turns, Qwen chooses (online arm dropped 2026-10-19; offline instead)
  llm_log     each step: Qwen samples K candidate turns, the first is executed (= `llm`); all K and the message
              history are logged for the offline decision evaluation
  s1_tool     each step: Clef chooses the tool (or FINISH) from the full menu, Qwen writes the call
  llm_tool    control for s1_tool: the same one-tool-per-turn restriction, the tool chosen by Qwen's own turn

Per task, this mirrors third_party/physicianbench/scripts/run_task.py: a fresh FHIR server (Apptainer, own copy of
the pristine database), the agent in a fresh process, the task's pytest graders (local judge), metadata.json.
Job directories hold patient data, so they live under data/physicianbench/jobs/ (mode 700), never in git.

  python -m src.s1clin.pb_run --arm clef --tasks aortic_aneurysm_cad,iron_deficiency_anemia --run 1 --workers 2

Several drivers (other processes or other SLURM nodes) may work on the same arm and run: each task is claimed first
(src/s1clin/claim.py). --llm-url may list several identical Qwen replicas, "URL*weight,..."; each task is routed to
one of them by its name, so a task meets the same replica in every arm and run.
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
from queue import Queue

from src.s1clin.claim import claim, release, route

REPO = Path(__file__).resolve().parents[2]
PB = REPO / "third_party" / "physicianbench"
GUARD_SITE = REPO / "src" / "s1clin" / "guard_site"
JOBS = Path("/ocean/projects/cis260032p/yliy/data/physicianbench/jobs")
FHIR_UP = REPO / "scripts" / "pb_fhir_up.sh"

ENV = {
    "LLM_JUDGE_BACKEND": "openai",
    "LLM_JUDGE_MODEL": "qwen3.8-27b",
    "S1CLIN_JUDGE_THINKING": "0",
    "S1CLIN_QWEN_THINKING": "1",           # leaderboard: reasoning effort "high" -> Qwen thinking on
    "S1CLIN_MAX_COMPLETION_TOKENS": "16384",
}


def child_env(llm_url):
    env = {k: v for k, v in os.environ.items() if k not in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY")}
    env.update(ENV)
    env["OPENAI_BASE_URL"] = llm_url  # agent, proposer and judge all use this replica (scripts/vllm_qwen_up.sh)
    env["PYTHONPATH"] = os.pathsep.join([str(GUARD_SITE), str(PB), str(REPO)])
    return env


# ---------------------------------------------------------------- one task (runs in its own process)

def run_one(task: str, arm: str, k: int, job_dir: Path, port: int, max_steps: int) -> int:
    sys.path[:0] = [str(PB), str(REPO)]
    from scripts.run_task import prepare_workspace, run_evaluation, wait_for_fhir
    from scripts.job_manager import parse_pytest_results, write_metadata
    from agent.llm_client import LLMClient
    from agent.mini_agent import MiniAgent
    from agent.tool_registry import ToolRegistry, register_all_tools
    from agent.trajectory import TrajectoryLogger
    from src.s1clin.pb_decider import (TOOL_QUESTION, ClefDecider, DeciderClient, FirstDecider, LLMSelectDecider,
                                       LLMToolClient, ToolChoiceClient)

    task_dir = PB / "tasks" / "v1" / task
    job_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"pb_fhir_{port}"
    pgid = int(subprocess.check_output(["bash", str(FHIR_UP), str(port), str(run_dir)], text=True).split()[-1])
    fhir_url = f"http://127.0.0.1:{port}/fhir"
    t0 = time.time()
    try:
        if not wait_for_fhir(fhir_url, timeout=300):
            raise RuntimeError("FHIR server did not start")
        t_fhir = time.time() - t0
        workspace = prepare_workspace(job_dir, task_dir)
        instruction = (task_dir / "instruction.md").read_text().replace("/workspace/", f"{workspace}/")
        instruction += (f"\n\n## Working Directory\n\nYour working directory is: {workspace}\n"
                        f"Output files should be saved under: {workspace / 'output'}/\n")
        os.environ["FHIR_BASE_URL"] = fhir_url + "/"
        log_dir = job_dir / "logs" / "agent"
        log_dir.mkdir(parents=True, exist_ok=True)
        registry = ToolRegistry()
        register_all_tools(registry)
        base = LLMClient(model_id="qwen3.8-27b")
        seed = f"{task}|{job_dir.parent.name}"  # per (task, run)
        if arm == "llm":
            client = base
        elif arm == "s1_tool":
            client = ToolChoiceClient(base, ClefDecider(question=TOOL_QUESTION), log_dir / "decisions.jsonl", seed)
        elif arm == "llm_tool":
            client = LLMToolClient(base, log_dir / "decisions.jsonl", seed)
        else:
            decider = {"clef": ClefDecider, "llm_log": FirstDecider}.get(arm, lambda: LLMSelectDecider(base, "high"))()
            client = DeciderClient(base, decider, k, log_dir / "decisions.jsonl", seed=seed)
        agent = MiniAgent(client=client, registry=registry, trajectory=TrajectoryLogger(log_dir / "trajectory.log"),
                          max_steps=max_steps, temperature=None, parallel_tool_calls=True, reasoning_effort="high")
        t1 = time.time()
        try:
            result = agent.run(instruction)
            (log_dir / "stdout.txt").write_text(result)
        except Exception as e:
            (log_dir / "stderr.txt").write_text(repr(e))
        t_agent = time.time() - t1
        t2 = time.time()
        success = run_evaluation(task_dir, job_dir, fhir_url)
        t_eval = time.time() - t2
    finally:
        try:
            os.killpg(pgid, 15)
        except ProcessLookupError:
            pass
        subprocess.run(["rm", "-rf", str(run_dir)])
    pytest_file = job_dir / "logs" / "verifier" / "pytest_output.txt"
    tests = parse_pytest_results(pytest_file.read_text()) if pytest_file.exists() else {}
    write_metadata(job_dir, model=client.model_id, task=task, arm=arm, k=k, max_steps=max_steps, temperature=None,
                   reasoning_effort="high(qwen thinking)", fhir_url=fhir_url, success=success, test_results=tests,
                   judge="qwen3.8-27b (local, thinking off)", llm_url=os.environ["OPENAI_BASE_URL"],
                   decider=os.environ.get("S1CLIN_DECIDER", "clef") if arm in ("clef", "s1_tool") else None,
                   llm_server=os.environ.get("S1CLIN_LLM_SERVER",
                                             "scripts/vllm_qwen_up.sh (bf16, 128k, decode CUDA graphs, MTP-2)"),
                   host=f"{os.uname().nodename} job {os.environ.get('SLURM_JOB_ID')}",
                   wall_s={"fhir": round(t_fhir, 1), "agent": round(t_agent, 1), "eval": round(t_eval, 1)})
    return 0 if success else 1


# ---------------------------------------------------------------- batch driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["llm", "clef", "llm_select", "llm_log", "s1_tool", "llm_tool"])
    ap.add_argument("--tasks", required=True, help="comma list, or @file with one task per line, or 'all'")
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--port-base", type=int, default=18100)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--llm-url", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--decider", default="clef", help="System One decider of the clef / s1_tool arms "
                    "(src/s1clin/pb_decider.DECIDER_URLS); other than clef, the job tag gets the suffix __<decider>")
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--port", type=int, help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.tasks == "all":
        tasks = sorted(p.name for p in (PB / "tasks" / "v1").iterdir() if p.is_dir())
    elif a.tasks.startswith("@"):
        tasks = [t.strip() for t in Path(a.tasks[1:]).read_text().split() if t.strip()]
    else:
        tasks = a.tasks.split(",")
    arm_tag = a.arm if a.arm in ("llm", "s1_tool", "llm_tool") else f"{a.arm}_k{a.k}"
    if a.arm in ("clef", "s1_tool") and a.decider != "clef":
        arm_tag += f"__{a.decider}"
    os.environ["S1CLIN_DECIDER"] = a.decider  # inherited by the per-task processes (child_env copies os.environ)

    def job_of(t):
        return JOBS / arm_tag / f"run{a.run}" / t

    if a.one:
        sys.exit(run_one(tasks[0], a.arm, a.k, job_of(tasks[0]), a.port, a.max_steps))

    disabled = REPO / "logs" / "disabled_arms"  # arms dropped from the online design (docs/full_experiment.md)
    if disabled.exists() and a.arm in disabled.read_text().split():
        print(f"arm {a.arm} is disabled (logs/disabled_arms); nothing to do", flush=True)
        return
    os.umask(0o077)
    todo = [t for t in tasks if not (job_of(t) / "metadata.json").exists()]
    print(f"{arm_tag} run{a.run}: {len(todo)}/{len(tasks)} tasks to do, {a.workers} workers", flush=True)

    ports = Queue()
    for i in range(a.workers):
        ports.put(a.port_base + i)

    def work(t):
        jd = job_of(t)
        if not claim(jd):  # another driver is on it
            return
        try:
            if (jd / "metadata.json").exists():  # finished by another driver meanwhile
                return
            if jd.exists():  # unfinished earlier attempt: the trajectory and decision logs are append-only, start clean
                subprocess.run(["rm", "-rf", str(jd)])
            jd.mkdir(parents=True, exist_ok=True)
            port, url = ports.get(), route(a.llm_url, t)
            cmd = [sys.executable, "-m", "src.s1clin.pb_run", "--one", "--arm", a.arm, "--tasks", t, "--run",
                   str(a.run), "--k", str(a.k), "--port", str(port), "--max-steps", str(a.max_steps),
                   "--decider", a.decider]
            t0 = time.time()
            try:
                with open(jd / "runner.log", "w") as log:
                    rc = subprocess.run(cmd, cwd=REPO, env=child_env(url), stdout=log,
                                        stderr=subprocess.STDOUT).returncode
            finally:
                ports.put(port)
            md = jd / "metadata.json"
            res = json.loads(md.read_text()).get("test_results", {}) if md.exists() else {}
            print(time.strftime("%H:%M:%S"), t, "rc", rc, f"{time.time() - t0:.0f}s", url.split(":")[-1],
                  json.dumps(res)[:200], flush=True)
        finally:
            release(jd)

    with ThreadPoolExecutor(a.workers) as ex:  # each worker holds one FHIR port while it runs a task
        list(ex.map(work, todo))


if __name__ == "__main__":
    main()
