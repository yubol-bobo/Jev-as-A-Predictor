"""Loaded automatically by every Python process that has this directory on PYTHONPATH.

PhysicianBench runs its grader in a pytest subprocess. Putting this directory on PYTHONPATH installs the
localhost-only socket guard in the agent process and in every grader subprocess. The guard removes cloud API
keys, so afterwards we set a dummy OPENAI_API_KEY: the openai SDK needs a key, and the base URL must be local.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from src.s1clin import local_guard  # noqa: E402,F401  (installs on import)

local_guard.assert_local_url(os.environ.get("OPENAI_BASE_URL"), "OPENAI_BASE_URL")
os.environ["OPENAI_API_KEY"] = "EMPTY"
for k in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.pop(k, None)
