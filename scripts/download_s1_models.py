"""Download the alternative System One deciders (weights only stay in the 032p HF cache)."""
import sys
from huggingface_hub import snapshot_download

MODELS = [("vllm-sr/d3", "v3.1.0"), ("perplexity-ai/pplx-decider-v1.2", None), ("Mapika/decider-31b", None),
          ("mukaddamzaid/amos-1-large", None), ("PNPM244/linne-oak-1-0", None)]
for rid, rev in MODELS:
    if len(sys.argv) > 1 and rid not in sys.argv[1:]:
        continue
    p = snapshot_download(rid, revision=rev, max_workers=16, ignore_patterns=["assets/*", "*.mp4", "*.png", "*.jpg"])
    print("done", rid, rev, p, flush=True)
