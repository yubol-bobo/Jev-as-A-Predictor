"""Per-task LoRA finetuning of the System One decision models (design §5).

One shared recipe and training loop (common.py); each model contributes only an adapter (its loader, forward path and
decision head), registered in ADAPTERS:

  kev27b        seqpred.ft.kev.KevAdapter      jaredpalmer/kev-27b v2 + pointer head
  pplx_decider  (to be added)                  perplexity-ai/pplx-decider-v1.1-27b + readout head
  clef27b       (to be added)                  Cloudflare Clef + joint schema head

Entry point: scripts/finetune.py.
"""

import importlib

ADAPTERS = {
    "kev27b": "seqpred.ft.kev:KevAdapter",
}


def get_adapter(system):
    mod, cls = ADAPTERS[system].split(":")
    return getattr(importlib.import_module(mod), cls)
