"""Kev-27B adapter: jaredpalmer/kev-27b v2 (full-weight Qwen3.8-27B backbone + pointer head) with a fresh LoRA.

kev.train --init_from cannot put a new LoRA on the full-weight v2 checkpoint (Checkpoint.warm_start requires matching
lora/weights fields), so the model is built exactly as kev.serve loads it for inference (kev.checkpoint.Checkpoint.load:
full v2 backbone in bf16, released pointer head; without the serving-only fused kernels and CUDA graphs) and the shared
LoRA (common.lora_config) is attached to the backbone with peft. Everything on the request -> logits path is Kev's own
library code: kev.api.to_record (request -> record, options in presentation order), kev.model.admit (serving encoder and
limits), kev.model.rows_of + DecisionModel._rows_hidden (the row form kev.train uses for hybrid backbones; one question,
so one causal row = state + branch) and kev.model.PointerHead. The head runs in fp32 as in serving (autocast off).

Needs the Kev-27B environment (jev_open/kev_v2/.venv) with jev_open/kev_v2 importable.
"""

import json
import sys
from pathlib import Path

import torch

from .common import Adapter, lora_config, lora_module_names

KEV_REPO = Path("/ocean/projects/cis260032p/yliy/research/jev/jev_open/kev_v2")
if str(KEV_REPO) not in sys.path:
    sys.path.insert(0, str(KEV_REPO))

HUB_ID = "jaredpalmer/kev-27b"
REVISION = "af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf"


class KevAdapter(Adapter):
    system = "kev27b"

    def __init__(self, device="cuda"):
        super().__init__(device)
        from kev.checkpoint import Checkpoint
        self.ck = Checkpoint(f"{HUB_ID}@{REVISION}")
        self.serving_temperature = float(self.ck.meta.temperature)   # 1.3195, applied by kev.serve
        self.tok = None
        self.model = None

    # --- request -> encoding (no GPU needed) ---------------------------------------------------------------------------

    def tokenizer(self):
        if self.tok is None:
            from kev.model import load_tokenizer
            self.tok = load_tokenizer(self.ck.meta.base, revision=self.ck.meta.base_revision)
        return self.tok

    def encode(self, item):
        """-> (encoding, permutation): kev.serve's path (api.to_record + model.admit, strict serving limits) and, for
        each canonical option item["options"][i], its index in the request's presentation order."""
        from kev.api import SystemOneRequest, to_record
        from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE, encode
        rec, meta = to_record(SystemOneRequest(**item["request"]))
        assert len(meta) == 1 and meta[0]["id"] == item["qid"] and meta[0]["type"] == "choice"
        # == kev.model.admit(model, tok, rec) with option_isolation False (the v2 checkpoint's setting); usable without a model
        assert not self.ck.meta.option_isolation
        enc = encode(self.tokenizer(), rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH, strict=True)
        pos = {k: j for j, k in enumerate(meta[0]["keys"])}
        return enc, [pos[o] for o in item["options"]]

    def n_tokens(self, item):
        return len(self.encode(item)[0]["ids"])

    # --- model ------------------------------------------------------------------------------------------------------

    def load(self):
        from kev.checkpoint import LoadOptions
        from peft import get_peft_model
        # kev.serve's loader in bf16; cuda_graphs/fused stay None (off): they are inference-only rewrites
        self.tok, m = self.ck.load(self.device, LoadOptions(dtype=torch.bfloat16))
        assert m.hybrid and not m.option_isolation and m.graphs is None
        for p in m.lm.parameters():
            p.requires_grad_(False)
        torch.manual_seed(0)   # LoRA A init (B = 0); common.train reseeds before training
        m.lm = get_peft_model(m.lm, lora_config("FEATURE_EXTRACTION"))   # task type as Kev's own LoRA (DecisionModel.__init__)
        m.lm.config.use_cache = False
        m.head.temperature = 1.0      # raw logits; serving T is applied by the caller where needed
        m.head.float()
        for p in m.head.parameters():
            p.requires_grad_(True)
        self.model = m
        return self

    def param_groups(self):
        head = list(self.model.head.parameters())
        hid = {id(p) for p in head}
        lora = [p for n, p in self.model.lm.named_parameters() if p.requires_grad and id(p) not in hid]
        assert all("lora_" in n for n, p in self.model.lm.named_parameters() if p.requires_grad)
        return {"lora": lora, "head": head}

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def enable_gradient_checkpointing(self):
        # as kev.train --checkpointing 1
        self.model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def option_logits(self, item):
        from kev.model import rows_of
        m = self.model
        enc, perm = self.encode(item)
        S, Sp, rows = rows_of(enc)
        (r,) = rows
        h = m._rows_hidden([(S + r["ids"], Sp + r["pos"])])[0]          # [L, d] fp32 (DecisionModel row form)
        d, opts = len(S) + r["decide"], [len(S) + o for o in r["opts"]]
        with torch.autocast("cuda", enabled=False):
            z = m.head(h[d].float(), h[torch.tensor(opts, device=h.device)].float())   # presentation order
        return z[torch.tensor(perm, device=z.device)]                                    # canonical order

    # --- persistence ------------------------------------------------------------------------------------------------

    def save(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.lm.save_pretrained(str(out / "lora"))
        torch.save(self.model.head.state_dict(), out / "head.pt")
        (out / "adapter_meta.json").write_text(json.dumps(self.describe(), indent=1))

    def load_trained(self, out_dir):
        from peft import load_peft_weights, set_peft_model_state_dict
        out = Path(out_dir)
        if self.model is None:
            self.load()
        w = load_peft_weights(str(out / "lora"), device="cpu")
        res = set_peft_model_state_dict(self.model.lm, w)
        assert not res.unexpected_keys, res.unexpected_keys[:3]
        self.model.head.load_state_dict(torch.load(out / "head.pt", map_location="cpu"))
        return self

    def describe(self):
        d = {"system": self.system, "hub_id": HUB_ID, "revision": REVISION, "base": self.ck.meta.base,
             "base_revision": self.ck.meta.base_revision, "head_dim": self.ck.meta.head_dim,
             "serving_temperature": self.serving_temperature, "kev_code": str(KEV_REPO)}
        if self.model is not None:
            g = self.param_groups()
            names = lora_module_names(self.model.lm)
            d.update({"lora_params": sum(p.numel() for p in g["lora"]), "head_params": sum(p.numel() for p in g["head"]),
                      "lora_modules": len(names),
                      "lora_module_suffixes": sorted({n.rsplit(".", 1)[-1] for n in names}),
                      "lora_module_example": names[:3]})
        return d
