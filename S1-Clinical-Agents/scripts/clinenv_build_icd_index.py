"""Build ClinEnv's ICD embedding indexes on a GPU (the in-repo builder runs bge-large on CPU, about 4.5 h).

The vocabularies are public (pyhealth ICD9CM / ICD10CM / ICD9PROC / ICD10PROC); no patient data is involved.
Output is exactly what evaluation/scorers/icd_scoring._load_index caches:
  third_party/clinenv/evaluation/cache/<VOCAB>_BAAI--bge-large-en-v1.5_index.pkl = (codes, L2-normalised float32)
bge-large-en-v1.5 in sentence-transformers is CLS pooling + normalisation (max 512 tokens); we compute the same with
plain transformers so a CUDA environment without sentence-transformers can do it.

  1) .venv/bin/python scripts/clinenv_build_icd_index.py dump  OUT_DIR          (pyhealth -> codes/descs JSON)
  2) <cuda python>   scripts/clinenv_build_icd_index.py embed OUT_DIR          (GPU -> .npy)
  3) .venv/bin/python scripts/clinenv_build_icd_index.py write OUT_DIR          (pickle into the ClinEnv cache,
                                                                               checks 64 rows against CPU s-t)
"""

import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "third_party/clinenv/evaluation/cache"
VOCABS = ["ICD9CM", "ICD10CM", "ICD9PROC", "ICD10PROC"]
MODEL = "BAAI/bge-large-en-v1.5"
SNAP = "/ocean/projects/cis260032p/yliy/huggingface_cache/hub/models--BAAI--bge-large-en-v1.5/snapshots/" \
       "d4aa6901d3a41ba39fb536a557fa166f842b0e09"


def dump(out):
    from pyhealth.medcode import InnerMap
    for v in VOCABS:
        icd = InnerMap.load(v)
        codes, descs = [], []
        for code in icd.graph.nodes():  # same filter as icd_scoring._load_index (S1CLIN PATCH: skip range nodes)
            try:
                desc = icd.lookup(code)
            except KeyError:
                continue
            if desc and desc.strip():
                codes.append(code)
                descs.append(desc)
        (out / f"{v}.json").write_text(json.dumps({"codes": codes, "descs": descs}))
        print(v, len(codes), flush=True)


def embed(out):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAP)
    model = AutoModel.from_pretrained(SNAP, torch_dtype=torch.float32).cuda().eval()
    for v in VOCABS:
        descs = json.loads((out / f"{v}.json").read_text())["descs"]
        embs = []
        with torch.inference_mode():
            for i in range(0, len(descs), 512):
                b = tok(descs[i:i + 512], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
                cls = model(**b).last_hidden_state[:, 0]
                embs.append(torch.nn.functional.normalize(cls, dim=-1).float().cpu().numpy())
        np.save(out / f"{v}.npy", np.vstack(embs).astype(np.float32))
        print(v, len(descs), flush=True)


def write(out):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(SNAP, device="cpu")
    CACHE.mkdir(parents=True, exist_ok=True)
    for v in VOCABS:
        d = json.loads((out / f"{v}.json").read_text())
        e = np.load(out / f"{v}.npy")
        e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)
        assert len(d["codes"]) == e.shape[0]
        idx = np.linspace(0, len(d["descs"]) - 1, 64).astype(int)
        ref = st.encode([d["descs"][i] for i in idx], normalize_embeddings=True, convert_to_numpy=True)
        cos = (ref * e[idx]).sum(1)
        print(v, len(d["codes"]), "min cosine vs CPU sentence-transformers", round(float(cos.min()), 5), flush=True)
        assert cos.min() > 0.999, f"{v}: GPU embeddings disagree with sentence-transformers"
        with open(CACHE / f"{v}_{MODEL.replace('/', '--')}_index.pkl", "wb") as f:
            pickle.dump((d["codes"], e), f)


if __name__ == "__main__":
    mode, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    {"dump": dump, "embed": embed, "write": write}[mode](out)
