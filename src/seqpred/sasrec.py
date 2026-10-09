"""SASRec (Kang & McAuley, ICDM 2018) with full-softmax cross-entropy.

Items are integers 1..n_items (0 = padding). Training data are (sequence, loss_from)
pairs: the model predicts seq[i+1] from seq[:i+1] for every i >= loss_from - 1, so
  * full-data training:   (whole training sequence, 1)        -> loss at every position
  * matched-N training:   (history + [target], len(history))  -> loss on the target only
Cross-entropy over all items is used instead of the original 1-negative BCE; it is the
stronger and now-standard SASRec variant (Klenitskiy & Vasilev, RecSys 2023).
Candidate scores are turned into probabilities with a softmax whose temperature is
fitted on dev NLL (`fit_temperature`).
"""

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SASRec(nn.Module):
    def __init__(self, n_items, max_len=200, hidden=64, blocks=2, heads=1, dropout=0.2):
        super().__init__()
        self.max_len = max_len
        self.item_emb = nn.Embedding(n_items + 1, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden, dropout, batch_first=True,
                                           norm_first=True, activation="relu")
        self.encoder = nn.TransformerEncoder(layer, blocks, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        # Small init (as in RecBole): nn.Embedding's default N(0, 1), times sqrt(hidden) and tied to
        # the output layer, gives initial logits with std ~ sqrt(hidden) (initial CE ~56 vs ln(n) ~8)
        # and training never recovers well.
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def forward(self, seqs):  # seqs: [B, L] left-padded
        L = seqs.size(1)
        pos = torch.arange(L, device=seqs.device).unsqueeze(0)
        x = self.item_emb(seqs) * math.sqrt(self.item_emb.embedding_dim) + self.pos_emb(pos)
        x = self.drop(x) * (seqs > 0).unsqueeze(-1)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seqs.device), 1)
        h = self.encoder(x, mask=causal, src_key_padding_mask=(seqs == 0))
        return self.norm(h)

    def logits(self, h):
        return h @ self.item_emb.weight.T  # [.., n_items + 1]


def _left_pad(seqs, L):
    out = np.zeros((len(seqs), L), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[-L:]
        if s:
            out[i, -len(s):] = s
    return out


def _batches(examples, max_len, batch_size, rng):
    order = list(range(len(examples)))
    rng.shuffle(order)
    for b in range(0, len(order), batch_size):
        chunk = [examples[i] for i in order[b:b + batch_size]]
        inp, tgt = [], []
        for seq, loss_from in chunk:
            off = max(0, len(seq) - (max_len + 1))  # keep the last max_len + 1 tokens
            s = seq[off:]
            x, y = s[:-1], list(s[1:])  # y[i] is original position i + 1 + off
            first = max(loss_from - 1 - off, 0)  # supervise original positions >= loss_from
            y = [0] * first + y[first:]  # 0 = not supervised
            inp.append(x)
            tgt.append(y)
        yield _left_pad(inp, max_len), _left_pad(tgt, max_len)


def train(examples, n_items, *, max_len=200, hidden=64, blocks=2, heads=1, dropout=0.2, lr=1e-3,
          batch_size=128, max_epochs=200, eval_fn=None, eval_every=5, patience=4, device="cpu",
          seed=0, log=print):
    """Train on (sequence, loss_from) examples; early-stop on eval_fn(model) (higher is better)."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = SASRec(n_items, max_len, hidden, blocks, heads, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    best, best_state, bad = -float("inf"), None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in _batches(examples, max_len, batch_size, rng):
            x, y = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
            h = model(x)
            sel = y > 0
            if not sel.any():
                continue
            loss = F.cross_entropy(model.logits(h[sel]), y[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n = total + loss.item() * sel.sum().item(), n + sel.sum().item()
        if eval_fn is not None and (epoch % eval_every == 0 or epoch == max_epochs):
            score = eval_fn(model)
            log(f"epoch {epoch} loss {total / max(n, 1):.4f} dev {score:.4f}")
            if score > best:
                best, bad = score, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def score_candidates(model, histories, candidates, device="cpu", batch_size=512):
    """Logits of each candidate list given each history. Returns list of np arrays."""
    model.eval()
    out = []
    for b in range(0, len(histories), batch_size):
        x = torch.as_tensor(_left_pad(histories[b:b + batch_size], model.max_len), device=device)
        h = model(x)[:, -1]  # last position
        full = model.logits(h).cpu().numpy()
        for row, cands in zip(full, candidates[b:b + batch_size]):
            out.append(row[np.asarray(cands)])
    return out


def softmax(z, T=1.0):
    z = np.asarray(z, dtype=float) / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def fit_temperature(logit_lists, golds):
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 121))
    nll = [np.mean([-np.log(softmax(z, T)[g] + 1e-12) for z, g in zip(logit_lists, golds)]) for T in grid]
    return float(grid[int(np.argmin(nll))])
