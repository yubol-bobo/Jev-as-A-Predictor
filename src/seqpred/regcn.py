"""RE-GCN (Li et al., SIGIR 2021) in plain PyTorch (no DGL), following the reference implementation
github.com/Lee-zix/RE-GCN (src/rrgcn.py, rgcn/layers.py UnionRGCNLayer, src/decoder.py ConvTransE):

  * per-snapshot graph over all entities with inverse edges (relation r + R), norm = 1 / in-degree
  * "uvrgcn" layer: msg = (h_src + rel_emb[type]) W_neighbor, summed and normalized at dst, plus a
    self-loop (loop_weight if the node has in-edges at this snapshot, else evolve_loop_weight),
    rrelu (eval-mode slope, as F.rrelu's default), dropout
  * relation evolution: GRUCell([emb_rel ; mean of entity states attached to r], h0), L2-normalized
  * entity evolution: time gate h <- g * normalize(RGCN(h)) + (1 - g) * h, g = sigmoid(h W + b)
  * ConvTransE decoder on (tanh(h_s), h0_r), scores against tanh(h) of all entities
  * training: one snapshot (day) per step, CE over all entities for every fact and its inverse,
    Adam(lr 1e-3, wd 1e-5), grad-norm clip 1.0; static graph off, entity prediction only.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Snapshot:
    def __init__(self, triples, n_ent, n_rel):
        s, r, o = (torch.as_tensor(x, dtype=torch.long) for x in np.asarray(triples).T)
        self.src = torch.cat([s, o])
        self.dst = torch.cat([o, s])
        self.rel = torch.cat([r, r + n_rel])
        deg = torch.bincount(self.dst, minlength=n_ent).float()
        self.in_idx = torch.nonzero(deg > 0).squeeze(1)
        self.norm = (1.0 / deg.clamp(min=1)).unsqueeze(1)
        # relation -> entities attached to it (both endpoints, for r and its inverse), unique pairs
        pairs = torch.stack([torch.cat([r, r, r + n_rel, r + n_rel]), torch.cat([s, o, s, o])], 1).unique(dim=0)
        self.pair_rel, self.pair_ent = pairs[:, 0], pairs[:, 1]
        self.rel_count = torch.bincount(self.pair_rel, minlength=2 * n_rel).float().clamp(min=1).unsqueeze(1)


class UnionRGCNLayer(nn.Module):
    def __init__(self, h, dropout):
        super().__init__()
        gain = nn.init.calculate_gain("relu")
        self.weight_neighbor = nn.Parameter(nn.init.xavier_uniform_(torch.empty(h, h), gain=gain))
        self.loop_weight = nn.Parameter(nn.init.xavier_uniform_(torch.empty(h, h), gain=gain))
        self.evolve_loop_weight = nn.Parameter(nn.init.xavier_uniform_(torch.empty(h, h), gain=gain))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, h, rel_emb):
        msg = (h[g.src] + rel_emb[g.rel]) @ self.weight_neighbor
        agg = torch.zeros_like(h).index_add_(0, g.dst, msg) * g.norm
        loop = (h @ self.evolve_loop_weight).index_copy(0, g.in_idx, h[g.in_idx] @ self.loop_weight)
        return self.dropout(F.rrelu(agg + loop))


class ConvTransE(nn.Module):
    def __init__(self, h, input_dropout=0.2, hidden_dropout=0.2, feat_dropout=0.2, channels=50, kernel=3):
        super().__init__()
        self.inp_drop, self.hidden_drop = nn.Dropout(input_dropout), nn.Dropout(hidden_dropout)
        self.feature_map_drop = nn.Dropout(feat_dropout)
        self.conv1 = nn.Conv1d(2, channels, kernel, padding=kernel // 2)
        self.bn0, self.bn1, self.bn2 = nn.BatchNorm1d(2), nn.BatchNorm1d(channels), nn.BatchNorm1d(h)
        self.fc = nn.Linear(h * channels, h)

    def forward(self, ent, rel, subj, rels):
        e_all = torch.tanh(ent)
        x = torch.stack([e_all[subj], rel[rels]], 1)  # B x 2 x h
        x = self.inp_drop(self.bn0(x))
        x = self.feature_map_drop(F.relu(self.bn1(self.conv1(x))))
        x = self.hidden_drop(self.fc(x.flatten(1)))
        if x.shape[0] > 1 or not self.training:
            x = self.bn2(x)
        return F.relu(x) @ e_all.T


class REGCN(nn.Module):
    def __init__(self, n_ent, n_rel, h=200, n_layers=2, dropout=0.2, input_dropout=0.2, hidden_dropout=0.2,
                 feat_dropout=0.2, layer_norm=True):
        super().__init__()
        self.n_ent, self.n_rel, self.layer_norm = n_ent, n_rel, layer_norm
        self.dynamic_emb = nn.Parameter(torch.randn(n_ent, h))
        self.emb_rel = nn.Parameter(nn.init.xavier_normal_(torch.empty(2 * n_rel, h)))
        self.layers = nn.ModuleList(UnionRGCNLayer(h, dropout) for _ in range(n_layers))
        self.time_gate_weight = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(h, h), gain=nn.init.calculate_gain("relu")))
        self.time_gate_bias = nn.Parameter(torch.zeros(h))
        self.relation_cell = nn.GRUCell(2 * h, h)
        self.decoder = ConvTransE(h, input_dropout, hidden_dropout, feat_dropout)

    def _norm(self, x):
        return F.normalize(x) if self.layer_norm else x

    def evolve(self, snaps):
        h = self._norm(self.dynamic_emb)
        h0 = self.emb_rel
        for g in snaps:
            x = torch.zeros_like(self.emb_rel).index_add_(0, g.pair_rel, h[g.pair_ent]) / g.rel_count
            h0 = self._norm(self.relation_cell(torch.cat([self.emb_rel, x], 1), h0))
            cur = h
            for layer in self.layers:
                cur = layer(g, cur, h0)
            cur = self._norm(cur)
            gate = torch.sigmoid(h @ self.time_gate_weight + self.time_gate_bias)
            h = gate * cur + (1 - gate) * h
        return self._norm(h), h0

    def score(self, snaps, subj, rels):
        """Scores over all entities for queries (subj, rels); rels >= n_rel are inverse relations."""
        ent, rel = self.evolve(snaps)
        return self.decoder(ent, rel, subj, rels)


def with_inverse(triples, n_rel):
    t = torch.as_tensor(np.asarray(triples), dtype=torch.long)
    inv = torch.stack([t[:, 2], t[:, 1] + n_rel, t[:, 0]], 1)
    return torch.cat([t, inv])


def ranks(scores, target, filt=None):
    """1-based rank of target (number of strictly higher scores + 1); `filt` = list of index tensors to mask."""
    scores = scores.clone()
    if filt is not None:
        gold = scores[torch.arange(len(target)), target].clone()
        for i, idx in enumerate(filt):
            if len(idx):
                scores[i, idx] = -math.inf
        scores[torch.arange(len(target)), target] = gold
    g = scores[torch.arange(len(target)), target].unsqueeze(1)
    return (scores > g).sum(1) + 1
