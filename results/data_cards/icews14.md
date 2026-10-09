# Data card — icews14 (ICEWS14 event forecasting, object prediction)

Builder: `scripts/build_icews14.py` (deterministic, base seed 20261009). Loader: `src/seqpred/icews14.py`.
Baselines: `scripts/run_icews14_recurrency.py`, `scripts/run_icews14_regcn.py` (+ `src/seqpred/regcn.py`).

## Which ICEWS14

Three different files are distributed as "ICEWS14". We use the split on which RE-GCN reported its ICEWS14 results
and which the recurrency baseline (Gastinger et al., 2024) uses:

| variant | entities / relations | train / valid / test | used |
|---|---|---|---|
| RE-GCN split: `nec-research/recurrency_baseline_tkg` `data/ICEWS14` = `Lee-zix/RE-GCN` `data-release.tar.gz` → `data/ICEWS14s` | 7,128 / 230 | 74,845 / 8,514 / 7,371 (RE-GCN paper Table 2) | **yes** |
| xERTE / TLogic split: `liu-yushan/TLogic` `data/icews14` | 7,128 / 230 | 63,685 / 13,823 / 13,222 | no (same 90,730 facts, different time cut) |
| RE-NET: `INK-USC/RE-Net` `data/ICEWS14` | 12,498 / 260 (stat.txt) | train / test only, no valid | no |

The RE-GCN release no longer ships a folder named `ICEWS14`; its `ICEWS14s` folder holds exactly the split in the
RE-GCN paper's ICEWS14 table (74,845 / 8,514 / 7,371). The recurrency-baseline repo's `ICEWS14` is the same data,
reformatted by its `reformat_icews14.py` to `t = (day - 1) * 24`. We verified: identical ids and (s, r, o) rows
line by line in all three splits, with `t_recurrency = (t_regcn - 1) * 24`; and the same 90,730 facts as the TLogic
files after mapping names and dates (only difference: `Xinjiang Uyghur` written without the space in the id files).

## Sources

Files actually read by the builder (downloaded to `data/external/icews14/dl/recurrency/`, gitignored; commit
`master` of https://github.com/nec-research/recurrency_baseline_tkg, raw URL prefix
`https://raw.githubusercontent.com/nec-research/recurrency_baseline_tkg/master/data/ICEWS14/`):

| file | sha256 |
|---|---|
| `train.txt` | `09d9ec8df4b779e9793229d240a79c762c5c6f7846b31ee5f2522413a6595d57` |
| `valid.txt` | `bb9462c02b6fb56f15822692ab9f80d782d6c2cc89e5ed7a83bae2c6022cab66` |
| `test.txt` | `a72ddd5cc3c86105d48b76620a1c08223fcdc43ef57d1eb9e10b27fd05b471ab` |
| `entity2id.txt` | `0251b09f1c2e56cdc5506f9d7f83e7a88d5d54c3df25609ae8ce4edb7fb3e573` |
| `relation2id.txt` | `32fcce0906d05205824f03e980981436a71a85feba72a817f5291e4afda9d04e` |

Cross-check copies (not read by the builder), also under `data/external/icews14/dl/`:

| file | URL | sha256 |
|---|---|---|
| `data-release.tar.gz` (RE-GCN, commit `bc64211`) | https://raw.githubusercontent.com/Lee-zix/RE-GCN/master/data-release.tar.gz | `00148c0deea7a9fe60c5759a6a419eca31480a2ca7fd3461aa7c7f9d1d1eab82` |
| ↳ `ICEWS14s/train.txt` / `valid.txt` / `test.txt` | (extracted) | `228da10c…f0b4` / `b046fe72…96` / `1ce0a75f…bd` |
| ↳ `ICEWS14s/entity2id.txt` / `relation2id.txt` | (extracted; byte-identical to the files above) | `0251b09f…67ef` / `32fcce09…d04e` |
| TLogic `icews14/{train,valid,test}.txt` (commit `9f6ffa3`) | https://raw.githubusercontent.com/liu-yushan/TLogic/main/data/icews14/ | `e66100f4…1a1a` / `ce4862c7…d2` / `7ecf0f15…25f6` |

RE-GCN reference source (`src/rrgcn.py`, `src/decoder.py`, `src/main.py`, `src/model.py`, `rgcn/layers.py`,
`rgcn/utils.py`) was downloaded to `dl/regcn_src/` and read only; it was **not run** (needs DGL, not installed).

## Timestamps

Files store `t` in hours, 24 per day, `t = 0` = 2014-01-01 (RE-GCN's own copy: days 1..365, 1 = 2014-01-01).
Every `t` is a multiple of 24. `day = t / 24` (0-based); date = 2014-01-01 + day. Checked against TLogic's
explicit date strings (all 90,730 facts match). Splits: train 2014-01-01 → 2014-10-31 (304 days),
valid 2014-11-01 → 2014-11-30 (30 days), test 2014-12-01 → 2014-12-31 (31 days). Every day has facts.

## Preprocessing

- Entity / relation names from `entity2id.txt` / `relation2id.txt` with `_` → space (e.g. `Citizen (Nigeria)`,
  `Make an appeal or request`); `XinjiangUyghur` → `Xinjiang Uyghur`. No duplicate display names (the
  ` [index]` disambiguation rule never fires).
- Item entity index = entity id + 1 (1..7,128); `vocab.json` has `n_items = 7128`, `index_to_name`,
  `relation_names` (relation id order).
- Queries: object prediction only, `(s, r, ?, t)` from one fact `(s, r, o, t)`. Facts of the source split are
  permuted with a seeded RNG and accepted in order while their `(s, day)` has < 3 accepted queries.
  test 1,000 from test.txt, dev 300 from valid.txt, train 10,000 from train.txt (stored in sampled order, so the
  first N lines are nested subsets 500 ⊂ 2,000 ⊂ 10,000).
- History (extrapolation, ground truth): every fact of train+valid+test with day < query day. Within a day, file
  order.
- State: (a) if any, up to 10 most recent facts with the same `(s, r)` (s as subject) that are not among (b), under
  `Earlier events with the same subject and relation (S — R), oldest to newest:`; (b) the 20 most recent facts
  involving s (as subject or object) under `Most recent events involving S, oldest to newest:` (or
  `No earlier events involving S are recorded.`); then `Query date: YYYY-MM-DD.` Each fact is rendered
  `YYYY-MM-DD: subject — relation — object`. Section (a) is placed first because its facts are all older than (b),
  so the whole state reads oldest → newest. The ≤ 6,000-char guard (drop oldest facts) never triggered.
- Instruction: `Which entity is the most likely object of this event: S — R — ? on YYYY-MM-DD?`
- Candidates: gold + 19 negatives sampled uniformly (seeded, without replacement) from all 7,128 entities,
  excluding every o' with (s, r, o', t) true in any split (time-aware filter). Canonical option order is a seeded
  shuffle (gold position uniform); descriptions null.
- meta: `option_ids` (entity id + 1), `history_ids` = object entity (id + 1) of each shown fact, chronological
  (for facts where s is the object this is s itself), `query = {s, r, t (hours, as in the file), day, date}` with raw
  ids, `fact_index` (row in train‖valid‖test), `n_history_shown`, `n_same_sr_shown`. `group` = subject id.
  No `train_sequences` file.

## Counts

- 90,730 facts; 7,128 entities (all occur); 230 relations.

| split | items | subjects | days | state chars mean / median / p95 / max | facts shown mean | no history | with same-(s,r) section | gold among shown objects |
|---|---|---|---|---|---|---|---|---|
| test | 1,000 | 494 | 31 | 1,890 / 1,998 / 2,797 / 3,750 | 23.6 | 19 | 701 | 58.5% |
| dev | 300 | 209 | 30 | 1,859 / 1,987 / 2,885 / 3,480 | 23.0 | 7 | 202 | 57.0% |
| train | 10,000 | 2,206 | 304 | 1,573 / 1,767 / 2,702 / 3,885 | 19.5 | 641 | 5,490 | 53.5% |

Same-(s, r, t) queries with different gold objects (allowed, each fact is its own query): test 20, dev 4, train 144.
Checks: no negative is a true answer of its (s, r, t); ≤ 3 queries per (s, day).

## Example (first test item)

```
Most recent events involving Governor (Kenya), oldest to newest:
2014-12-01: Governor (Kenya) — Make statement — Government (Kenya)
...
2014-12-15: Pokot — Consult — Governor (Kenya)
2014-12-17: Governor (Kenya) — Make statement — Government (Kenya)

Query date: 2014-12-22.
```
Instruction: `Which entity is the most likely object of this event: Governor (Kenya) — Yield — ? on 2014-12-22?`
(gold: `Citizen (Kenya)`).

## Baselines

Candidate-set metrics (20 options; 95% bootstrap CIs clustered by subject in `report.json`):

| system | split | HR@1 | NDCG@10 | MRR | NLL | ECE |
|---|---|---|---|---|---|---|
| recurrency | dev | 0.822 | 0.883 | 0.866 | 0.743 | 0.043 |
| recurrency | test | 0.818 | 0.883 | 0.867 | 0.696 | 0.033 |
| regcn | dev | 0.850 | 0.906 | 0.890 | 0.683 | 0.066 |
| regcn | test | 0.832 | 0.892 | 0.876 | 0.753 | 0.065 |

(uniform guessing: HR@1 0.05, NLL 3.00)

**recurrency** (`scripts/run_icews14_recurrency.py`, 15 s CPU). This is a simplified version of Gastinger et al. (2024).
Each entity gets recency-weighted shares, with weight 2^(−λ(t−t')) per past fact, normalized over all entities:
strict = share among past objects of (s, r, ·); so = share among s's past partners (any relation, either direction);
ro = share among past objects of (·, r, ·). score = α·strict + (1−α)(β·so + (1−β)·ro), and
p = softmax(log(score + ε)/T) over the 20 candidates. λ, α, β, ε and T are fitted jointly on dev NLL over a grid:
λ = 0.003, α = 0.8, β = 0.75, ε = 1e-6, T = 1.356. Deviations from the paper: the paper's recency term uses the most
recent occurrence and adds a frequency term; here a decayed count gives both (λ → 0 is pure frequency). The paper
selects parameters per relation by validation MRR; here one global set is fitted on dev NLL. The relaxed part mixes
(s, ·, o) and (·, r, o). Full-ranking time-aware filtered MRR over all 7,128 entities on the 1,000 test queries: **0.413**.

**regcn** (`scripts/run_icews14_regcn.py`, `src/seqpred/regcn.py`). This is a plain-PyTorch reimplementation of the
reference code (DGL not installed): uvrgcn layers, GRU relation evolution, time-gated entity evolution, ConvTransE,
history 3 snapshots, hidden 200, 2 layers, dropout 0.2, layer norm, self-loop, Adam 1e-3 (wd 1e-5), clip 1.0,
static graph off, entity prediction only. Training uses each fact and its inverse, as in the reference. It ran on
12 CPU threads; each epoch took 56–80 s. Model selection used validation raw MRR (as in the reference) with patience 3
and at most 25 epochs. The run stopped after 20 epochs (best epoch 16); total time was 20 min. Items are scored by the
decoder score of (s, r, ?) given the ground-truth snapshots of the 3 days before t; softmax temperature
T = 1.732 was fitted on dev NLL.

Full-ranking sanity check on all of test.txt with ground-truth history:

| | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|
| raw, both directions (RE-GCN protocol) | 0.392 | 0.293 | 0.441 | 0.582 |
| time-aware filtered, both directions | **0.400** | 0.305 | 0.447 | 0.585 |
| time-aware filtered, object queries only | 0.421 | 0.328 | 0.465 | 0.601 |

Reference: the RE-GCN paper reports ICEWS14 raw MRR ≈ 0.40 with the static graph; later papers report ≈ 0.42
time-aware filtered. Our run has no static graph and is a short CPU run, and lands about 0.01–0.02 below.

## Deviations / notes

- The task brief asked for "ICEWS14 as in RE-NET/RE-GCN/TLogic, not ICEWS14s". These are three different
  splits (table above); the RE-GCN repo's `ICEWS14s` folder *is* the RE-GCN paper's ICEWS14, and the recurrency
  baseline uses the same data. We therefore use that split (via the recurrency-baseline copy named `ICEWS14`).
  Switching to the TLogic split would only require a different loader (same facts and names).
- `history_ids` follows the brief (object ids of the shown facts); it is not a meaningful SASRec sequence here.
