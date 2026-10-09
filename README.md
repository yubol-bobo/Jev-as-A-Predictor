# Jev-as-A-Predictor

Are System One decision models (JEV, pplx-decider, Cloudflare Clef, Kev) good sequential predictors — untrained, and after finetuning?

System One 决策模型能否作为序列预测器？比较不训练与 finetune 后的表现。

- Design / 设计: [docs/design.md](docs/design.md)

## Progress / 进展

| Date | Step | Status |
|---|---|---|
| 2026-10-09 | Design agreed | ✅ |
| 2026-10-09 | 1a. Shared item format, metrics, SASRec (CE) + driver; SASRec sanity check passed | ✅ |
| 2026-10-09 | 1b. Data frozen for 5 tasks (dev 300 / test 1,000 [NYC: all 1,083 users] / train 10,000) + baselines | ✅ dev/test committed; train files pending versioning decision |
| | 2. Untrained evaluation (JEV, pplx-decider, Clef, Kev) | ⏳ |
| | 3. Finetuning grid (45 runs) + matched-N SASRec | ⏳ |
| | 4. Analysis | ⏳ |

## Step 1 baselines (test, 20 candidates; BPIC12: 24) / 对照组结果

| task | system | HR@1 | NDCG@10 | MRR | NLL | ECE |
|---|---|---|---|---|---|---|
| ml1m | popularity | 0.270 | 0.557 | 0.457 | 2.331 | 0.031 |
| ml1m | sasrec | 0.686 | 0.834 | 0.790 | 1.087 | 0.037 |
| beauty | popularity | 0.163 | 0.388 | 0.321 | 2.920 | 0.093 |
| beauty | sasrec | 0.386 | 0.562 | 0.509 | 2.239 | 0.056 |
| fsq_nyc | popularity | 0.167 | 0.385 | 0.316 | 2.834 | 0.062 |
| fsq_nyc | sasrec | 0.711 | 0.823 | 0.786 | 1.017 | 0.066 |
| icews14 | recurrency | 0.818 | 0.883 | 0.867 | 0.696 | 0.033 |
| icews14 | regcn | 0.832 | 0.892 | 0.876 | 0.753 | 0.065 |
| bpic12 | majority | 0.112 | 0.369 | 0.293 | 2.898 | 0.023 |
| bpic12 | ngram | 0.738 | 0.884 | 0.847 | 0.791 | 0.033 |
| bpic12 | sasrec | 0.746 | 0.889 | 0.852 | 0.761 | 0.039 |

Data cards: `results/data_cards/<task>.md`. BPIC12 SASRec uses `--prefix-examples 20`.
