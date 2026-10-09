# Design — System One decision models as sequential predictors

*Status: agreed with the user (2026-10-09).*

## 1. Research objective / 研究目标

Evaluate whether **System One decision models** (state + typed question in → probability
distribution over supplied candidates out, no decoding) are good predictors on classic
**sequential prediction** tasks:

1. **Untrained (zero-shot):** how well do they predict the next element directly?
2. **Trained (finetuned):** how much does task-specific finetuning improve them, and how many
   training examples are needed to approach a dedicated sequence model (SASRec)?

评测 System One 类决策模型在经典序列预测任务上的能力：不训练直接使用的效果，以及 finetune 后的提升和数据效率。

## 2. Systems / 系统

| System | Backbone | Untrained | Trained |
|---|---|---|---|
| JEV `jev-1.13.0` | closed API | ✅ reference | ✗ (closed weights) |
| pplx-decider-v1.1-27b | Qwen3.8-27B + readout head (≤255 options, noncausal attention) | ✅ | ✅ |
| Cloudflare Clef (27B) | Qwen3.8-27B + joint schema head | ✅ | ✅ |
| Kev-27B | Qwen3.8-27B + LoRA + pointer head | ✅ | ✅ |
| SASRec (dedicated baseline) | — | — | ✅ trained on full training data |

ICEWS14 uses **RE-GCN** and a **recurrency** heuristic (Gastinger et al., 2024) instead of SASRec.

## 3. Tasks / 任务

| Task | Dataset | Prediction target |
|---|---|---|
| Sequential recommendation | MovieLens-1M | next movie |
| Sequential recommendation | Amazon Beauty | next product |
| Next POI | Foursquare NYC | next venue |
| Event forecasting (TKG) | ICEWS14 | object entity of (s, r, ?, t) |
| Next activity (process mining) | BPI Challenge 2012 | next activity or [END] |

## 4. Evaluation protocol / 评测协议

| Item | Protocol |
|---|---|
| Candidates | 1 positive + 19 uniformly sampled negatives (seeded); BPIC12 uses the full activity set + [END] |
| History in state | most recent 20 steps, rendered as text (titles, venue categories, activity names, …) |
| Splits | rec / POI: leave-one-out (last = test, second-to-last = dev); BPIC12 and ICEWS14: chronological |
| Sizes | ~1,000 test and ~300 dev items per task, identical across all systems |
| Metrics | HR@1, NDCG@10, MRR; NLL and ECE for probability quality; 95% bootstrap CIs |
| Same input for all | every System One model receives the identical request (state + Choice question) |

## 5. Finetuning / 训练设置

- **Per-task** finetuning (one model per task × system).
- **Identical recipe** for pplx-decider, Clef and Kev: LoRA rank 16 on the backbone + each model's own
  decision head, cross-entropy on the gold option, same hyperparameters / steps / data; temperature
  fitted on dev. Kev uses its released trainer; training code for pplx-decider and Clef is written to
  match their released inference code and aligned with Kev's loss and LoRA settings.
- **Learning curve:** N ∈ {0 (untrained), 500, 2,000, 10,000} training examples, built from training
  data only in the same format as the test items (fresh negatives).
- **Scale:** full grid — 3 models × 5 tasks × 3 sizes = **45 finetuning runs**.
- SASRec is trained on full training data **and** on the same N subsets (matched-data comparison).
- Out of scope for now: checking forgetting of general decision ability after finetuning.

## 6. Order of work / 执行顺序

1. Data: build and freeze dev/test items for the 5 tasks; train SASRec / RE-GCN / recurrency baselines.
2. Untrained evaluation of all 4 System One systems.
3. Finetuning grid (45 runs) and evaluation.
4. Analysis and write-up.

Progress is tracked in the README and pushed to `main` after each step; results are reported to the user before moving to the next step.
