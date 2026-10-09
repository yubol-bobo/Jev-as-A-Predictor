# Data card — bpic12 (BPI Challenge 2012, next activity)

Builder: `scripts/build_bpic12.py` (deterministic; re-running reproduces every output byte-for-byte).

## Sources

| file | URL | md5 / sha256 |
|---|---|---|
| `BPI_Challenge_2012.xes.gz` | official record: 4TU.ResearchData, DOI [10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f](https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f) (article 12689204; API `https://data.4tu.nl/v2/articles/12689204/files`, download_url `https://data.4tu.nl/file/533f66a4-8911-4ac7-8612-1235d65d1f37/3276db7f-8bee-4f2b-88ee-92dbffb5a893`). **Fetched from** the official BPI 2012 challenge page at TU/e: https://www.win.tue.nl/bpi/2012/financial_log.xes.gz | md5 `74c7ba9aba85bfcb181a22c9d565e5b5` (= 4TU `supplied_md5` and `computed_md5`), sha256 `5cd9cc16b9bcb20bd4aae45666a5d87479ddbf47e6371618b6ad217174cecdf3` |

On 2026-10-09 the 4TU file endpoint returned HTTP 503 `StorageMaintenance`. The TU/e copy is byte-identical to the
4TU record (same MD5 as published by 4TU, same size 3,342,406 B). Saved as `data/external/bpic12/BPI_Challenge_2012.xes.gz`
(gitignored) and parsed with `xml.etree.ElementTree.iterparse` (data only).

## Preprocessing

- Only events with `lifecycle:transition == COMPLETE` (164,506 of 262,200 events); all 13,087 cases keep ≥ 3 events.
  Events within a case sorted by `time:timestamp` (stable on file order for ties).
- 23 activities (`A_*`, `O_*`, `W_*`, e.g. `W_Completeren aanvraag`); all appear in the train part, so no test
  prefix is dropped for unseen activities.
- Cases sorted by first-event time (ties by case id). First 80% = train part (10,469 cases), last 20% = test
  (2,618 cases). Last 10% of the train part by time = dev (1,047 cases); the remaining 9,422 = train cases.
  Case start ranges: train 2011-10-01 → 2012-01-22, dev 2012-01-22 → 2012-02-03, test 2012-02-03 → 2012-02-29.
- Vocabulary: 23 activities sorted by name (indices 1..23) + `[END]` (index 24). `vocab.json` also stores `end_index`.
- Items are prefixes: given events 1..k (k ≥ 1) predict event k+1, or `[END]` if k = case length.
  Candidates = all 24 vocabulary entries (no sampling); option names = activity names, descriptions null.
- State: `Events of this case so far, oldest to newest:`, `(earlier events omitted: n)` if k > 20, then the 20 most
  recent events as `<event number>. <activity> (t+<elapsed since the case's first COMPLETE event>)`, elapsed
  formatted `0m`, `2h 5m`, `3d 4h` (minutes / hours+minutes / days+hours, truncated). Event numbers are the true
  positions in the case (e.g. 28..47 when 27 events are omitted).
- Instruction: `Which activity will happen next in this case? Choose [END] if the case is finished.`
- Prefix sampling: population = all prefixes (k = 1..len) of the split's cases; strata = prefix-length bins
  (1,2,…,8, 9–10, 11–12, 13–15, 16–20, 21–25, 26–30, 31–40, 41–60, 61+); **proportional allocation** (largest
  remainder) so the sampled prefix-length distribution matches the population of all prefixes, as in the
  next-activity literature; within strata a seeded shuffle, strata interleaved, at most 3 prefixes per case.
  Test 1,000 (775 cases), dev 300 (234 cases), train 10,000 (5,347 cases, stored in seeded random order → nested subsets).
- `meta`: `option_ids` (1..24), `history_ids` (= the ≤ 20 shown events), `case`, `prefix_len`, `case_len`,
  `prefix_ids` (full prefix; used by the n-gram baseline).
- `train_sequences.jsonl.gz`: each train case's activity-index sequence followed by the `[END]` index
  (9,422 sequences, 129,236 tokens). Dev and test cases are not in it.

## Counts

| split | items | cases | state chars mean | state chars max | prefix length mean | truncated (k > 20) | gold = [END] |
|---|---|---|---|---|---|---|---|
| test | 1,000 | 775 | 329.2 | 979 | 10.46 | 12.6% | 8.7% |
| dev | 300 | 234 | 372.3 | 921 | 12.21 | 18.7% | 7.0% |
| train | 10,000 | 5,347 | 364.9 | 1,014 | 12.19 | 18.2% | 8.5% |

Case length (COMPLETE events): mean 12.57, min 3, max 96.

## Baselines

Predictions in `results/predictions/bpic12/<system>/{dev,test}.jsonl`; means with 95% case-clustered bootstrap CIs.

| system | split | n | HR@1 | NDCG@10 | MRR | NLL | ECE |
|---|---|---|---|---|---|---|---|
| majority | dev | 300 | 0.137 [0.102, 0.173] | 0.367 [0.331, 0.401] | 0.296 [0.263, 0.329] | 2.929 [2.854, 3.009] | 0.001 |
| majority | test | 1000 | 0.112 [0.093, 0.133] | 0.369 [0.349, 0.391] | 0.293 [0.276, 0.313] | 2.898 [2.853, 2.942] | 0.023 |
| ngram | dev | 300 | 0.757 [0.709, 0.805] | 0.896 [0.876, 0.916] | 0.860 [0.833, 0.887] | 0.700 [0.615, 0.788] | 0.041 |
| ngram | test | 1000 | 0.738 [0.710, 0.765] | 0.884 [0.872, 0.897] | 0.847 [0.831, 0.864] | 0.791 [0.732, 0.856] | 0.033 |
| sasrec | dev | 300 | 0.780 [0.732, 0.825] | 0.907 [0.886, 0.926] | 0.875 [0.847, 0.901] | 0.671 [0.581, 0.767] | 0.048 |
| sasrec | test | 1000 | 0.746 [0.720, 0.772] | 0.889 [0.876, 0.901] | 0.852 [0.836, 0.868] | 0.761 [0.701, 0.826] | 0.039 |

- **majority**: next-activity frequencies over all prefixes of the train cases (+1 smoothing), normalized over
  the 24 candidates (a constant distribution; argmax `W_Nabellen offertes`).
- **ngram**: back-off prefix n-gram of order ≤ 4 (≤ 3 context activities, case start padded with a BOS token),
  fit on train cases; uses the longest context seen in training, additive smoothing α over the 24 candidates,
  α = 0.001 chosen on dev NLL from {0.001, …, 1} (dev NLL is flat for α ≤ 0.03).
- **sasrec**: `scripts/run_sasrec.py --task bpic12 --max-len 100 --dropout 0.2 --prefix-examples 20`
  (119,814 prefix → next-activity examples from the 9,422 train cases, each cropped to the last 20 events to
  match the shown history; hidden 64, 2 blocks, seed 0; CPU, 12 threads). Best dev NDCG@10 at epoch 30,
  stopped at 50; temperature 1.05; 3,993 s training on a shared, heavily loaded node.
  Without `--prefix-examples` (whole-sequence training) the same command gave test HR@1 0.440: with left
  padding the last real token always sits at the last position and is always followed by `[END]`, so the
  position embedding leaks the distance to the case end and the model predicted `[END]` for 49.5% of test
  prefixes (true rate 8.7%). Both SASRec runs used the fixed embedding init in `src/seqpred/sasrec.py`.
- Sanity anchor: LSTM next-activity accuracy on BPIC12 (complete events) is ~0.75–0.80 in the literature;
  SASRec 0.746 and n-gram 0.738 on test are in range (our test split is chronological and right-censored,
  see caveats).

## Caveats / deviations

- **Right-censoring of test cases:** test cases start in Feb 2012 and the log ends 2012-03-14, so some test cases
  are still running when the log stops, yet their last recorded event is followed by `[END]` in the items.
  Test cases are shorter (mean 11.5 vs 12.7 train) and more often end on in-progress activities
  (`W_Completeren aanvraag`, `W_Nabellen offertes`). This is inherent to the specified chronological split
  and affects all systems equally; it slightly lowers attainable accuracy on `[END]` items.
- Train cases may overlap in time with dev/test cases (split by case start, not by event time); standard for
  case-based chronological splits.

## Example state (longest test prefix, k = 47)

```
Events of this case so far, oldest to newest:
(earlier events omitted: 27)
28. W_Nabellen incomplete dossiers (t+22d 13h)
29. W_Nabellen incomplete dossiers (t+22d 18h)
...
46. W_Valideren aanvraag (t+27d 12h)
47. W_Nabellen incomplete dossiers (t+27d 13h)
```
