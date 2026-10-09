# Data card — fsq_nyc (Foursquare NYC, next POI)

Builder: `scripts/build_fsq_nyc.py` (deterministic; re-running reproduces every output byte-for-byte).

## Sources

| file | URL | sha256 |
|---|---|---|
| `dataset_tsmc2014.zip` | http://www-public.imtbs-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip (original author page, Yang et al., TSMC 2014) | `cbe3fdab373d24b09b5fc53509c8958c77ff72b6c1a68589ce337d4f9a80235b` |
| ↳ `dataset_tsmc2014/dataset_TSMC2014_NYC.txt` | member of the zip (227,428 check-ins) | `2f39ce698e5bff6683c74bec30c436cc022c160b0f3d73c5268b55491b2445f9` |

Downloaded 2026-10-09 to `data/external/fsq_nyc/` (gitignored). The file is latin-1 encoded (e.g. `Café`).

## Preprocessing

- Columns: userId, venueId, venueCategoryId, venueCategory, lat, lon, timezoneOffset (min), utcTimestamp.
  Local time = UTC + timezoneOffset.
- Iterative filtering until stable: drop venues with < 10 check-ins, then users with < 10 check-ins.
  227,428 → 147,938 check-ins; 1,083 users (unchanged); 38,333 → 5,135 venues.
- Per user, check-ins sorted by local time; ties keep file order (stable sort).
- Item index 1..5,135 = rank of the raw Foursquare venue id (a hex hash) in sorted order, so the index (shown as
  `V0123`) carries no popularity or time information. 0 = padding.
- Venue category = the venue's most frequent category over its check-ins (ties → earliest check-in);
  142 venues carry more than one category in the raw data. Venue coordinates = median lat/lon of its check-ins.
- Option name `V<index:04d> · <category>` (unique by construction). Description
  `<category>; <d> km from the last check-in; visited <k> time(s) before by this user`, where d is the haversine
  distance from the last shown check-in's venue, and **k counts the user's entire history before the target**
  (not only the 20 shown check-ins).
- State: header line, `(earlier check-ins omitted: n)` if truncated, then the 20 most recent check-ins before the
  target, oldest → newest, numbered 1..20, e.g. `3. Tue 18:40 — Coffee Shop (venue V0123)` (local weekday and
  time). The date/time of the target check-in is not given.
- Instruction: `Which venue will the user check in at next?`
- Candidates: gold + 19 venues sampled uniformly (seeded per item id) from all 5,135 filtered venues except the
  gold. **Venues the user visited before are not excluded** (revisits are the norm in POI data: the gold was
  previously visited for ~70% of test items), so negatives can also be previously-visited venues.
  Canonical option order = ascending item index; presentation order is the seeded shuffle of `make_item`.
- `meta`: `option_ids`, `history_ids` (= exactly the shown check-ins), `user`, `target_pos`, `n_prior`,
  `gold_prior_visits`, `option_raw` (raw venue ids).

## Splits (leave-one-out)

- **Deviation (as agreed for < 1,300 users):** NYC has only 1,083 users, so **all 1,083 users** are test items
  (target = last check-in), and dev = the second-to-last check-in of **300 users sampled from the same users**
  (dev and test users overlap; dev targets are inputs in the corresponding test histories, as in standard
  leave-one-out).
- Train: targets at positions ≤ len−3 (0-based ≥ 3, i.e. ≥ 3 history steps), never a dev/test target.
  Eligible pool 142,523; 10,000 sampled round-robin over users in a seeded user order (every user contributes
  9–10 items, all 1,083 users covered), then stored in a seeded random order → first N lines = nested subsets.
- `train_sequences.jsonl.gz`: each user's full venue-index sequence with the last two check-ins removed
  (1,083 sequences, 145,772 check-ins).

## Counts

| split | items | state chars mean | state chars max | history steps mean | gold previously visited | gold in shown history |
|---|---|---|---|---|---|---|
| test | 1,083 | 954.4 | 1,398 | 19.99 | 69.6% | 59.7% |
| dev | 300 | 950.1 | 1,398 | 19.98 | 67.0% | 57.7% |
| train | 10,000 | 859.1 | 1,468 | 17.97 | 66.8% | 60.3% |

Users 1,083; venues 5,135; categories 208; check-ins per user mean 136.6 (min 17, median 93, max 2,515);
time span 2012-04-03 → 2013-02-15 (local).

## Baselines

Predictions in `results/predictions/fsq_nyc/<system>/{dev,test}.jsonl`; means with 95% user-clustered bootstrap CIs.

| system | split | n | HR@1 | NDCG@10 | MRR | NLL | ECE |
|---|---|---|---|---|---|---|---|
| popularity | dev | 300 | 0.134 [0.098, 0.174] | 0.344 [0.305, 0.382] | 0.282 [0.247, 0.317] | 2.933 [2.840, 3.033] | 0.080 |
| popularity | test | 1083 | 0.167 [0.145, 0.189] | 0.385 [0.364, 0.405] | 0.316 [0.296, 0.335] | 2.834 [2.778, 2.890] | 0.062 |
| sasrec | dev | 300 | 0.707 [0.653, 0.760] | 0.821 [0.785, 0.855] | 0.783 [0.742, 0.822] | 1.092 [0.939, 1.255] | 0.056 |
| sasrec | test | 1083 | 0.711 [0.684, 0.737] | 0.823 [0.806, 0.840] | 0.786 [0.767, 0.806] | 1.017 [0.943, 1.094] | 0.066 |

- **popularity**: training check-in counts (from `train_sequences`) + 1, normalized over the 20 candidates
  (written by the builder). Weak because the uniformly sampled negatives are not popularity-matched and it ignores
  the user's own revisits.
- **sasrec**: `scripts/run_sasrec.py --task fsq_nyc --max-len 50 --dropout 0.3` (full-data, CE over all items,
  hidden 64, 2 blocks, seed 0; CPU, 12 threads). Early-stopped at epoch 105 (best dev NDCG@10 at epoch 85),
  temperature 1.73 fitted on dev NLL, 79 s training. Run after the embedding-init fix in `src/seqpred/sasrec.py`
  (std 0.02 instead of PyTorch's N(0,1)); with the original init the same command reached only 0.657 test HR@1 and
  had not converged after 200 epochs.

## Example state (test item 1)

```
Check-ins of this user in New York City, oldest to newest (local time):
(earlier check-ins omitted: 24)
1. Tue 19:15 — Bar (venue V3893)
2. Fri 20:39 — Concert Hall (venue V0566)
3. Sun 15:44 — Flea Market (venue V3734)
...
19. Fri 12:40 — Airport (venue V0406)
20. Wed 22:05 — Concert Hall (venue V5124)
```
Example criterion: `"V1476 · Airport": "Airport; 10.4 km from the last check-in; visited 1 time before by this user"`.
