# Data card — beauty (Amazon Beauty 2014, 5-core)

## Sources

| file | URL | sha256 |
|---|---|---|
| `reviews_Beauty_5.json.gz` | https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz | `e5925ec99023f1dc9c7d3dff7ef34cfeacd40d1b981b5525e1e4ba9c8abc18fe` |
| `meta_Beauty.json.gz` | https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz | `c7977dc0e0ead14ac7df8c1c6de74da00e4bebfad87a0f0c5763d4c5bc9b53a0` |

## Preprocessing

- Interactions = all reviews in the official 5-core file (no further filtering; no duplicate (user, item) pairs present).
- Per user, sorted by `unixReviewTime` (day resolution); ties keep file order (stable sort).
- Item indices 1..n_items assigned in ascending (string) asin order.
- Metadata parsed with `ast.literal_eval` for the asins in the 5-core file only.
- Name = HTML-unescaped, whitespace-normalised `title`, clipped to 100 chars; items without a title are named `Product <asin>`.
- Description = first category path under `Beauty` without the root (e.g. `Skin Care > Body > Moisturizers`), plus ` (brand: X)` when a brand exists, clipped to 150 chars; null if neither exists.
- Base seed 20261009; 1,000 test + 300 dev users (disjoint), leave-one-out; 10,000 train items (targets at positions 3..len-3, ≤3 per user) in seeded order.
- State: header + the most recent ≤20 history items, numbered oldest → newest.
- Candidates: gold + 19 negatives uniform over items the user never reviewed (seeded per item id); option order seeded shuffle; duplicate names disambiguated with ` (#<item index>)`.

## Counts

- users: 22,363; items: 12,101; interactions: 198,502
- sequence length: mean 8.88, min 5, max 204
- SASRec train_sequences: 22,363 sequences, 153,776 interactions (last two items of every user removed)
- train-item pool (eligible targets): 86,687; sampled 10,000 from 6,364 users (≤3 per user)

| split | items | state chars mean | state chars max | history items mean | history items min | null descriptions | items with disambiguated names |
|---|---|---|---|---|---|---|---|
| test | 1,000 | 504.5 | 2132 | 7.07 | 4 | 0.00% | 0 |
| dev | 300 | 431.8 | 1528 | 6.01 | 3 | 0.00% | 0 |
| train | 10,000 | 574.6 | 2135 | 8.04 | 3 | 0.00% | 1 |

## Example state (test item 1, truncated)

```
Products the user reviewed, oldest to newest:
1. It's A 10 + Keratin Leave-In Conditioner 10 oz.
2. It's A 10 Miracle Oil Plus Keratin for Unisex, 3 Ounce
3. It's a 10 Miracle Shampoo Plus Keratin, 10 Ounce
4. It's a 10 Miracle Deep Conditioner Plus Keratin 17.5 oz
```

## Deviations / notes

- 7 of 12,101 items have no title in the metadata (0 have no metadata line at all); they stay in all sequences and are named `Product <asin>`.
- 0 items have neither category nor brand (description null).
- Timestamps are day-resolution: 77,402 consecutive same-day pairs within users, ordered by file order (which is grouped by asin, not true time).
