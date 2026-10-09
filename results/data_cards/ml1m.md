# Data card — ml1m (MovieLens-1M)

## Sources

| file | URL | sha256 |
|---|---|---|
| `ml-1m.zip` | https://files.grouplens.org/datasets/movielens/ml-1m.zip | `a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20` |

## Preprocessing

- `ratings.dat` and `movies.dat` decoded as latin-1; every rating (any score) is an interaction.
- Per user, interactions sorted by timestamp; ties keep `ratings.dat` file order (stable sort).
- Item indices 1..n_items assigned in ascending MovieID order over movies with ≥1 rating.
- Option/history name = `movies.dat` title verbatim (e.g. `American President, The (1995)`), clipped to 100 chars; description = genres joined by `, `.
- Base seed 20261009; 1,000 test + 300 dev users (disjoint), leave-one-out; 10,000 train items (targets at positions 3..len-3, ≤3 per user) in seeded order.
- State: header + the most recent ≤20 history items, numbered oldest → newest.
- Candidates: gold + 19 negatives uniform over movies the user never rated (seeded per item id); option order seeded shuffle; duplicate names disambiguated with ` (#<item index>)`.

## Counts

- users: 6,040; items: 3,706; interactions: 1,000,209
- sequence length: mean 165.6, min 20, max 2314
- SASRec train_sequences: 6,040 sequences, 988,129 interactions (last two items of every user removed)
- train-item pool (eligible targets): 970,009; sampled 10,000 from 4,476 users (≤3 per user)

| split | items | state chars mean | state chars max | history items mean | history items min | null descriptions | items with disambiguated names |
|---|---|---|---|---|---|---|---|
| test | 1,000 | 599.2 | 871 | 19.99 | 19 | 0.00% | 0 |
| dev | 300 | 596.3 | 770 | 19.94 | 18 | 0.00% | 0 |
| train | 10,000 | 561.1 | 831 | 18.56 | 3 | 0.00% | 0 |

## Example state (test item 1, truncated)

```
Movies the user watched, oldest to newest:
1. Tomorrow Never Dies (1997)
2. Smoke (1995)
3. Mansfield Park (1999)
4. Pi (1998)
5. Twin Falls Idaho (1999)
6. Man Bites Dog (C'est arrivé près de chez vous) (1992)
7. Madonna: Truth or Dare (1991)
8. High Art (1998)
9. Thin Red Line, The (1998)
10. Afterglow (1997)
11. X-Files: Fight the Future, The (1998)
12. Mighty Aphrodite (1995)
13. Fifth Element, The (1997)
14. Nightmare Before Christmas, The (1993)
15. Crow, The (1994)
16. Death and the Maiden (1994)
17. Billy's Hollywood Screen Kiss (1997)
18. Liar Liar (1997)
19. Awakenings (1990)
20. Fea
```

## Deviations / notes

- No k-core filtering (ML-1M users already have ≥20 ratings).
- Movie titles are not re-ordered (`X, The (year)` kept as in the source).
