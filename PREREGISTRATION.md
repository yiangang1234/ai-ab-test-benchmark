# Pre-registration: natural-mix holdout test

Written and committed on 2026-11-11, **before any AI run on these tests**. The commit time on GitHub is the timestamp.
The answer keys are kept out of the repo until all runs are done; their SHA-256 hashes are below so anyone can check later that they were not changed.

## Why

The first results (README) used a quiz with 45 clear winners out of 95 tests, far more than in reality (4%). A Reddit commenter rightly asked for a test on the **natural mix** of outcomes, and pointed out that "no detectable difference" is not equivalence. This test answers both.

## Data and sample

- Source: Upworthy Research Archive, **confirmatory** dataset (`upworthy-archive-confirmatory-packages-03.12.2020.csv`), never used before in this project.
- Same filters as the quiz (`scripts/build_sets.py`): drop June 2013 to January 2014, keep the test's most common image, unique headlines, 1,000+ impressions per option, 2–4 options.
- Pool after filters: **5,097 tests**: clear winner 210 (4.1%), no detectable difference 1,024 (20.1%), inconclusive 3,863.
- Sample: **200 tests, simple random sample** (seed 20261111), so the mix is natural. Drawn: 7 clear winners, 32 no detectable difference, 161 inconclusive.
- Split into two files of 100: `data/holdout_A_blind.csv`, `data/holdout_B_blind.csv`. Option order is shuffled per test.
- Built by `python scripts/build_holdout.py path/to/confirmatory.csv`.

## Labels (unchanged from the quiz)

| Label | Rule |
|---|---|
| Clear winner | best option vs runner-up, two-proportion z-test p < 0.001 |
| No detectable difference | chi-square across all options p > 0.5 (**not** equivalence) |
| Inconclusive | neither |
| Equivalent | TOST on best vs worst option: upper bound of the 90% CI of the relative CTR lift below **+20%** (margin fixed now) |

## Procedure

- Model: Claude Opus 5.5 in claude.ai, web search off, no project or memory context.
- Prompts: `prompts/5_holdout_A.txt` and `prompts/6_holdout_B.txt`, identical to `prompts/1_baseline.txt` except the file name and count.
- Each file is run in **3 separate fresh chats** (6 chats total). Raw CSV answers are saved unedited as `results/holdout_{A,B}_run{1,2,3}.csv`.
- If an answer file is malformed (missing rows, wrong ids), that chat is rerun in a new chat and the reason is logged here. No other reruns.

## Metrics and hypotheses (analysis code: `scripts/analyze_holdout.py`, committed now)

**Primary 1, "skip" flag.** Tests where the AI answered TIE in all 3 runs.
- H1: the share of *no detectable difference* among flagged tests is **higher than the pool share (20.1%)**. One-sided binomial test, alpha 0.05.
- Reported with Wilson 95% CIs: share of *clear winners* among flagged tests (what you would miss by skipping them) vs the pool share (4.1%), the number of clear winners flagged out of 7, and the share of *equivalent* tests among flagged tests vs the whole sample.

**Primary 2, picking winners.** On the 7 clear-winner tests, how often the AI's answer is the true best option, per run and mean, vs random pick. With 7 tests this is descriptive only.

**Secondary 3, does following the AI help on average?** For every test, CTR of the AI's top-scored option divided by the test's mean option CTR, minus 1; ties among top scores are averaged. Mean over the 200 tests and the 3 runs, with a bootstrap 95% CI (2,000 resamples, seed 20261111).

**Secondary 4–5 (descriptive).** Share of tests with the same answer in all 3 runs; accuracy on the 39 non-inconclusive tests; number of TIE answers per run.

## Known limits, stated in advance

- 7 clear winners is a small number, so the miss rate will have a wide confidence interval.
- At about 1% CTR and a few thousand impressions per option, very few tests will pass the ±20% equivalence test (a dry run with random answers found 0 of 200). That is a property of the data, not of the AI.
- One model, 2013–2015 US viral-news headlines.

## Answer key hashes (SHA-256)

```
holdout_key/holdout_answer_key.csv   0a7cdeb56ef351129300ef5b2a3c8ed207d39eefb2229718a5f6a5b4c98a2295
holdout_key/holdout_all_options.csv  363d83b7d8475d960125042f665f43d836cffd17ca8fbc62c5a3947d354d0c59
```

## Deviations log

None. All 6 runs came back complete and well-formed on the first try (2026-11-13); no chat was rerun.

## Results

See the Nov 13 update at the top of the [README](README.md) and [`results/holdout_summary.txt`](results/holdout_summary.txt). The answer keys in `holdout_key/` match the hashes above.
