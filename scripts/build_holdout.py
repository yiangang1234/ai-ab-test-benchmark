"""Build the natural-mix holdout test from the Upworthy CONFIRMATORY dataset.

Usage (from the repo root):
    python scripts/build_holdout.py path/to/upworthy-archive-confirmatory-packages-03.12.2020.csv

Writes:
    data/holdout_A_blind.csv, data/holdout_B_blind.csv   what the AI sees (100 tests each)
    prompts/5_holdout_A.txt, prompts/6_holdout_B.txt     exact prompts
    holdout_key/holdout_answer_key.csv                   kept out of git until all runs are done
    holdout_key/holdout_all_options.csv                  kept out of git until all runs are done
    and prints the SHA-256 of both key files (recorded in PREREGISTRATION.md before any run).
"""
import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sets import LET, TASK, clean_arms, load, test_stats  # same filters and labels as the quiz

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
DATA, PROMPTS, KEY = ROOT / "data", ROOT / "prompts", ROOT / "holdout_key"
N, SEED = 200, 20261111


def main(src):
    KEY.mkdir(exist_ok=True)
    _, df = load(src)
    tests = []
    for tid, g in df.groupby("clickability_test_id"):
        g = clean_arms(g)
        if not 2 <= len(g) <= 4:
            continue
        g, p_top, p_all = test_stats(g)
        label = "winner" if p_top < 0.001 else ("no_difference" if p_all > 0.5 else "inconclusive")
        tests.append(dict(test_id=tid, g=g, p_top=p_top, p_all=p_all, label=label))
    pool = pd.DataFrame([{k: v for k, v in t.items() if k != "g"} for t in tests])
    print(f"confirmatory pool: {len(pool)} tests, labels {pool['label'].value_counts().to_dict()}")

    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(len(tests), size=N, replace=False))  # simple random sample = natural mix
    order = list(rng.permutation(pick))
    quiz, key, detail = [], [], []
    for n, i in enumerate(order, 1):
        t = tests[i]
        g = t["g"].sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)
        qid = f"H{n:03d}"
        best = LET[int(g["ctr"].idxmax())]
        answer = best if t["label"] == "winner" else ("TIE" if t["label"] == "no_difference" else "-")
        row = dict(id=qid, n_options=len(g))
        for j in range(4):
            row[f"headline_{LET[j]}"] = g.loc[j, "headline"] if j < len(g) else ""
        quiz.append(row)
        key.append(dict(id=qid, test_id=t["test_id"], n_options=len(g), label=t["label"], answer=answer,
                        best_option=best, p_top=t["p_top"], p_all=t["p_all"]))
        for j in range(len(g)):
            detail.append(dict(id=qid, option=LET[j], headline=g.loc[j, "headline"],
                               impressions=int(g.loc[j, "impressions"]), clicks=int(g.loc[j, "clicks"])))
    quiz = pd.DataFrame(quiz)
    quiz.iloc[:100].to_csv(DATA / "holdout_A_blind.csv", index=False)
    quiz.iloc[100:].to_csv(DATA / "holdout_B_blind.csv", index=False)
    pd.DataFrame(key).to_csv(KEY / "holdout_answer_key.csv", index=False)
    pd.DataFrame(detail).to_csv(KEY / "holdout_all_options.csv", index=False)
    print("sample labels:", pd.DataFrame(key)["label"].value_counts().to_dict())

    for name, part in (("5_holdout_A.txt", "holdout_A_blind.csv"), ("6_holdout_B.txt", "holdout_B_blind.csv")):
        (PROMPTS / name).write_text(TASK.format(file=part, n=100) + "\n", encoding="utf-8")
    for f in ("holdout_answer_key.csv", "holdout_all_options.csv"):
        print(f"sha256 {f}: {hashlib.sha256((KEY / f).read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main(sys.argv[1])
