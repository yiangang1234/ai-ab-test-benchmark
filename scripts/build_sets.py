"""Build every test set and prompt from the Upworthy exploratory dataset.

Usage (from the repo root):
    python scripts/build_sets.py path/to/upworthy-archive-exploratory-packages-03.12.2020.csv

Download the CSV from https://osf.io/jd64p/ (folder upworthy-archive-datasets).
All random choices use fixed seeds, so the output matches the files in data/ and prompts/.
"""
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parent.parent
DATA, PROMPTS = ROOT / "data", ROOT / "prompts"
LET = "ABCD"


def load(src):
    raw = pd.read_csv(src)
    raw["created_at"] = pd.to_datetime(raw["created_at"], errors="coerce")
    # The archive documentation warns that randomization may be invalid from June 2013 to January 2014.
    bad = (raw["created_at"] >= "2013-06-01") & (raw["created_at"] < "2014-02-01")
    df = raw[~bad & raw["impressions"].notna() & raw["clicks"].notna() & raw["headline"].notna()]
    return raw, df


def p_two(c1, n1, c2, n2):
    """Two-sided two-proportion z-test."""
    p = (c1 + c2) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = abs(c1 / n1 - c2 / n2) / se
    return 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))


def clean_arms(g):
    """Keep headline-only variation: arms with the test's most common image, unique headlines, 1,000+ impressions."""
    eye = g["eyecatcher_id"].mode()
    if len(eye):
        g = g[g["eyecatcher_id"] == eye.iloc[0]]
    g = g.drop_duplicates("headline")
    return g[g["impressions"] >= 1000]


def test_stats(g):
    g = g.assign(ctr=g["clicks"] / g["impressions"]).sort_values("ctr", ascending=False).reset_index(drop=True)
    a, b = g.iloc[0], g.iloc[1]
    p_top = p_two(a["clicks"], a["impressions"], b["clicks"], b["impressions"])
    t = np.array([g["clicks"], g["impressions"] - g["clicks"]])
    p_all = chi2_contingency(t)[1] if (t > 0).all() else 1.0
    return g, p_top, p_all


# ---------------------------------------------------------------- quiz (95)
def build_quiz(df):
    """Clear winner: best arm beats runner-up at p < 0.001.  No detectable difference: chi-square across arms p > 0.5."""
    rng = np.random.default_rng(20261102)
    tests = []
    for tid, g in df.groupby("clickability_test_id"):
        g = clean_arms(g)
        if not 2 <= len(g) <= 4:
            continue
        g, p_top, p_all = test_stats(g)
        tests.append(dict(test_id=tid, date=g.loc[0, "created_at"].date(), k=len(g), arms=g, p_top=p_top, p_all=p_all))
    meta = pd.DataFrame([{k: v for k, v in t.items() if k != "arms"} for t in tests])
    sig_idx = meta.index[meta["p_top"] < 0.001]
    tie_idx = meta.index[meta["p_all"] > 0.5]

    def stratified(idx, total=50):
        groups = {k: list(meta.loc[idx][meta.loc[idx, "k"] == k].index) for k in (2, 3, 4)}
        for k in groups:
            rng.shuffle(groups[k])
        picked = []
        while len(picked) < total and any(groups.values()):
            for k in (2, 3, 4):
                if groups[k] and len(picked) < total:
                    picked.append(groups[k].pop())
        return sorted(picked)

    sig_pick, tie_pick = stratified(sig_idx), stratified(tie_idx)
    quiz_rows, detail_rows = [], []
    for picks, kind in ((sig_pick, "winner"), (tie_pick, "no_difference")):
        for n, i in enumerate(picks, 1):
            t = tests[i]
            g = t["arms"].sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)
            qid = f"{'S' if kind == 'winner' else 'T'}{n:02d}"
            answer = LET[g["ctr"].idxmax()] if kind == "winner" else "TIE"
            q = dict(id=qid, n_options=t["k"])
            for j in range(4):
                q[f"headline_{LET[j]}"] = g.loc[j, "headline"] if j < len(g) else ""
            quiz_rows.append(q)
            for j in range(len(g)):
                detail_rows.append(dict(id=qid, kind=kind, test_id=t["test_id"], date=t["date"], n_options=t["k"],
                                        option=LET[j], headline=g.loc[j, "headline"],
                                        impressions=int(g.loc[j, "impressions"]), clicks=int(g.loc[j, "clicks"]),
                                        p_top=t["p_top"], p_all=t["p_all"], answer=answer))
    quiz = pd.DataFrame(quiz_rows).sample(frac=1, random_state=11).reset_index(drop=True)
    detail = pd.DataFrame(detail_rows)
    quiz.to_csv(DATA / "quiz_95_blind.csv", index=False)
    detail.to_csv(DATA / "quiz_95_all_options.csv", index=False)
    detail.drop_duplicates("id")[["id", "kind", "test_id", "n_options", "answer"]].to_csv(DATA / "quiz_95_answer_key.csv", index=False)
    meta.drop(columns=[]).to_csv(DATA / "all_2to4_option_tests.csv", index=False)
    print(f"quiz: {len(meta)} usable 2-4 option tests, {len(sig_idx)} clear winners, {len(tie_idx)} no detectable difference")
    return detail


# ---------------------------------------------------------------- training pool + 30 examples
def build_train(raw, df, quiz_detail):
    rng = np.random.default_rng(20261102)
    quiz_ids = set(quiz_detail["test_id"])
    quiz_slugs = set(raw[raw["clickability_test_id"].isin(quiz_ids)]["slug"].dropna())
    rows = []
    for tid, g in df.groupby("clickability_test_id"):
        if tid in quiz_ids or set(g["slug"].dropna()) & quiz_slugs:  # same story re-tested = leakage
            continue
        g = clean_arms(g)
        k = len(g)
        if not 2 <= k <= 4:
            continue
        g, p_top, p_all = test_stats(g)
        label = "winner" if p_top < 0.01 else ("no_difference" if p_all > 0.5 else "inconclusive")
        g = g.sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)
        best = LET[int(g["ctr"].idxmax())]
        for j in range(k):
            rows.append(dict(test_id=tid, date=g.loc[0, "created_at"].date(), n_options=k, label=label,
                             answer=best if label == "winner" else ("TIE" if label == "no_difference" else "-"),
                             option=LET[j], headline=g.loc[j, "headline"].replace("<br>", " ").strip(),
                             impressions=int(g.loc[j, "impressions"]), clicks=int(g.loc[j, "clicks"]),
                             ctr=g.loc[j, "ctr"], p_top=p_top, p_all=p_all))
    pool = pd.DataFrame(rows)
    tests = pool.drop_duplicates("test_id")

    def pick(label, total):
        t = tests[tests["label"] == label]
        per = {k: list(t[t["n_options"] == k]["test_id"]) for k in (2, 3, 4)}
        for k in per:
            rng.shuffle(per[k])
        out = []
        while len(out) < total and any(per.values()):
            for k in (2, 3, 4):
                if per[k] and len(out) < total:
                    out.append(per[k].pop())
        return out

    few_ids = pick("winner", 20) + pick("no_difference", 10)
    order = list(rng.permutation(few_ids))
    few = pool[pool["test_id"].isin(few_ids)].copy()
    few["ex_no"] = few["test_id"].map({t: i + 1 for i, t in enumerate(order)})
    few = few.sort_values(["ex_no", "option"]).reset_index(drop=True)
    pool["in_examples"] = pool["test_id"].isin(few_ids)
    pool.to_csv(DATA / "train_pool.csv", index=False)

    lines = ["Below are 30 real headline tests from Upworthy (2013-2015). Each test showed the headlines to "
             "randomly split real readers. Click-through rate (CTR) is shown for each option, then the outcome.",
             "Outcome 'TIE' means the differences were too small to matter.", ""]
    for n, g in few.groupby("ex_no"):
        lines.append(f"Example {n} ({g['n_options'].iloc[0]} options)")
        for _, r in g.iterrows():
            lines.append(f"  {r['option']}: {r['headline']}  [CTR {r['ctr'] * 100:.2f}%]")
        ans = g["answer"].iloc[0]
        lines.append(f"  Outcome: {'TIE' if ans == 'TIE' else 'Winner ' + ans}")
        lines.append("")
    (DATA / "examples_30.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"train pool: {tests['label'].value_counts().to_dict()}")
    return pool


# ---------------------------------------------------------------- train 100 + memorization check
def build_train100_and_memorization(pool, quiz_detail):
    rng = np.random.default_rng(20261103)
    pool = pool[~pool["in_examples"]]
    tests = pool.drop_duplicates("test_id")
    win_ids = list(tests[tests["label"] == "winner"]["test_id"])
    tie_all = tests[tests["label"] == "no_difference"]
    per = {k: list(rng.permutation(tie_all[tie_all["n_options"] == k]["test_id"])) for k in (2, 3, 4)}
    tie_ids = []
    while len(tie_ids) < 100 - len(win_ids):
        for k in (2, 3, 4):
            if per[k] and len(tie_ids) < 100 - len(win_ids):
                tie_ids.append(per[k].pop())
    ids = list(rng.permutation(win_ids + tie_ids))
    quiz, key = [], []
    for n, tid in enumerate(ids, 1):
        g = pool[pool["test_id"] == tid].sort_values("option")
        qid = f"L{n:03d}"
        row = dict(id=qid, n_options=len(g))
        for L in LET:
            h = g[g["option"] == L]["headline"]
            row[f"headline_{L}"] = h.iloc[0] if len(h) else ""
        quiz.append(row)
        key.append(dict(id=qid, test_id=tid, n_options=len(g), label=g["label"].iloc[0], answer=g["answer"].iloc[0]))
    pd.DataFrame(quiz).to_csv(DATA / "quiz_train_100_blind.csv", index=False)
    pd.DataFrame(key).to_csv(DATA / "quiz_train_100_answer_key.csv", index=False)

    det = quiz_detail[quiz_detail["headline"].str.split().str.len() >= 8].copy()
    det["headline"] = det["headline"].str.replace("<br>", " ", regex=False).str.strip()
    picked = pd.concat([det[det["kind"] == "winner"].sample(15, random_state=1),
                        det[det["kind"] == "no_difference"].sample(15, random_state=2)])
    picked = picked.sample(frac=1, random_state=3).reset_index(drop=True)
    ck = []
    for i, r in picked.iterrows():
        w = r["headline"].split()
        cut = max(4, len(w) // 2)
        ck.append(dict(id=f"C{i + 1:02d}", quiz_id=r["id"], start=" ".join(w[:cut]), true_rest=" ".join(w[cut:])))
    ck = pd.DataFrame(ck)
    ck.to_csv(DATA / "memorization_key.csv", index=False)
    return ck


# ---------------------------------------------------------------- prompts
TASK = """You will predict the outcome of real headline tests from Upworthy, a US viral-news website (2013-2015).
In each test, real Facebook-era readers were randomly split, and each group saw one headline option. The option with the highest click-through rate (CTR) won.

The attached file {file} has {n} tests. Columns: id, n_options, headline_A, headline_B, headline_C, headline_D (empty if the test had fewer options).

For each test:
1. Give every option a score from 0 to 100 for how likely a typical reader is to click it.
2. Then give your answer: the letter of the option that clearly won, or TIE if the options would perform about the same.

Rules:
- Judge each test on its own. Do not try to guess patterns in the answer letters.
- Do not search the web or look up the original articles.
- Leave the score blank for options that do not exist.

Output only one CSV code block with this header, one row per test, all {n} rows, in the same order as the file:
id,score_A,score_B,score_C,score_D,answer
If you run out of space, stop at the end of a row; I will type "continue" and you continue from the next row without repeating the header."""


def build_prompts(ck):
    base = TASK.format(file="quiz_95_blind.csv", n=95)
    (PROMPTS / "1_baseline.txt").write_text(base + "\n", encoding="utf-8")
    ex = (DATA / "examples_30.txt").read_text(encoding="utf-8").strip()
    with_ex = base.replace("The attached file quiz_95_blind.csv",
                           "First, study these 30 solved examples from other Upworthy tests. They are not in the quiz.\n\n"
                           "=== EXAMPLES START ===\n" + ex + "\n=== EXAMPLES END ===\n\nNow the quiz. The attached file quiz_95_blind.csv", 1)
    (PROMPTS / "2_with_30_examples.txt").write_text(with_ex + "\n", encoding="utf-8")
    (PROMPTS / "3_train_100.txt").write_text(TASK.format(file="quiz_train_100_blind.csv", n=100) + "\n", encoding="utf-8")
    starts = "\n".join(f"{r['id']}: {r['start']} ..." for _, r in ck.iterrows())
    mem = f"""Below are the beginnings of 30 headlines. Some may be real headlines from Upworthy (a US viral-news site, 2013-2015).

For each one:
1. If you recognize the exact headline from memory, write the rest of it exactly as you remember it.
2. If you do not recognize it, write the most natural ending you can.
3. Say whether you recognized it: YES or NO.

Do not search the web.

{starts}

Output only one CSV code block with this header, one row per line, all 30 rows:
id,recognized,rest_of_headline
Put rest_of_headline in double quotes."""
    (PROMPTS / "4_memorization_check.txt").write_text(mem + "\n", encoding="utf-8")


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    PROMPTS.mkdir(exist_ok=True)
    raw, df = load(sys.argv[1])
    detail = build_quiz(df)
    pool = build_train(raw, df, detail)
    ck = build_train100_and_memorization(pool, detail)
    build_prompts(ck)
    print("done")
