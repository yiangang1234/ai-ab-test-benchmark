"""Reproduce every number in the README from data/ and results/.

Usage (from the repo root):  python scripts/analyze.py
Writes results/summary.txt and prints the same text.
"""
import difflib
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact

ROOT = Path(__file__).resolve().parent.parent
D, R = ROOT / "data", ROOT / "results"
LET = "ABCD"
out = []


def say(s=""):
    print(s)
    out.append(s)


def pct(x):
    return f"{100 * x:.1f}%"


def load_runs(key, files):
    runs = {}
    for f in files:
        p = pd.read_csv(R / f)
        assert len(p) == len(key) and set(p["id"]) == set(key["id"]), f
        runs[f] = key.merge(p, on="id", suffixes=("_true", "_ai"))
    return runs


def gap(r):
    v = sorted([r[f"score_{L}"] for L in LET if not pd.isna(r[f"score_{L}"])], reverse=True)
    return v[0] - v[1]


def rule(r, t):
    s = {L: r[f"score_{L}"] for L in LET if not pd.isna(r[f"score_{L}"])}
    return "TIE" if gap(r) <= t else max(s, key=s.get)


def consistency(runs, truth):
    A = pd.DataFrame({n: m.set_index("id")["answer_ai"] for n, m in runs.items()})
    t = truth.reindex(A.index)
    same = A.nunique(axis=1) == 1
    first = A.iloc[:, 0]
    vote = A.apply(lambda r: (lambda c: c[0][0] if c[0][1] >= 2 else "TIE")(Counter(r).most_common()), axis=1)
    ok_same = first[same] == t[same]
    ok_split = vote[~same] == t[~same]
    tie3 = same & (first == "TIE")
    let3 = same & (first != "TIE")
    return dict(same_n=int(same.sum()), same_acc=ok_same.mean(), split_n=int((~same).sum()), split_acc=ok_split.mean(),
                fisher_p=fisher_exact([[ok_same.sum(), (~ok_same).sum()], [ok_split.sum(), (~ok_split).sum()]], alternative="greater")[1],
                tie3_n=int(tie3.sum()), tie3_ok=int((first[tie3] == t[tie3]).sum()),
                let3_n=int(let3.sum()), let3_ok=int((first[let3] == t[let3]).sum()), vote_acc=(vote == t).mean())


# ------------------------------------------------------------------ data
allt = pd.read_csv(D / "all_2to4_option_tests.csv")
n = len(allt)
win = (allt["p_top"] < 0.001).sum()
nodiff = (allt["p_all"] > 0.5).sum()
say("== Test pool (2-4 option headline tests, 1,000+ impressions per option)")
say(f"tests: {n}  by options: {allt['k'].value_counts().sort_index().to_dict()}")
say(f"clear winner (runner-up p < 0.001): {win} ({pct(win / n)})")
say(f"no detectable difference (chi-square p > 0.5): {nodiff} ({pct(nodiff / n)})")
say(f"inconclusive (neither): {n - win - nodiff} ({pct((n - win - nodiff) / n)})")

# ------------------------------------------------------------------ quiz 95
key = pd.read_csv(D / "quiz_95_answer_key.csv")
w = key["kind"] == "winner"
say("\n== Quiz: 95 tests (45 clear winner, 50 no detectable difference)")
rand = (1 / (key["n_options"] + 1)).mean()
say(f"baseline random answer (options + TIE): {pct(rand)}   always 'TIE': {pct((~w).mean())}   random pick on winner tests: {pct((1 / key[w]['n_options']).mean())}")
base_runs = load_runs(key, ["quiz95_baseline_run1.csv", "quiz95_baseline_run2.csv", "quiz95_baseline_run3.csv"])
rows = []
for name, m in base_runs.items():
    ok = m["answer_true"] == m["answer_ai"]
    rows.append(dict(run=name, all=ok.mean(), winner=ok[w].mean(), no_diff=ok[~w].mean(), tie_answers=int((m["answer_ai"] == "TIE").sum())))
t = pd.DataFrame(rows)
for _, r in t.iterrows():
    say(f"{r['run']}: all {pct(r['all'])}  winner {pct(r['winner'])}  no-difference {pct(r['no_diff'])}  TIE answers {r['tie_answers']}")
say(f"mean of 3 runs: all {pct(t['all'].mean())}  winner {pct(t['winner'].mean())} (range {pct(t['winner'].min())}-{pct(t['winner'].max())})  no-difference {pct(t['no_diff'].mean())}")
correct = int(sum((m["answer_true"] == m["answer_ai"]).sum() for m in base_runs.values()))
say(f"285 answers, {correct} correct; one-sided binomial vs random answer: p = {binomtest(correct, 285, rand, alternative='greater').pvalue:.1e}")

ex = load_runs(key, ["quiz95_examples_run1.csv"])["quiz95_examples_run1.csv"]
b1 = base_runs["quiz95_baseline_run1.csv"]
okb, oke = b1["answer_true"] == b1["answer_ai"], ex["answer_true"] == ex["answer_ai"]
bb, cc = int((okb & ~oke).sum()), int((~okb & oke).sum())
say(f"\n30 solved examples: {pct(oke.mean())} vs baseline run1 {pct(okb.mean())}; only-baseline {bb}, only-examples {cc}, McNemar p = {binomtest(min(bb, cc), bb + cc, 0.5).pvalue:.2f}")

truth95 = key.set_index("id")["answer"]
c95 = consistency(base_runs, truth95)

# ------------------------------------------------------------------ train 100
k100 = pd.read_csv(D / "quiz_train_100_answer_key.csv")
w100 = k100["label"] == "winner"
tr_runs = load_runs(k100, ["train100_baseline_run1.csv", "train100_baseline_run2.csv", "train100_baseline_run3.csv"])
say("\n== Train set: 100 different tests (36 winner at p < 0.01, 64 no detectable difference)")
for name, m in tr_runs.items():
    ok = m["answer_true"] == m["answer_ai"]
    say(f"{name}: all {pct(ok.mean())}  winner {pct(ok[w100].mean())}  no-difference {pct(ok[~w100].mean())}  TIE answers {int((m['answer_ai'] == 'TIE').sum())}")

m = tr_runs["train100_baseline_run1.csv"].copy()
m["gap"] = m.apply(gap, axis=1)
say(f"\nscore gap (top - 2nd), run1: winner tests {m[w100]['gap'].mean():.1f}, no-difference tests {m[~w100]['gap'].mean():.1f}")
say("threshold t (gap <= t -> TIE): balanced accuracy")
best = None
for th in range(-1, 16):
    ok = m.apply(lambda r: rule(r, th), axis=1) == m["answer_true"]
    bal = (ok[w100].mean() + ok[~w100].mean()) / 2
    say(f"  t={th:2d}: winner {pct(ok[w100].mean())}  no-diff {pct(ok[~w100].mean())}  balanced {pct(bal)}")
    if best is None or bal > best[1] + 1e-9:
        best = (th, bal)
q = b1.copy()
okt = q.apply(lambda r: rule(r, best[0]), axis=1) == q["answer_true"]
say(f"best t={best[0]} applied once to quiz baseline run1: {pct(okb.mean())} -> {pct(okt.mean())}")

c100 = consistency(tr_runs, k100.set_index("id")["answer"])
say("\n== Consistency across 3 fresh chats")
for name, c, base_rate in (("quiz 95", c95, (~w).mean()), ("train 100", c100, (~w100).mean())):
    say(f"{name}: same answer 3/3 -> {c['same_n']} tests, {pct(c['same_acc'])} correct | split -> {c['split_n']} tests, majority {pct(c['split_acc'])} | Fisher p = {c['fisher_p']:.4f}")
    say(f"   'no difference' 3/3 -> {c['tie3_ok']}/{c['tie3_n']} ({pct(c['tie3_ok'] / c['tie3_n'])}) real test also found no detectable difference (base rate {pct(base_rate)})")
    say(f"   same letter 3/3 -> {c['let3_ok']}/{c['let3_n']} ({pct(c['let3_ok'] / c['let3_n'])}) correct | majority vote overall {pct(c['vote_acc'])}")

# ------------------------------------------------------------------ equivalence of "no detectable difference"
say("\n== Are the 'no detectable difference' tests equivalent? (best vs worst option)")
z90, z95 = 1.6449, 1.96


def equiv(det, idcol):
    res = []
    for _, g in det.groupby(idcol):
        g = g.assign(ctr=g["clicks"] / g["impressions"]).sort_values("ctr", ascending=False)
        a, b = g.iloc[0], g.iloc[-1]
        p1, n1, p2, n2 = a["ctr"], a["impressions"], b["ctr"], b["impressions"]
        se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        sel = np.sqrt((1 - p1) / (n1 * p1) + (1 - p2) / (n2 * p2))
        res.append(dict(ctr=g["ctr"].mean(), diff=p1 - p2, half95=z95 * se, rel=p1 / p2 - 1,
                        rel90_hi=np.exp(np.log(p1 / p2) + z90 * sel) - 1))
    return pd.DataFrame(res)


qd = pd.read_csv(D / "quiz_95_all_options.csv")
tp = pd.read_csv(D / "train_pool.csv")
for name, d in (("quiz 95 (50 tests)", equiv(qd[qd["kind"] == "no_difference"], "id")),
                ("train 100 (64 tests)", equiv(tp[tp["test_id"].isin(k100[~w100]["test_id"])], "test_id"))):
    say(f"{name}: median CTR {pct(d['ctr'].median())}, median best-worst gap {100 * d['diff'].median():.2f}pp, "
        f"median 95% CI half-width +/-{100 * d['half95'].median():.2f}pp, median upper 90% CI of relative lift +{100 * d['rel90_hi'].median():.0f}%")
    say("   equivalent by TOST (upper 90% CI of relative lift below margin): " +
        ", ".join(f"+/-{int(mg * 100)}%: {(d['rel90_hi'] < mg).sum()}/{len(d)}" for mg in (0.1, 0.2, 0.3, 0.5)))

# ------------------------------------------------------------------ memorization
mk = pd.read_csv(D / "memorization_key.csv").merge(pd.read_csv(R / "memorization_check.csv"), on="id")
norm = lambda s: re.sub(r"[^a-z0-9 ]", "", str(s).lower()).split()
mk["sim"] = mk.apply(lambda r: difflib.SequenceMatcher(None, norm(r["true_rest"]), norm(r["rest_of_headline"])).ratio(), axis=1)
say("\n== Memorization check (30 quiz headlines, first half given)")
say(f"said recognized: {(mk['recognized'] == 'YES').sum()}/30  exact: {(mk['sim'] == 1).sum()}  similarity >= 0.8: {(mk['sim'] >= 0.8).sum()}  >= 0.5: {(mk['sim'] >= 0.5).sum()}  mean {mk['sim'].mean():.2f}")

(R / "summary.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
