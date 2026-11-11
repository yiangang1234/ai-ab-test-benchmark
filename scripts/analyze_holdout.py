"""Pre-registered analysis of the natural-mix holdout (see PREREGISTRATION.md).

Usage (from the repo root), after all 6 runs are saved and holdout_key/ is published:
    python scripts/analyze_holdout.py
Reads  results/holdout_{A,B}_run{1,2,3}.csv  and  holdout_key/*.csv
Writes results/holdout_summary.txt
"""
from collections import Counter
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
R, KEY = ROOT / "results", ROOT / "holdout_key"
LET = "ABCD"
POP = dict(winner=0.041, no_difference=0.201)   # shares in the full confirmatory pool (5,097 tests), fixed before the runs
MARGIN = 0.20                                   # equivalence margin, relative CTR, fixed before the runs
out = []


def say(s=""):
    print(s)
    out.append(s)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


key = pd.read_csv(KEY / "holdout_answer_key.csv").set_index("id")
opts = pd.read_csv(KEY / "holdout_all_options.csv")
opts["ctr"] = opts["clicks"] / opts["impressions"]

runs = []
for r in (1, 2, 3):
    parts = [pd.read_csv(R / f"holdout_{p}_run{r}.csv") for p in ("A", "B")]
    run = pd.concat(parts).set_index("id")
    assert len(run) == 200 and set(run.index) == set(key.index), f"run {r} incomplete"
    runs.append(run)
ans = pd.DataFrame({f"run{i + 1}": r["answer"].astype(str).str.strip().str.upper() for i, r in enumerate(runs)})
lab = key.loc[ans.index, "label"]

say("== Sample: 200 tests drawn at random from the confirmatory pool (natural mix)")
say(f"labels: {lab.value_counts().to_dict()}  | pool shares: winner {POP['winner']:.1%}, no detectable difference {POP['no_difference']:.1%}")

# ---------- equivalence (TOST, best vs worst option, relative CTR, 90% CI) for every test
z90 = 1.6449


def equivalent(i):
    g = opts[opts["id"] == i].sort_values("ctr", ascending=False)
    a, b = g.iloc[0], g.iloc[-1]
    if b["ctr"] == 0:
        return False
    sel = sqrt((1 - a["ctr"]) / (a["impressions"] * a["ctr"]) + (1 - b["ctr"]) / (b["impressions"] * b["ctr"]))
    return np.exp(np.log(a["ctr"] / b["ctr"]) + z90 * sel) - 1 < MARGIN


eq = pd.Series({i: equivalent(i) for i in ans.index})
say(f"equivalent within +/-{MARGIN:.0%} (TOST): {int(eq.sum())}/200")

# ---------- primary 1: tests the AI called TIE in all 3 runs
flag = (ans == "TIE").all(axis=1)
nf = int(flag.sum())
say(f"\n== Primary 1: AI said 'no difference' in all 3 runs -> {nf} tests ({nf / 200:.1%})")
if nf:
    for name, base in (("no_difference", POP["no_difference"]), ("winner", POP["winner"])):
        k = int((lab[flag] == name).sum())
        lo, hi = wilson(k, nf)
        alt = "greater" if name == "no_difference" else "less"
        p = binomtest(k, nf, base, alternative=alt).pvalue
        say(f"  {name}: {k}/{nf} = {k / nf:.1%} (95% CI {lo:.1%}-{hi:.1%}) vs pool {base:.1%}, one-sided ({alt}) p = {p:.3f}")
    k = int(eq[flag].sum())
    say(f"  equivalent within +/-{MARGIN:.0%}: {k}/{nf} = {k / nf:.1%} vs {eq.mean():.1%} in the whole sample")
    say(f"  inconclusive: {int((lab[flag] == 'inconclusive').sum())}/{nf}")
say(f"  clear winners the AI flagged (would be missed by skipping): {int((flag & (lab == 'winner')).sum())} of {int((lab == 'winner').sum())}")

# ---------- primary 2: clear-winner tests
w = lab == "winner"
say(f"\n== Primary 2: {int(w.sum())} clear-winner tests, AI picked the true best option")
accs = [(ans.loc[w, c] == key.loc[ans.index[w], "best_option"]).mean() for c in ans]
say("  per run: " + ", ".join(f"{a:.0%}" for a in accs) + f" | mean {np.mean(accs):.1%} | random pick {np.mean(1 / key.loc[w, 'n_options']):.1%}")


# ---------- secondary 3: CTR lift of the AI's top-scored option, all 200 tests
def pick_lift(run):
    lifts = []
    for i, r in run.iterrows():
        g = opts[opts["id"] == i].set_index("option")["ctr"]
        sc = {L: r[f"score_{L}"] for L in LET if L in g.index and not pd.isna(r.get(f"score_{L}"))}
        if not sc:
            continue
        top = max(sc.values())
        chosen = [L for L, v in sc.items() if v == top]
        lifts.append(g[chosen].mean() / g.mean() - 1)
    return np.array(lifts)


L = np.mean([pick_lift(r) for r in runs], axis=0)
rng = np.random.default_rng(20261111)
boot = [rng.choice(L, size=len(L), replace=True).mean() for _ in range(2000)]
say(f"\n== Secondary 3: CTR of the AI's top-scored option vs the average option (all {len(L)} tests, mean of 3 runs)")
say(f"  mean relative lift {L.mean():+.2%} (bootstrap 95% CI {np.percentile(boot, 2.5):+.2%} to {np.percentile(boot, 97.5):+.2%})")

# ---------- secondary 4-5: descriptive
same = ans.nunique(axis=1) == 1
say(f"\n== Secondary 4: same answer in all 3 runs: {int(same.sum())}/200")
known = lab != "inconclusive"
truth = key.loc[ans.index, "answer"]
acc = [(ans.loc[known, c] == truth[known]).mean() for c in ans]
say(f"== Secondary 5: accuracy on the {int(known.sum())} non-inconclusive tests: " + ", ".join(f"{a:.0%}" for a in acc))
say(f"TIE answers per run: {[int((ans[c] == 'TIE').sum()) for c in ans]}")

(R / "holdout_summary.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
