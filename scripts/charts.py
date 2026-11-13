"""Draw the three charts in charts/. Numbers come from results/summary.txt (python scripts/analyze.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch

for p in fm.findSystemFonts():
    if "Inter" in p:  # uses Inter if installed, otherwise the default font
        fm.fontManager.addfont(p)
plt.rcParams["font.family"] = "Inter"

SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8a8984"; GRID = "#e6e5e1"
S1 = "#2a78d6"; S2 = "#eb6834"; NEUTRAL = "#b9b8b2"
from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "charts"

def base(title, subtitle, foot="Data: Upworthy Research Archive (Matias et al., 2021), exploratory set, 2–4-option headline tests. Same prompt, 3 fresh chats."):
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=SURF)
    fig.text(0.06, 0.92, title, fontsize=34, weight="bold", color=INK, va="top")
    fig.text(0.06, 0.845, subtitle, fontsize=19, color=INK2, va="top")
    fig.text(0.06, 0.03, foot, fontsize=13, color=MUTED)
    return fig

def rbar(ax, y, w, h, color):
    # 4px 둥근 끝(데이터 쪽), 기준선 쪽은 각지게
    ax.barh(y, w, height=h, color=color, edgecolor=SURF, linewidth=2)

def style(ax, xmax):
    ax.set_facecolor(SURF)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xlim(0, xmax); ax.xaxis.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=MUTED, labelsize=14, length=0)
    ax.tick_params(axis="y", colors=INK, labelsize=19, length=0, pad=12)

# 1) 대부분의 테스트는 승자가 없다
fig = base("Most headline tests don't have a real winner",
           "1,123 real A/B/C/D tests, each option shown to 1,000+ readers",
           foot="Data: Upworthy Research Archive (Matias et al., 2021), exploratory set, 2–4-option headline tests, 2013–2015.")
ax = fig.add_axes([0.36, 0.18, 0.56, 0.58])
labels = ["Clear winner", "No detectable\ndifference", "Inconclusive"]
vals = [45, 231, 847]; pct = [v / 1123 for v in vals]
cols = [S1, NEUTRAL, NEUTRAL]
ys = [2, 1, 0]
for y, p_, c in zip(ys, pct, cols): rbar(ax, y, p_ * 100, 0.6, c)
ax.set_yticks(ys); ax.set_yticklabels(labels)
for y, p_, v in zip(ys, pct, vals):
    ax.text(p_ * 100 + 1.2, y, f"{p_:.0%}  ({v:,})", va="center", fontsize=19, color=INK, weight="bold" if y == 2 else "normal")
style(ax, 100); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
fig.text(0.06, 0.085, "Clear winner: best option beat the runner-up at p < 0.001.  No detectable difference: chi-square across options p > 0.5 (not proof they are equal).  Inconclusive: neither.",
         fontsize=13, color=INK2)
fig.savefig(f"{OUT}/1_most_tests_no_winner.png", facecolor=SURF); plt.close(fig)

# 2) AI vs 찍기 (승자 있는 테스트만, 공정한 기준선)
fig = base("AI found the real winner ~2× more often than chance",
           "45 real tests with a clear winner (2–4 options). AI saw only the headlines")
ax = fig.add_axes([0.36, 0.30, 0.56, 0.42])
labels = ["Random pick", "AI (avg of 3 runs)"]
mean = [35.2, 65.9]; lo = [None, 60.0]; hi = [None, 71.1]
cols = [NEUTRAL, S1]; ys = [1, 0]
for y, m, c in zip(ys, mean, cols): rbar(ax, y, m, 0.55, c)
for y, m, l, h in zip(ys, mean, lo, hi):
    if l is not None:
        ax.plot([l, h], [y, y], color=INK, linewidth=2, zorder=3)
        for x in (l, h): ax.plot([x, x], [y - 0.1, y + 0.1], color=INK, linewidth=2, zorder=3)
    txt = f"{m:.0f}%" + (f"   (runs: {l:.0f}–{h:.0f}%)" if l is not None else "")
    ax.text(max(m, h or 0) + 1.5, y, txt, va="center", fontsize=21, color=INK, weight="bold" if l else "normal")
ax.set_yticks(ys); ax.set_yticklabels(labels)
style(ax, 100); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
fig.text(0.06, 0.205, "Random pick = 1 / number of options. 'No difference' answers counted as wrong. Whisker = lowest to highest run.",
         fontsize=13, color=INK2, va="top")
fig.text(0.06, 0.16, "Honest caveat: across all 95 tests (incl. 50 with no detectable difference) the AI scored 57%, while always answering\n"
         "'no difference' would score 53%. Overall accuracy alone is not the story.", fontsize=13, color=INK2, linespacing=1.5, va="top")
fig.savefig(f"{OUT}/2_ai_vs_random.png", facecolor=SURF); plt.close(fig)

# 3) 일관성 = 신뢰도
fig = base("AI said 'no difference' 3×: the real test agreed 88–94%",
           "Agreed = real test also found no detectable difference (p > 0.5). Found on 95 tests, confirmed on 100 new ones")
ax = fig.add_axes([0.36, 0.20, 0.50, 0.52])
cats = ["Said 'no difference'\nall 3 times", "Same answer\nall 3 times", "Answers split"]
a = [88.2, 67.2, 48.6]; b = [94.4, 61.1, 26.1]
na = ["15 of 17", "39 of 58", "18 of 37"]; nb = ["17 of 18", "33 of 54", "12 of 46"]
ys = [2, 1, 0]; h = 0.34
for y, va_, vb_, ta, tb in zip(ys, a, b, na, nb):
    rbar(ax, y + h / 2 + 0.01, va_, h, S1)
    rbar(ax, y - h / 2 - 0.01, vb_, h, S2)
    ax.text(va_ + 1.2, y + h / 2 + 0.01, f"{va_:.0f}%  ({ta})", va="center", fontsize=16, color=INK)
    ax.text(vb_ + 1.2, y - h / 2 - 0.01, f"{vb_:.0f}%  ({tb})", va="center", fontsize=16, color=INK)
ax.set_yticks(ys); ax.set_yticklabels(cats)
style(ax, 100); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
# 범례
fig.patches.append(FancyBboxPatch((0.36, 0.765), 0.014, 0.022, boxstyle="round,pad=0,rounding_size=0.003", color=S1, transform=fig.transFigure))
fig.text(0.38, 0.776, "First 95 tests", fontsize=16, color=INK, va="center")
fig.patches.append(FancyBboxPatch((0.52, 0.765), 0.014, 0.022, boxstyle="round,pad=0,rounding_size=0.003", color=S2, transform=fig.transFigure))
fig.text(0.54, 0.776, "100 new tests (confirmation)", fontsize=16, color=INK, va="center")
fig.text(0.36, 0.085, "% correct. Share of no-detectable-difference tests in each set: 53% and 64%, so 88–94% is above the base rate.\n"
         "Limits: the AI was this sure on only ~18% of tests, and 'no detectable difference' is not proof the headlines are equal.", fontsize=13, color=INK2, linespacing=1.5)
fig.savefig(f"{OUT}/3_consistency.png", facecolor=SURF); plt.close(fig)
print("ok")

# 4) holdout: CTR lift of the AI's top pick (numbers from results/holdout_summary.txt and README)
fig = base("Picking the AI's favorite headline: ~10% more clicks",
           "200 real tests drawn at random (natural mix), pre-registered. CTR relative to the average option in each test",
           foot="Data: Upworthy Research Archive (Matias et al., 2021), confirmatory set. AI: Claude Opus 5.5, same prompt, 3 fresh chats per test.")
ax = fig.add_axes([0.30, 0.20, 0.62, 0.55])
labels = ["Worst option (hindsight)", "Random pick", "AI's top-scored option", "Best option (hindsight)"]
vals = [-27.9, 0.0, 9.6, 32.9]
cols = [NEUTRAL, NEUTRAL, S1, NEUTRAL]
ys = [3, 2, 1, 0]
for y, v, c in zip(ys, vals, cols):
    ax.barh(y, v, height=0.55, color=c, edgecolor=SURF, linewidth=2)
lo, hi = 5.8, 13.6
ax.plot([lo, hi], [1, 1], color=INK, linewidth=2, zorder=3)
for x in (lo, hi):
    ax.plot([x, x], [0.9, 1.1], color=INK, linewidth=2, zorder=3)
for y, v in zip(ys, vals):
    txt = f"{v:+.0f}%" if v else "0%"
    if y == 1:
        txt = "+9.6%   (95% CI +5.8% to +13.6%)"
    xpos = hi + 1.5 if y == 1 else (v + 1.2 if v >= 0 else v - 1.2)
    ax.text(xpos, y, txt, va="center", ha="left" if v >= 0 else "right", fontsize=19, color=INK,
            weight="bold" if y == 1 else "normal")
ax.axvline(0, color=INK2, linewidth=1)
ax.set_yticks(ys); ax.set_yticklabels(labels)
ax.set_facecolor(SURF)
for s in ax.spines.values(): s.set_visible(False)
ax.set_xlim(-45, 60); ax.xaxis.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True)
ax.set_xticks([-40, -20, 0, 20, 40]); ax.set_xticklabels(["-40%", "-20%", "0%", "+20%", "+40%"])
ax.tick_params(axis="x", colors=MUTED, labelsize=14, length=0)
ax.tick_params(axis="y", colors=INK, labelsize=19, length=0, pad=12)
fig.text(0.06, 0.12, "Per run: +7.7%, +10.3%, +10.8%. Random pick = average option = 0%. About 29% of the gain of always picking the true best. No gain on 2-option tests (n=34).",
         fontsize=13, color=INK2)
fig.savefig(f"{OUT}/4_holdout_ai_pick_lift.png", facecolor=SURF); plt.close(fig)
