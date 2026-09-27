# 발표자료/make_close_vs_target.py — 전일 종가(원) vs 실측 타겟(train) 4패널(2026-09-27, EDA)
# (a) 선형 x (b) 로그 x (c) 종목 간: 종목별 평균 log10 종가 vs 평균 타겟(100점) (d) 종목 내: 종목별 평균 제거 후.
# 캐시(_cache_train_rows.parquet, close_prev 포함)는 make_feature_scatter_grid.py가 만든다.
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr, spearmanr
from constants import STOCK_NAMES

INK, INK2, SURF, BLUE, ORANGE = "#0b0b0b", "#52514e", "#ffffff", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 12, "figure.facecolor": SURF})
OUT = os.path.dirname(os.path.abspath(__file__))
tr = pd.read_parquet(os.path.join(OUT, "_cache_train_rows.parquet")).dropna(subset=["close_prev"])
y, c = tr.target.values, tr.close_prev.values
lc = np.log10(c); ymax = float(np.percentile(y, 99.5))


def med_line(ax, x, yy, n=20, color=ORANGE):
    q = np.unique(np.quantile(x, np.linspace(0, 1, n + 1))); b = np.clip(np.digitize(x, q[1:-1]), 0, len(q) - 2)
    g = pd.DataFrame({"b": b, "x": x, "y": yy}).groupby("b").median()
    ax.plot(g.x, g.y, color=color, lw=2.4, marker="o", ms=4, zorder=5)


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=13, color=INK)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


def won_ticks(ax, lo, hi):
    tk = [v for v in (1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6) if lo <= np.log10(v) <= hi]
    ax.set_xticks(np.log10(tk)); ax.set_xticklabels([f"{v/1e4:g}만" if v >= 1e4 else f"{v:,.0f}" for v in tk])


fig, axes = plt.subplots(2, 2, figsize=(17, 13))
# (a) 선형
ax = axes[0, 0]; hi = np.percentile(c, 99.5); m = (c <= hi) & (y <= ymax)
ax.hexbin(c[m], y[m], gridsize=55, cmap="Blues", norm=LogNorm(vmin=1), mincnt=1, linewidths=0); med_line(ax, c, y)
ax.set_xlim(0, hi); ax.set_ylim(0, ymax); ax.set_xlabel("전일 종가(원)"); ax.set_ylabel("실측 타겟 |로그수익률|×100")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e4:g}만"))
style(ax, f"(a) 선형 x — 고가 종목에 눌려 대부분이 왼쪽에 몰림 (표시 밖 {1 - m.mean():.1%})")
# (b) 로그
ax = axes[0, 1]; lo, hi2 = np.percentile(lc, [0.5, 99.5]); m = (lc >= lo) & (lc <= hi2) & (y <= ymax)
hb = ax.hexbin(lc[m], y[m], gridsize=55, cmap="Blues", norm=LogNorm(vmin=1), mincnt=1, linewidths=0); med_line(ax, lc, y)
ax.set_xlim(lo, hi2); ax.set_ylim(0, ymax); won_ticks(ax, lo, hi2); ax.set_xlabel("전일 종가(원, 로그 스케일)")
style(ax, f"(b) 로그 x — 전체: Pearson(log) {pearsonr(lc, y)[0]:+.3f}, Spearman {spearmanr(c, y)[0]:+.3f}")
# (c) 종목 간
g = tr.assign(lc=lc).groupby("ticker").agg(lc=("lc", "mean"), y=("target", "mean"))
ax = axes[1, 0]; ax.scatter(g.lc, g.y, s=46, color=BLUE, alpha=0.8, edgecolors="white", linewidths=0.8, zorder=3)
k = np.polyfit(g.lc, g.y, 1); xs = np.linspace(g.lc.min(), g.lc.max(), 50); ax.plot(xs, np.polyval(k, xs), color=ORANGE, lw=2)
for t in pd.concat([g.y.nlargest(3), g.y.nsmallest(2)]).index.union(g.lc.nlargest(2).index).union(g.lc.nsmallest(2).index):
    ax.annotate(STOCK_NAMES.get(t, t), (g.lc[t], g.y[t]), xytext=(5, 4), textcoords="offset points", fontsize=10, color=INK2)
won_ticks(ax, g.lc.min() - 0.05, g.lc.max() + 0.05); ax.set_xlabel("종목별 평균 전일 종가(원, 로그)"); ax.set_ylabel("종목별 평균 타겟")
style(ax, f"(c) 종목 간(100종목, 1점=1종목) — Pearson {pearsonr(g.lc, g.y)[0]:+.3f}, Spearman {spearmanr(g.lc, g.y)[0]:+.3f}")
# (d) 종목 내
d = tr.assign(lc=lc); d["lc_dm"] = d.lc - d.groupby("ticker").lc.transform("mean"); d["y_dm"] = d.target - d.groupby("ticker").target.transform("mean")
x4, y4 = d.lc_dm.values, d.y_dm.values
lo4, hi4 = np.percentile(x4, [0.5, 99.5]); ylo, yhi = np.percentile(y4, [0.5, 99.5]); m = (x4 >= lo4) & (x4 <= hi4) & (y4 >= ylo) & (y4 <= yhi)
ax = axes[1, 1]; ax.hexbin(x4[m], y4[m], gridsize=55, cmap="Blues", norm=LogNorm(vmin=1), mincnt=1, linewidths=0); med_line(ax, x4, y4)
ax.axhline(0, color=INK2, lw=0.8); ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlim(lo4, hi4); ax.set_ylim(ylo, yhi)
ax.set_xlabel("종목 내 log10 종가 - 종목 평균 (+0.3 = 약 2배, -0.3 = 약 절반)"); ax.set_ylabel("타겟 - 종목 평균")
style(ax, f"(d) 종목 내(각 종목 평균 제거) — Pearson {pearsonr(x4, y4)[0]:+.3f}, Spearman {spearmanr(x4, y4)[0]:+.3f}")
fig.subplots_adjust(right=0.9, hspace=0.25, wspace=0.18)
fig.colorbar(hb, cax=fig.add_axes([0.925, 0.35, 0.012, 0.3]), label="칸당 행 수(로그 색)")
fig.suptitle("전일 종가 vs 실측 변동성(train) — 가격 레벨 효과는 종목 간 차이인가, 종목 내 시간 변화인가",
             x=0.01, ha="left", fontsize=17, color=INK)
fig.text(0.01, 0.005, f"train ~2024-10-10 · 100종목 · n={len(tr):,} · 전일 종가 = close(t-1) · 주황선: (a)(b)(d) x 분위 20구간 중앙값, "
         "(c) 최소제곱 직선 · 표시 범위 0.5~99.5%(선·지표는 전체로 계산)", fontsize=10.5, color=INK2)
p = os.path.join(OUT, "fig7_close_vs_target.png"); fig.savefig(p, dpi=160, bbox_inches="tight"); print(p)
