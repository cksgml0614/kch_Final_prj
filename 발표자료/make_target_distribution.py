# 발표자료/make_target_distribution.py — 변동성 타겟(|로그수익률|×100) 분포: train vs val (2026-09-27)
# 분할 경계는 운영과 동일(compute_global_split_dates). 가격 이력의 모든 거래일-종목 기준(모델 시퀀스는
# lookback 등으로 앞쪽 일부가 빠지므로 행 수가 약간 다를 수 있음). 로그판은 log(target+EPS), 종가 불변일(0) 때문에 EPS 필요.
# 실행: python -m 발표자료.make_target_distribution
import os
from datetime import date
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skew
from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
from 가격예측.garch_baseline import compute_log_returns_pct, load_close_prices
from 가격예측.pooled_dataset import compute_global_split_dates
from 가격예측.split_dataset import build_merged_dataset_v2

EPS = 0.01
INK, INK2, GRID, SURF, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 14,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF})
OUT = os.path.dirname(os.path.abspath(__file__))

end = date.today().isoformat()
ref, _ = build_merged_dataset_v2(ACTIVE_TICKERS[0], STOCK_INITIAL_LOAD_START, end)
train_end, val_end = compute_global_split_dates(ref.index)
tr, va = [], []
for t in ACTIVE_TICKERS:
    a = compute_log_returns_pct(load_close_prices(t, STOCK_INITIAL_LOAD_START, end)).abs()
    tr.append(a[a.index <= train_end].values); va.append(a[(a.index > train_end) & (a.index <= val_end)].values)
S = {"train": np.concatenate(tr), "val": np.concatenate(va)}
COL = {"train": BLUE, "val": ORANGE}
PER = {"train": f"~{train_end.date()}", "val": f"{(train_end + pd.Timedelta(days=1)).date()}~{val_end.date()}"}

qs = [1, 5, 25, 50, 75, 95, 99]
rows = []
for k, x in S.items():
    lx = np.log(x + EPS)
    rows.append({"split": k, "n": len(x), "zero(종가 불변)": f"{(x == 0).mean():.2%}", "mean": x.mean(), "std": x.std(),
                 "skew(선형)": skew(x), "skew(log)": skew(lx), **{f"p{q}": np.percentile(x, q) for q in qs}, "max": x.max()})
table = pd.DataFrame(rows).set_index("split")
pd.set_option("display.width", 250); pd.set_option("display.float_format", "{:.3f}".format)
print(table.to_string())
table.to_csv(os.path.join(OUT, "target_distribution_percentiles.csv"), float_format="%.4f")

fig, axes = plt.subplots(1, 2, figsize=(17, 6.6))
xmax = float(np.ceil(max(x.max() for x in S.values())))
bins_lin = np.arange(0, xmax + 0.25, 0.25)
bins_log = np.linspace(np.log(EPS), np.log(xmax + EPS), 90)
for k, x in S.items():
    axes[0].hist(x, bins=bins_lin, density=True, histtype="stepfilled", alpha=0.25, color=COL[k])
    axes[0].hist(x, bins=bins_lin, density=True, histtype="step", lw=1.8, color=COL[k],
                 label=f"{k} ({PER[k]}, n={len(x):,}, skew {skew(x):.2f})")
    lx = np.log(x + EPS)
    axes[1].hist(lx, bins=bins_log, density=True, histtype="stepfilled", alpha=0.25, color=COL[k])
    axes[1].hist(lx, bins=bins_log, density=True, histtype="step", lw=1.8, color=COL[k],
                 label=f"{k} (skew {skew(lx):.2f})")
axes[0].set_xlim(0, 12)
out12 = {k: (x > 12).mean() for k, x in S.items()}
axes[0].text(11.9, axes[0].get_ylim()[1] * 0.55, "x>12 꼬리: " + ", ".join(f"{k} {v:.2%}" for k, v in out12.items())
             + f"\n(최댓값 {xmax:.0f}까지, 오른쪽 그림에 포함)", ha="right", fontsize=11, color=INK2)
axes[0].set_title("선형 스케일 — 오른쪽 꼬리가 긴 분포", loc="left", fontsize=15, color=INK)
axes[0].set_xlabel("|로그수익률|×100"); axes[0].set_ylabel("밀도")
axes[1].set_title(f"로그 스케일 log(target + {EPS})", loc="left", fontsize=15, color=INK)
axes[1].set_xlabel(f"log(|로그수익률|×100 + {EPS})"); axes[1].set_ylabel("밀도")
for q in (1, 50, 99):
    v = np.log(np.percentile(S["train"], q) + EPS)
    axes[1].axvline(v, color=INK2, lw=1, ls=":"); axes[1].text(v, axes[1].get_ylim()[1] * 0.97, f" train p{q}", fontsize=10, color=INK2, va="top")
for ax in axes:
    ax.grid(True, color=GRID); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=11.5, loc="upper right" if ax is axes[0] else "upper left", bbox_to_anchor=None if ax is axes[0] else (0.03, 0.9))
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.suptitle("변동성 타겟 분포 — train vs val (100종목, 밀도 정규화)", x=0.01, ha="left", fontsize=17, color=INK)
fig.text(0.01, 0.01, f"target = |log(close_t / close_t-1)|×100 · 로그판 EPS={EPS}(종가 불변일 0 처리) · 분할 경계는 운영과 동일",
         fontsize=10.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.94))
p = os.path.join(OUT, "fig4_target_distribution.png"); fig.savefig(p, dpi=200); print(p)
