# 발표자료/diagnose_market_bigmove_ratio.py — 횡단면 급변 비율 피처 상관 진단(2026-09-27, 재학습 없음, train만)
# market_bigmove_ratio_t = 그날 급변일(|r| >= 2x 직전60일평균)인 종목 비율(60일 평균이 정의된 종목만 분모).
# 행 t에는 전날 값(ratio_(t-1))을 붙인다 — 다른 15피처와 같은 "행 t = t-1까지 정보" 원칙(당일 값은 누수).
# train 행은 make_feature_scatter_grid.py가 만든 캐시(_cache_train_rows.parquet) 재사용.
import os
from datetime import date
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
from 가격예측.garch_baseline import compute_log_returns_pct, load_close_prices
from 가격예측.검증용.highvol_regime_diagnosis import compute_big_move_labels

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10})
tr = pd.read_parquet(os.path.join(OUT, "_cache_train_rows.parquet")); tr["date"] = pd.to_datetime(tr.date)

labs = {}
for t in ACTIVE_TICKERS:
    ret = compute_log_returns_pct(load_close_prices(t, STOCK_INITIAL_LOAD_START, date.today().isoformat()))
    trail = ret.abs().shift(1).rolling(60, min_periods=60).mean()
    labs[t] = compute_big_move_labels(ret).where(trail.notna())          # 60일 평균 미정의 날은 NaN(분모 제외)
L = pd.DataFrame(labs).astype(float)                                      # index=날짜, columns=종목
ratio = L.mean(axis=1, skipna=True)                                       # 당일 횡단면 급변 비율
cal = ratio.dropna()
ratio_prev = cal.shift(1)                                                 # 공유 거래일 캘린더 기준 전날 값
tr["market_bigmove_ratio"] = ratio_prev.reindex(tr.date).values
tr["bigmove"] = [bool(labs[t].get(d, False) == 1) for t, d in zip(tr.ticker, tr.date)]
tr = tr.dropna(subset=["market_bigmove_ratio"])
# 누수 점검: 행 t의 값이 t 당일 비율과 같으면 안 된다
same = np.isclose(tr.market_bigmove_ratio.values, ratio.reindex(tr.date).values).mean()
print(f"train 행 {len(tr):,} | 피처 분포: mean {tr.market_bigmove_ratio.mean():.3f} p50 {tr.market_bigmove_ratio.median():.3f} "
      f"p95 {tr.market_bigmove_ratio.quantile(.95):.3f} max {tr.market_bigmove_ratio.max():.3f} | (참고) 당일값과 우연 일치 비율 {same:.3f}")

feats = [c for c in tr.columns if c not in ("date", "ticker", "target", "close_prev", "bigmove")]
top = tr.target > tr.target.quantile(0.8)
G = {"전체 train": np.ones(len(tr), bool), "① 급변일(상위20% ∩ 2x)": (top & tr.bigmove).values, "③ 평상일(하위80%)": (~top).values}
rows = []
for f in feats:
    rec = {"feature": f}
    for g, m in G.items():
        x, y = tr[f].values[m], tr.target.values[m]
        rec[f"{g} P"] = pearsonr(x, y)[0] if np.std(x) > 0 else np.nan
        rec[f"{g} S"] = spearmanr(x, y)[0] if np.std(x) > 0 else np.nan
    rows.append(rec)
T = pd.DataFrame(rows).set_index("feature")
T["순위(전체 |S|)"] = T["전체 train S"].abs().rank(ascending=False).astype(int)
T["순위(① |S|)"] = T["① 급변일(상위20% ∩ 2x) S"].abs().rank(ascending=False).astype(int)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print("\n[상관표 — train만, |Spearman| 전체 순]"); print(T.sort_values("전체 train S", key=abs, ascending=False).round(4).to_string())
T.to_csv(os.path.join(OUT, "market_bigmove_ratio_corr.csv"), float_format="%.4f")

# 방향성 질문: 전날 시장 급변 비율이 높을수록 오늘 이 종목이 급변일일 확률이 올라가는가(분위 구간별 조건부 확률)
q = pd.qcut(tr.market_bigmove_ratio, 10, duplicates="drop")
lift = tr.groupby(q, observed=True).agg(n=("bigmove", "size"), 급변확률=("bigmove", "mean"), 타겟평균=("target", "mean"))
lift["배율(기저 대비)"] = lift.급변확률 / tr.bigmove.mean()
print(f"\n[전날 시장 급변 비율 10분위별 — 오늘 급변일 확률(기저 {tr.bigmove.mean():.3f})·타겟 평균]"); print(lift.round(3).to_string())

C = tr[feats + ["target"]].corr()
fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(C))); ax.set_yticks(range(len(C))); ax.set_xticklabels(C.columns, rotation=90); ax.set_yticklabels(C.index)
for i in range(len(C)):
    for j in range(len(C)):
        ax.text(j, i, f"{C.values[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if abs(C.values[i, j]) > 0.6 else "black")
fig.colorbar(im, shrink=0.7, label="Pearson")
ax.set_title(f"train-only 상관행렬 — 15피처 + market_bigmove_ratio(전날 값) + target (n={len(tr):,})", loc="left")
p = os.path.join(OUT, "fig8_corr_with_market_bigmove_ratio.png"); fig.tight_layout(); fig.savefig(p, dpi=150); print(p)
print("다중공선성(새 피처와 |r|>0.3):", {k: round(v, 3) for k, v in C["market_bigmove_ratio"].drop(["market_bigmove_ratio", "target"]).items() if abs(v) > 0.3})
