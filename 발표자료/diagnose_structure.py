# 발표자료/diagnose_structure.py — 예측 구조 진단 3종(2026-09-27): A 회귀 혼동행렬, B 피처공간 클러스터링,
# C 피처-타겟 선형/비선형 관계. 운영 LSTM(seed 42) val 예측 사용. test 미사용.
# 실행: python -m 발표자료.diagnose_structure --rows <val_diag_rows.parquet> --pred <lstm_s42.npy>
import argparse, os
from datetime import date
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
import 가격예측.가격예측_변동성_공통 as c
from 가격예측.momentum_feature import load_true_kospi
from 가격예측.pooled_dataset import compute_global_split_dates
from 가격예측.split_dataset import build_merged_dataset_v2, build_merged_dataset_v2_volatility_hybrid
from 시장지표.feature_loader import load_indicator_cache
from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START

INK, INK2, SURF = "#0b0b0b", "#52514e", "#ffffff"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 13, "figure.facecolor": SURF})
OUT = os.path.dirname(os.path.abspath(__file__))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)

ap = argparse.ArgumentParser(); ap.add_argument("--rows", required=True); ap.add_argument("--pred", required=True); a = ap.parse_args()
v = pd.read_parquet(a.rows)[["ticker", "date", "y"]].copy(); v["pred"] = np.load(a.pred)

# ── 행 단위 15피처(행 t = t-1까지 정보) 재구성 ──
start, end = STOCK_INITIAL_LOAD_START, date.today().isoformat()
ref, _ = build_merged_dataset_v2(ACTIVE_TICKERS[0], start, end)
train_end, val_end = compute_global_split_dates(ref.index)
ind = load_indicator_cache(end); tk = load_true_kospi(start, end)
frames = []
for t in ACTIVE_TICKERS:
    b = c.compute_ticker_baselines(t, start, end, train_end, val_end)
    m, _ = build_merged_dataset_v2_volatility_hybrid(t, start, end, train_end, precomputed_sigma=b["sigma_full"],
        garch_params=b["params"], precomputed_indicators=ind, true_kospi=tk)
    m = m.copy(); m["ticker"] = t; frames.append(m.rename_axis("date").reset_index())
df = pd.concat(frames)
feats = [x for x in df.columns if x not in ("date", "ticker", "target")]
tr = df[df.date <= train_end]; va = df[(df.date > train_end) & (df.date <= val_end)]
v = v.merge(va, on=["ticker", "date"], how="left")
assert v[feats].notna().all().all() and np.allclose(v.y, v.target, atol=1e-4), "val 행 정렬 실패"
print(f"train 행={len(tr)}, val 행={len(v)}, feats={len(feats)}", flush=True)

# ── A. 회귀 혼동행렬(train 타겟 분위수 경계, 실측·예측 동일 경계) ──
def conf(nb):
    edges = np.quantile(tr.target, np.linspace(0, 1, nb + 1)); edges[0], edges[-1] = -np.inf, np.inf
    ra = np.digitize(v.y, edges[1:-1]); rp = np.digitize(v.pred, edges[1:-1])
    cm = pd.crosstab(ra, rp).reindex(index=range(nb), columns=range(nb), fill_value=0)
    return cm, edges
fig, axes = plt.subplots(1, 2, figsize=(17, 7.6))
for ax, nb in zip(axes, (5, 10)):
    cm, edges = conf(nb)
    pct = cm.div(cm.sum(1), axis=0) * 100
    lab = [f"Q{i+1}\n{'<' if i == 0 else ''}{edges[i+1]:.2f}" if i < nb - 1 else f"Q{nb}\n≥{edges[i]:.2f}" for i in range(nb)]
    im = ax.imshow(pct.values, cmap="Blues", vmin=0, vmax=100)
    for i in range(nb):
        for j in range(nb):
            val = pct.values[i, j]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=11 if nb == 5 else 8.5,
                    color="white" if val > 55 else INK)
    ax.set_xticks(range(nb)); ax.set_yticks(range(nb)); ax.set_xticklabels(lab, fontsize=9 if nb == 5 else 7.5)
    ax.set_yticklabels(lab, fontsize=9 if nb == 5 else 7.5)
    ax.set_xlabel("예측 구간"); ax.set_ylabel("실측 구간")
    diag = np.trace(cm.values) / cm.values.sum()
    ax.set_title(f"{nb}분위 (행 기준 %) · 대각선 일치율 {diag:.1%} (무작위 기대 {1/nb:.0%})", loc="left", fontsize=13)
    print(f"\n[A] {nb}분위 — 대각선 일치율 {diag:.3f} | 예측 구간별 비율(열 합): "
          + " ".join(f"{x:.1%}" for x in cm.sum(0) / cm.values.sum()))
    print("    실측 구간별 비율(행 합): " + " ".join(f"{x:.1%}" for x in cm.sum(1) / cm.values.sum()))
fig.colorbar(im, ax=axes, shrink=0.8, label="행 기준 %")
fig.suptitle("회귀 혼동행렬 — 운영 LSTM(seed 42), 구간 경계는 train 타겟 분위수(실측·예측 동일)", x=0.01, ha="left", fontsize=16)
fig.text(0.01, 0.01, f"fold3 val {(train_end + pd.Timedelta(days=1)).date()} ~ {val_end.date()} · 100종목 · n={len(v):,}", fontsize=10.5, color=INK2)
p = os.path.join(OUT, "fig5_regression_confusion.png"); fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig); print(p)

# ── B. 피처공간 KMeans(train 평균/표준편차로 표준화) ──
mu, sd = tr[feats].mean(), tr[feats].std().replace(0, 1)
Z = ((v[feats] - mu) / sd).values
def eta2(x, lab):
    g = pd.Series(x).groupby(lab); return float((g.size() * (g.mean() - x.mean()) ** 2).sum() / ((x - x.mean()) ** 2).sum())
print("\n[B] KMeans — eta²(클러스터가 설명하는 분산 비율): 실측 / 예측")
for k in (3, 4, 5):
    lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z)
    tab = v.assign(cl=lab, resid=v.pred - v.y).groupby("cl").agg(n=("y", "size"), 실측평균=("y", "mean"), 실측중앙=("y", "median"),
          예측평균=("pred", "mean"), 예측중앙=("pred", "median"), 잔차평균=("resid", "mean"))
    tab["비중"] = tab.n / len(v)
    top = pd.DataFrame(Z, columns=feats).groupby(lab).mean()
    tab["대표 피처(표준화 평균 |z| 상위2)"] = [", ".join(f"{f}{r[f]:+.1f}" for f in r.abs().sort_values(ascending=False).index[:2]) for _, r in top.iterrows()]
    print(f"\n k={k}: eta² 실측={eta2(v.y.values, lab):.3f}, 예측={eta2(v.pred.values, lab):.3f}")
    print(tab.sort_values("실측평균").round(3).to_string())

# ── C. 피처-타겟 선형/비선형 관계(train, MI는 3만 행 무작위 표본) ──
rng = np.random.default_rng(0)
idx = rng.choice(len(tr), size=min(30000, len(tr)), replace=False)
mi = mutual_info_regression(tr[feats].values[idx], tr.target.values[idx], random_state=0)
rows = []
for f, m in zip(feats, mi):
    rows.append({"feature": f, "pearson": pearsonr(tr[f], tr.target)[0], "spearman": spearmanr(tr[f], tr.target)[0], "MI": m,
                 "val_pearson": pearsonr(v[f], v.y)[0], "val_spearman": spearmanr(v[f], v.y)[0]})
C = pd.DataFrame(rows).set_index("feature")
C["|S|-|P|"] = C.spearman.abs() - C.pearson.abs()
C["rank_P"] = C.pearson.abs().rank(ascending=False).astype(int); C["rank_MI"] = C.MI.rank(ascending=False).astype(int)
print("\n[C] train 기준(MI는 30,000행 표본) — |Spearman| 내림차순")
print(C.sort_values("spearman", key=abs, ascending=False).round(4).to_string())
C.to_csv(os.path.join(OUT, "feature_target_relations.csv"), float_format="%.4f")
