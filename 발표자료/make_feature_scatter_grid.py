# 발표자료/make_feature_scatter_grid.py — train 실측 타겟 vs 15피처 산점도 그리드(2026-09-27, EDA)
# hexbin(로그 색) + x 분위 구간별 타겟 중앙값 추세선(전체 행으로 계산). 표시 범위는 x 0.5~99.5%, y 0~99.5%로
# 자르되 잘린 비율을 각 칸에 명시. 제목의 Pearson/Spearman/MI는 diagnose_structure.py 결과(C) 재사용.
# 실행: python -m 발표자료.make_feature_scatter_grid   (train 행 캐시: 발표자료/_cache_train_rows.parquet)
import os
from datetime import date
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

INK, INK2, SURF = "#0b0b0b", "#52514e", "#ffffff"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 11, "figure.facecolor": SURF})
OUT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(OUT, "_cache_train_rows.parquet")
MACRO = {"KOSPI", "KOSDAQ", "USD_KRW", "ECOS_722Y001_0101000", "bond_spread_10y_3y", "ECOS_901Y009_0",
         "ECOS_161Y005_BBHS00", "ECOS_901Y067_I16E"}
LABEL = {"ECOS_722Y001_0101000": "기준금리", "ECOS_901Y009_0": "CPI", "ECOS_161Y005_BBHS00": "M2",
         "ECOS_901Y067_I16E": "선행지수 순환변동치", "bond_spread_10y_3y": "국고채 10y-3y 스프레드"}


def load_train():
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    import 가격예측.가격예측_변동성_공통 as c
    from 가격예측.momentum_feature import load_true_kospi
    from 가격예측.pooled_dataset import compute_global_split_dates
    from 가격예측.split_dataset import build_merged_dataset_v2, build_merged_dataset_v2_volatility_hybrid
    from 시장지표.feature_loader import load_indicator_cache
    from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
    start, end = STOCK_INITIAL_LOAD_START, date.today().isoformat()
    ref, _ = build_merged_dataset_v2(ACTIVE_TICKERS[0], start, end)
    te, ve = compute_global_split_dates(ref.index)
    ind, tk = load_indicator_cache(end), load_true_kospi(start, end)
    fr = []
    for t in ACTIVE_TICKERS:
        b = c.compute_ticker_baselines(t, start, end, te, ve)
        m, _ = build_merged_dataset_v2_volatility_hybrid(t, start, end, te, precomputed_sigma=b["sigma_full"],
            garch_params=b["params"], precomputed_indicators=ind, true_kospi=tk)
        m = m[m.index <= te].copy(); m["ticker"] = t; fr.append(m.rename_axis("date").reset_index())
    df = pd.concat(fr, ignore_index=True); df.to_parquet(CACHE); return df


def add_close(tr):
    """행 t에 전일 종가(close_(t-1), 원)를 붙인다 — 다른 피처와 같은 '행 t = t-1까지 정보' 규칙."""
    if "close_prev" in tr.columns:
        return tr
    from 가격예측.garch_baseline import load_close_prices
    from constants import STOCK_INITIAL_LOAD_START
    parts = []
    for t, g in tr.groupby("ticker"):
        cl = load_close_prices(t, STOCK_INITIAL_LOAD_START, str(pd.Timestamp(g.date.max()).date())).shift(1)
        parts.append(g.assign(close_prev=cl.reindex(pd.to_datetime(g.date)).values))
    out = pd.concat(parts).sort_index(); out.to_parquet(CACHE); return out


tr = add_close(load_train())
rel = pd.read_csv(os.path.join(OUT, "feature_target_relations.csv"), index_col=0)
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
_i = np.random.default_rng(0).choice(len(tr), 30000, replace=False)
rel.loc["close_prev"] = {"pearson": pearsonr(tr.close_prev, tr.target)[0], "spearman": spearmanr(tr.close_prev, tr.target)[0],
                         "MI": mutual_info_regression(np.log(tr.close_prev.values[_i])[:, None], tr.target.values[_i], random_state=0)[0]}
order = rel.spearman.abs().sort_values(ascending=False).index.tolist()
LABEL["close_prev"] = "전일 종가(원, 로그 x축)"
y = tr.target.values
ymax = float(np.percentile(y, 99.5))
fig, axes = plt.subplots(4, 4, figsize=(21, 19))
for ax, f in zip(axes.ravel(), order):
    x = tr[f].values
    logx = f == "close_prev"
    if logx: x = np.log10(x)
    lo, hi = np.percentile(x, [0.5, 99.5])
    if lo == hi: lo, hi = x.min(), x.max()
    show = (x >= lo) & (x <= hi) & (y <= ymax)
    hb = ax.hexbin(x[show], y[show], gridsize=45, cmap="Blues", norm=LogNorm(vmin=1), mincnt=1, linewidths=0)
    q = np.unique(np.quantile(x, np.linspace(0, 1, 21)))
    if len(q) >= 3:
        bi = np.clip(np.digitize(x, q[1:-1]), 0, len(q) - 2)
        g = pd.DataFrame({"b": bi, "x": x, "y": y}).groupby("b").agg(xm=("x", "median"), ym=("y", "median"), n=("y", "size"))
        g = g[(g.n >= 200) & (g.xm >= lo) & (g.xm <= hi)]
        ax.plot(g.xm, g.ym, color="#eb6834", lw=2.2, marker="o", ms=3.5)
    ax.set_xlim(lo, hi); ax.set_ylim(0, ymax)
    if logx:
        tk_ = [v for v in (1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6) if lo <= np.log10(v) <= hi]
        ax.set_xticks(np.log10(tk_)); ax.set_xticklabels([f"{v/1e4:g}만" if v >= 1e4 else f"{v:,.0f}" for v in tk_])
    r = rel.loc[f]
    ax.set_title(f"{LABEL.get(f, f)}\nP={r.pearson:+.3f}  S={r.spearman:+.3f}  MI={r.MI:.3f}", fontsize=11.5, color=INK, loc="left")
    hidden = 1 - show.mean()
    note = f"표시 범위 밖 {hidden:.1%}"
    if f == "close_prev":
        note = f"MI 과대: 가격 레벨이 사실상 '종목·시점 식별자'라\nkNN MI가 변동성 군집을 잡은 것(가격 효과 아님)\n{note}"
    if f in MACRO:
        note = f"거시지표: 같은 날 100종목 공유값 → 수직 띠\n{note}"
        ax.set_facecolor("#f7f6f3")
    ax.text(0.98, 0.97, note, transform=ax.transAxes, ha="right", va="top", fontsize=9.3, color=INK2,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))
    ax.tick_params(labelsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
for ax in axes[:, 0]: ax.set_ylabel("실측 타겟 |로그수익률|×100")
cb = fig.colorbar(hb, ax=axes, shrink=0.6, pad=0.01); cb.set_label("칸당 행 수(로그 색)")
fig.suptitle("train 실측 타겟 vs 15피처 — hexbin 밀도 + 주황선 = x 분위 20구간별 타겟 중앙값(전체 행 기준) · 전일 종가 패널 추가",
             x=0.01, ha="left", fontsize=17, color=INK)
fig.text(0.01, 0.01, f"train ~2024-10-10 · 100종목 · n={len(tr):,} · |Spearman| 순 정렬 · P/S/MI는 train 전체(MI는 3만 행 표본) · "
         f"표시: x 0.5~99.5%, y 0~{ymax:.1f}(99.5%) — 추세선·지표는 자르기 전 전체로 계산 · 회색 배경 = 거시지표", fontsize=10.5, color=INK2)
p = os.path.join(OUT, "fig6_feature_scatter_grid.png"); fig.savefig(p, dpi=150, bbox_inches="tight"); print(p, len(tr))
