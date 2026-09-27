# val 윈도우 진단(읽기 전용, test 미사용, 저장 없음). 운영과 동일 데이터 준비 + Transformer seed 42 재학습.
import os, sys, time
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\kch_Final_prj"); os.chdir(r"C:\kch_Final_prj")
from datetime import date
import numpy as np, pandas as pd, torch
from scipy.stats import skew
import 가격예측.가격예측_변동성_공통 as c
from 가격예측.garch_baseline import compute_log_returns_pct, compute_sma_baseline, load_close_prices, rmse_mae
from 가격예측.parkinson_baseline import compute_parkinson_vol_pct, load_high_low, PARK_WINDOW
from 가격예측.pooled_dataset import build_pooled_sequences, compute_global_split_dates
from 가격예측.sequence_dataset import FeatureScaler
from 가격예측.split_dataset import build_merged_dataset_v2, build_merged_dataset_v2_volatility_hybrid
from 가격예측.train_common import evaluate_pooled_predictions, train_pooled_transformer
from db_manager import get_db_connection
from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
pd.set_option("display.width", 200)

tickers, start, end = ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START, date.today().isoformat()
ref, _ = build_merged_dataset_v2(tickers[0], start, end)
train_end, val_end = compute_global_split_dates(ref.index)
B = {t: c.compute_ticker_baselines(t, start, end, train_end, val_end) for t in tickers}

# ── 종목별 baseline/실현값/급변일 행 단위 테이블 ──
rows = []
for t in tickers:
    close = load_close_prices(t, start, end); ret = compute_log_returns_pct(close)
    hl = load_high_low(t, start, end)
    park = compute_parkinson_vol_pct(hl["high"], hl["low"]).rolling(PARK_WINDOW, min_periods=PARK_WINDOW).mean().shift(1)
    a = ret.abs()
    df = pd.DataFrame({"realized": a, "garch": B[t]["sigma_full"].reindex(a.index), "sma20": compute_sma_baseline(ret),
                       "parkinson": park.reindex(a.index), "spike": a >= 2 * a.shift(1).rolling(60, min_periods=60).mean(),
                       "close": close.reindex(a.index)})
    df["ticker"] = t; rows.append(df)
base = pd.concat(rows); base.index.name = "date"; base = base.reset_index()
base["split"] = np.where(base.date <= train_end, "train", np.where(base.date <= val_end, "val", "test"))

# ── 운영과 동일 학습(재현 확인) ──
def hybrid_build_fn(ticker, s, e, precomputed_indicators=None, true_kospi=None):
    b = B[ticker]
    return build_merged_dataset_v2_volatility_hybrid(ticker, s, e, train_end, precomputed_sigma=b["sigma_full"],
        garch_params=b["params"], precomputed_indicators=precomputed_indicators, true_kospi=true_kospi)
splits, fc, t2id, _ = build_pooled_sequences(tickers, start, end, lookback=c.LOOKBACK, build_fn=hybrid_build_fn)
Xtr, ytr, tidtr, dtr, _ = splits["train"]; Xva, yva, tidva, dva, tnva = splits["val"]
sc = FeatureScaler().fit(Xtr)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
r = train_pooled_transformer(sc.transform(Xtr).astype(np.float32), ytr, tidtr, sc.transform(Xva).astype(np.float32), yva, tidva,
    len(fc), c.LOOKBACK, len(tickers), c.EMBEDDING_DIM, c.D_MODEL, c.NHEAD, c.NUM_LAYERS, c.DIM_FEEDFORWARD, c.DROPOUT,
    c.BATCH_SIZE, c.LR, c.WEIGHT_DECAY, c.SMOKE_EPOCHS, c.MAX_EPOCHS, c.PATIENCE, dev, seed=42, verbose=False)
p, a, _ = evaluate_pooled_predictions(r["model"], r["val_loader"], dev)
hy = pd.DataFrame({"ticker": tnva, "date": pd.to_datetime(dva), "hybrid": p, "y": a})
v = hy.merge(base[base.split == "val"], on=["ticker", "date"], how="left")
print(f"val rows hybrid={len(hy)}  merged baseline 결측={int(v.garch.isna().sum())}  |y-realized| max={float((v.y - v.realized).abs().max()):.2e}")
v.to_parquet(os.path.join(OUT, "val_diag_rows.parquet"))

M = ["hybrid", "garch", "sma20", "parkinson"]
def rm(x, col): return float(np.sqrt(np.mean((x[col] - x.realized) ** 2)))
def ma(x, col): return float(np.mean(np.abs(x[col] - x.realized)))
def table(x, label):
    x = x.dropna(subset=M)
    pooled = {m: rm(x, m) for m in M}
    per_t = {m: float(np.mean([rm(g, m) for _, g in x.groupby("ticker")])) for m in M}
    mae = {m: ma(x, m) for m in M}
    print(f"\n[{label}] n={len(x)} tickers={x.ticker.nunique()}")
    for name, d in (("pooled RMSE", pooled), ("종목평균 RMSE", per_t), ("MAE", mae)):
        verdict = " ".join(f"vs{m}={'이김' if d['hybrid'] < d[m] else '짐'}" for m in M[1:])
        print(f"  {name:<12} " + "  ".join(f"{m}={d[m]:.4f}" for m in M) + f"   | {verdict}")
    return pooled, per_t

print("\n===== B1. 게이트 집계 방식 =====")
g = {m: float(np.mean([B[t]["val_rmse"][m if m != 'parkinson' else 'parkinson'] for t in tickers])) for m in ("garch", "sma20", "parkinson")}
print(f"[운영 방식 재현] hybrid pooled RMSE={rmse_mae(p, a)[0]:.4f} vs baseline 종목평균 RMSE: " + "  ".join(f"{k}={x:.4f}" for k, x in g.items()))
table(v, "같은 행 기준 — 전체 val")

print("\n===== B1-2. 19영업일 백테스트 행(DB, N=1,800)을 RMSE로 다시 =====")
with get_db_connection() as conn:
    bt = pd.read_sql("""SELECT ticker, target_date AS date, predicted_volatility AS hybrid, garch_baseline AS garch,
        sma20_baseline AS sma20, parkinson_sma20_baseline AS parkinson, actual_volatility AS realized
        FROM model_predictions WHERE prediction_date BETWEEN '2026-08-14' AND '2026-09-11' AND actual_volatility IS NOT NULL""", conn)
for col in M + ["realized"]: bt[col] = bt[col].astype(float)
table(bt, "백테스트 2026-08~09")

print("\n===== B2. target(|로그수익률x100|) 분포: train vs val =====")
for s in ("train", "val"):
    x = base[(base.split == s) & base.garch.notna()]
    sp = x.spike.dropna()
    print(f"  {s:<5} n={len(x):>6} mean={x.realized.mean():.4f} std={x.realized.std():.4f} skew={skew(x.realized):.3f} "
          f"p95={x.realized.quantile(.95):.3f} max={x.realized.max():.2f}  급변일(>=2x 직전60일평균) 비율={sp.mean():.4f}")
x = bt; print(f"  (참고) 백테스트 행 n={len(x)} mean={x.realized.mean():.4f} std={x.realized.std():.4f} skew={skew(x.realized):.3f}")

print("\n===== B3. 소수 종목 영향 =====")
vv = v.dropna(subset=M).copy()
for m in M: vv[f"se_{m}"] = (vv[m] - vv.realized) ** 2
contrib = vv.groupby("ticker")[[f"se_{m}" for m in M]].sum()
contrib["share_hybrid_SSE"] = contrib.se_hybrid / contrib.se_hybrid.sum()
contrib["gap_vs_best_baseline"] = contrib.se_hybrid - contrib[["se_garch", "se_sma20", "se_parkinson"]].min(axis=1)
top = contrib.sort_values("share_hybrid_SSE", ascending=False)
print("hybrid 제곱오차 합 상위 10종목 비중:"); print(top[["share_hybrid_SSE", "gap_vs_best_baseline"]].head(10).round(4))
print(f"상위 1/3/5/10종목 누적 비중: " + ", ".join(f"{k}={top.share_hybrid_SSE.head(k).sum():.3f}" for k in (1, 3, 5, 10)))
for k in (1, 3, 5, 10):
    table(vv[~vv.ticker.isin(top.index[:k])], f"hybrid SSE 상위 {k}종목 제외")
# 가격 레벨 실험의 '38σ 이탈' 종목: val 종가가 train 종가 평균에서 몇 σ 떨어졌는지
z = []
for t, g0 in base.groupby("ticker"):
    tr = g0[g0.split == "train"].close; va = g0[g0.split == "val"].close
    z.append((t, float(((va - tr.mean()) / tr.std()).abs().max())))
z = pd.Series(dict(z)).sort_values(ascending=False)
print("\nval 종가의 train 대비 최대 |z| 상위 10:", z.head(10).round(1).to_dict())
for thr in (10, 5):
    ex = z[z > thr].index
    table(vv[~vv.ticker.isin(ex)], f"가격 |z|>{thr} 종목 {len(ex)}개 제외")
