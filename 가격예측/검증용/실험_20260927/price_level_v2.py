# 종가(레벨) 예측 실험 — val만, test 미사용, 저장 없음. Transformer(운영과 동일 하이퍼파라미터, seed 42).
# target = 다음날 종가(원), 종목별 train 구간 평균/표준편차로 z-score 후 학습, 평가는 원 단위 역변환.
# A: 기존 15피처 그대로 / B: 15피처 + close_prev(오늘 종가, 같은 종목별 train 통계로 z-score)
import json, os, sys, time
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\kch_Final_prj"); os.chdir(r"C:\kch_Final_prj")
from datetime import date
import numpy as np, pandas as pd, torch
import 가격예측.가격예측_변동성_공통 as c
from 가격예측.garch_baseline import load_close_prices
from 가격예측.pooled_dataset import build_pooled_sequences, compute_global_split_dates
from 가격예측.sequence_dataset import FeatureScaler
from 가격예측.split_dataset import build_merged_dataset_v2, build_merged_dataset_v2_volatility_hybrid
from 가격예측.train_common import evaluate_pooled_predictions, train_pooled_transformer
from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START

tickers, start, end = ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START, date.today().isoformat()
ref, _ = build_merged_dataset_v2(tickers[0], start, end)
train_end, val_end = compute_global_split_dates(ref.index)
baselines = {t: c.compute_ticker_baselines(t, start, end, train_end, val_end) for t in tickers}

def build_fn(ticker, s, e, precomputed_indicators=None, true_kospi=None):
    b = baselines[ticker]
    merged, meta = build_merged_dataset_v2_volatility_hybrid(
        ticker, s, e, train_end, precomputed_sigma=b["sigma_full"], garch_params=b["params"],
        precomputed_indicators=precomputed_indicators, true_kospi=true_kospi)
    close = load_close_prices(ticker, s, e)
    merged = merged.copy()
    merged["close_prev"] = close.shift(1).reindex(merged.index)   # t-1 종가(피처 행과 같은 정보 시점)
    merged["target"] = close.reindex(merged.index)                # t 종가(raw 원)
    assert merged[["close_prev", "target"]].notna().all().all(), ticker
    cols = [x for x in merged.columns if x not in ("close_prev", "target")] + ["close_prev", "target"]
    return merged[cols], meta

splits, feature_cols, t2id, (te2, ve2) = build_pooled_sequences(tickers, start, end, lookback=c.LOOKBACK, build_fn=build_fn)
assert te2 == train_end and ve2 == val_end
Xtr, ytr, tidtr, _, _ = splits["train"]; Xva, yva, tidva, dva, _ = splits["val"]
ci = feature_cols.index("close_prev"); assert ci == len(feature_cols) - 1
print(f"split train_end={train_end.date()} val_end={val_end.date()} n_train={len(Xtr)} n_val={len(Xva)} feats={feature_cols}")

# 종목별 train 통계(target=종가, train 행만)
nt = len(tickers); mu = np.zeros(nt); sd = np.ones(nt)
for i in range(nt):
    m = tidtr == i; mu[i] = ytr[m].astype(np.float64).mean(); sd[i] = ytr[m].astype(np.float64).std()
def z(v, tid): return ((v - mu[tid]) / sd[tid]).astype(np.float32)
ytr_z, yva_z = z(ytr.astype(np.float64), tidtr), z(yva.astype(np.float64), tidva)
Xtr = Xtr.copy(); Xva = Xva.copy()
Xtr[:, :, ci] = (Xtr[:, :, ci] - mu[tidtr][:, None]) / sd[tidtr][:, None]
Xva[:, :, ci] = (Xva[:, :, ci] - mu[tidva][:, None]) / sd[tidva][:, None]
close_prev_va = np.round(Xva[:, -1, ci].astype(np.float64) * sd[tidva] + mu[tidva])   # 오늘 종가(원) 복원 — 원 단위 정수로 반올림(부동소수 오차로 보합일이 사라지는 것 방지)
actual_va = np.round(yva.astype(np.float64))
print("close_prev reconstruct max abs err(원):", float(np.abs(close_prev_va - np.array([0])).max()) if False else "skip")
print(f"val z 범위: target_z min={yva_z.min():.2f} max={yva_z.max():.2f} (train은 정의상 평균0/표준편차1)")

def metrics(name, pred_won):
    err = pred_won - actual_va
    rmse = float(np.sqrt(np.mean(err ** 2)))
    pct_rmse = float(np.sqrt(np.mean((err / actual_va * 100) ** 2)))
    zrmse = float(np.sqrt(np.mean((err / sd[tidva]) ** 2)))
    d_act = np.sign(actual_va - close_prev_va); d_pred = np.sign(pred_won - close_prev_va)
    nz = d_act != 0
    dir_acc = float((d_pred[nz] == d_act[nz]).mean())
    pred_up = float((d_pred[nz] > 0).mean())
    return dict(name=name, rmse_won=rmse, pct_err_rmse=pct_rmse, z_rmse=zrmse, dir_acc=dir_acc, pred_up_share=pred_up)

naive = metrics("naive(오늘 종가 복사)", close_prev_va)
d_act = np.sign(actual_va - close_prev_va); nz = d_act != 0
up_share = float((d_act[nz] > 0).mean())
print(f"val 실제 상승 비율(보합 제외)={up_share:.4f}, 보합 비율={float((~nz).mean()):.4f}")
res = [naive]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for variant, use_cols in (("A_15feats", list(range(ci))), ("B_15feats+close_prev", list(range(ci + 1)))):
    sc = FeatureScaler().fit(Xtr[:, :, use_cols])
    a = sc.transform(Xtr[:, :, use_cols]).astype(np.float32); b = sc.transform(Xva[:, :, use_cols]).astype(np.float32)
    t0 = time.time()
    r = train_pooled_transformer(a, ytr_z, tidtr, b, yva_z, tidva, len(use_cols), c.LOOKBACK, nt, c.EMBEDDING_DIM,
        c.D_MODEL, c.NHEAD, c.NUM_LAYERS, c.DIM_FEEDFORWARD, c.DROPOUT, c.BATCH_SIZE, c.LR, c.WEIGHT_DECAY,
        c.SMOKE_EPOCHS, c.MAX_EPOCHS, c.PATIENCE, device, seed=42, label=variant)
    pz, az, ptid = evaluate_pooled_predictions(r["model"], r["val_loader"], device)
    assert np.array_equal(ptid, tidva)
    mres = metrics(variant, pz.astype(np.float64) * sd[tidva] + mu[tidva]); mres["best_epoch"] = r["best_epoch"]; mres["sec"] = round(time.time() - t0)
    res.append(mres)
    np.savez(os.path.join(OUT, f"price_level_preds_{variant}.npz"), pred_won=pz.astype(np.float64) * sd[tidva] + mu[tidva], actual=actual_va, close_prev=close_prev_va, tid=tidva, dates=dva.astype("datetime64[D]").astype(str))
    print("RESULT", json.dumps(mres, ensure_ascii=False), flush=True)
print("\n=== SUMMARY ===")
for m in res:
    rel = (m["rmse_won"] / naive["rmse_won"] - 1) * 100
    print(f"{m['name']:<24} RMSE={m['rmse_won']:,.1f}원 (naive 대비 {rel:+.2f}%)  %오차RMSE={m['pct_err_rmse']:.3f}%  zRMSE={m['z_rmse']:.4f}  "
          f"방향정확도={m['dir_acc']:.4f}  예측상승비율={m['pred_up_share']:.3f}  best_ep={m.get('best_epoch','-')}")
print(f"(참고) 항상 '상승' 예측 시 방향정확도={up_share:.4f}, 항상 '하락'={1-up_share:.4f}")
