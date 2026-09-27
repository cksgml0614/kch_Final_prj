# 2026-09-27 세션에서 인라인으로만 실행했던 실험들을 regime_eval로 재구성(결과 수치 재현 확인 완료):
#   1) 가격 추정 비교(naive / 종가모델 B / 변동성+오라클 방향)  -> 발표자료/price_estimate_comparison.csv
#   2) 1)의 ①급변일/③평상일/④지속중 그룹별 재집계              -> 발표자료/price_estimate_by_regime.csv
#   3) garch_sigma 임계값 GARCH<->LSTM 블렌딩(train 70/80/90분위)  -> 결과 문서 [10] 시도 1
#   4) GBM 급변 확률 스위치의 ①/③ 전환율                            -> 결과 문서 [10] 시도 2
# 입력(--inputs 폴더, 세션 산출물): val_diag_rows.parquet(val 행: ticker,date,y,garch,sma20,parkinson,spike),
#   lossrecheck_lstm_s42.npy(운영 LSTM val 예측, 같은 행 순서), price_level_preds_B_15feats+close_prev.npz,
#   bigmove_clf_val_pred.npy / bigmove_clf_train_pred.npy, 그리고 발표자료/_cache_train_rows.parquet.
# 실행: python -m 가격예측.검증용.실험_20260927.price_estimate_and_blend --inputs <폴더>
import argparse, os
from datetime import date
import numpy as np, pandas as pd
from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
from 가격예측.검증용.regime_eval import (bigmove_labels, oracle_price_estimate, per_ticker_rmse, price_errors, regime_masks)

ap = argparse.ArgumentParser(); ap.add_argument("--inputs", required=True)
ap.add_argument("--train-cache", default=os.path.join("발표자료", "_cache_train_rows.parquet")); a = ap.parse_args()
I = a.inputs
v = pd.read_parquet(os.path.join(I, "val_diag_rows.parquet"))[["ticker", "date", "y", "garch", "sma20", "parkinson"]].copy()
v["lstm"] = np.load(os.path.join(I, "lossrecheck_lstm_s42.npy")); v["gbm_p"] = np.load(os.path.join(I, "bigmove_clf_val_pred.npy"))
b = np.load(os.path.join(I, "price_level_preds_B_15feats+close_prev.npz"), allow_pickle=True)
B = pd.DataFrame({"ticker": np.array(ACTIVE_TICKERS)[b["tid"]], "date": pd.to_datetime(b["dates"]),
                  "B_pred": b["pred_won"], "actual": b["actual"], "close_prev": b["close_prev"]})
d = v.merge(B, on=["ticker", "date"], validate="one_to_one")
d = d.merge(bigmove_labels(ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START, date.today().isoformat()), on=["ticker", "date"], how="left")
d[["bigmove", "firstday"]] = d[["bigmove", "firstday"]].fillna(False).astype(bool)
tr = pd.read_parquet(a.train_cache)
pd.set_option("display.width", 220)

# 1)·2) 가격 추정 — 전체 23,800행
M = regime_masks(d.y, d.bigmove, d.firstday)
est = {"naive": d.close_prev.values, "종가모델 B": d.B_pred.values,
       "상수크기+오라클": oracle_price_estimate(d.close_prev, d.actual, np.full(len(d), tr.target.mean()))}
for m in ("lstm", "garch", "sma20", "parkinson"):
    est[f"{m}+오라클"] = oracle_price_estimate(d.close_prev, d.actual, d[m])
rows = [{"그룹": g, "방법": k, "n": int(msk.sum()), **dict(zip(("RMSE(원)", "%오차 RMSE"), price_errors(e, d.actual, msk)))}
        for g, msk in M.items() for k, e in est.items()]
P = pd.DataFrame(rows); P.to_csv(os.path.join("발표자료", "price_estimate_rebuilt.csv"), index=False, float_format="%.4f")
print("[1·2] 가격 추정 (발표자료/price_estimate_rebuilt.csv)"); print(P.round(4).to_string(index=False))

# 3) garch_sigma 블렌딩 — baseline 유효 23,737행(게이트 재검증과 동일)
dv = d.dropna(subset=["garch", "sma20", "parkinson"]).reset_index(drop=True)
Mv = regime_masks(dv.y, dv.bigmove, dv.firstday)
print("\n[3] garch_sigma 블렌딩(종목평균 변동성 RMSE / ①·③ 오라클 가격 %오차 / ①·③ 전환율)")
for q in (70, 80, 90):
    th = tr.garch_sigma.quantile(q / 100)
    blend = np.where(dv.garch > th, dv.garch, dv.lstm)
    e = oracle_price_estimate(dv.close_prev, dv.actual, blend)
    print(f"  {q}분위 th={th:.3f}: 게이트RMSE={per_ticker_rmse(blend, dv.y, dv.ticker):.4f} | "
          f"① {price_errors(e, dv.actual, Mv['①급변일'])[1]:.4f}% ③ {price_errors(e, dv.actual, Mv['③평상일'])[1]:.4f}% | "
          f"전환 ① {(dv.garch > th)[Mv['①급변일']].mean():.1%} ③ {(dv.garch > th)[Mv['③평상일']].mean():.1%}")

# 4) GBM 급변 확률 스위치 전환율 — 전체 23,800행, 임계값은 train 확률 분위수
ptr = np.load(os.path.join(I, "bigmove_clf_train_pred.npy"))
print("\n[4] GBM 확률 스위치 전환율")
for q in (70, 80, 90):
    th = np.quantile(ptr, q / 100); on = d.gbm_p.values > th
    s1, s3 = on[M["①급변일"]].mean(), on[M["③평상일"]].mean()
    print(f"  {q}분위 th={th:.4f}: ① {s1:.1%} vs ③ {s3:.1%} ({s1 / s3:.2f}x) | 전체 {on.mean():.1%}")
