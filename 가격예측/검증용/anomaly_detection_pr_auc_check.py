# anomaly_detection_pr_auc_check.py — 이상탐지 전환 검토 1~2단계 (2026-09-20)
#
# 결과_TaskT_급변구간_손실함수_시도.md에서 "구조적 한계"로 결론지은 뒤, 회귀(레벨 맞히기)
# 대신 이진분류(급변일 탐지)로 문제를 재정의할 수 있는지 확인한다. 3단계(새 BCE 모델 학습)는
# 사람 승인 후 별도 진행 — 이 스크립트는 "기존 예측값을 탐지기로 재활용할 수 있는가"만 본다.
#
# 라벨: |수익률| >= 2x 직전60일평균("급변일", highvol_regime_diagnosis.py와 동일 정의).
# 점수(스코어)로 재활용: baseline(15피처 MSE)/tau=0.8 pinball의 predicted_volatility,
# GARCH sigma 단독 — 셋 다 "값이 클수록 급변일일 것 같다"는 연속 점수로 취급해 PR-AUC 계산.
#
# ⚠️ baseline/tau=0.8 pinball 예측값은 이전 실행에서 디스크에 저장하지 않아 사라졌으므로,
# 동일 시드/데이터로 두 모델을 재현(신규 실험 아님 — pinball_loss_sweep.py 결과와 일치하는지
# 재현 확인 포함)한다. test는 이번에도 전혀 사용하지 않는다.

import sys
from datetime import date

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve

from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
from 가격예측.가격예측_변동성_공통 import (
    BATCH_SIZE, D_MODEL, DIM_FEEDFORWARD, DROPOUT, EMBEDDING_DIM, LOOKBACK, LR,
    MAX_EPOCHS, NHEAD, NUM_LAYERS, PATIENCE, SEED, SMOKE_EPOCHS, WEIGHT_DECAY,
)
from 가격예측.garch_baseline import compute_log_returns_pct, load_close_prices, rmse_mae
from 가격예측.검증용.garch_baseline_check import compute_full_period_sigma
from 가격예측.검증용.highvol_regime_diagnosis import (
    BIG_MOVE_MULTIPLIER, BIG_MOVE_WINDOW, compute_big_move_labels,
)
from 가격예측.pooled_dataset import build_pooled_sequences, compute_global_split_dates
from 가격예측.sequence_dataset import FeatureScaler
from 가격예측.split_dataset import build_merged_dataset_v2, build_merged_dataset_v2_volatility_hybrid
from 가격예측.train_common import (
    evaluate_pooled_predictions, make_pooled_loader, train_pooled_transformer,
    train_pooled_transformer_pinball,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PINBALL_TAU = 0.8
THRESH_PERCENTILES = [70, 80, 90]  # 상위 30%/20%/10%를 "양성 예측"으로 플래그


def precompute_garch(tickers, start_date, end_date, train_end):
    cache = {}
    for i, ticker in enumerate(tickers, 1):
        sigma, params = compute_full_period_sigma(ticker, start_date, end_date, train_end)
        cache[ticker] = (sigma, params)
        if i % 20 == 0 or i == len(tickers):
            print(f"  GARCH sigma 사전계산 {i}/{len(tickers)}", flush=True)
    return cache


def make_build_fn(train_end, garch_cache):
    def build_fn(ticker, s, e, precomputed_indicators=None, true_kospi=None):
        sigma, params = garch_cache[ticker]
        return build_merged_dataset_v2_volatility_hybrid(
            ticker, s, e, train_end, precomputed_sigma=sigma, garch_params=params,
            precomputed_indicators=precomputed_indicators, true_kospi=true_kospi,
        )
    return build_fn


def report_pr(name, scores, labels, valid_mask, prevalence):
    v = valid_mask & ~np.isnan(scores)
    ap = average_precision_score(labels[v], scores[v])
    print(f"\n[{name}] PR-AUC(average precision)={ap:.4f}  (무작위/유병률 기준선={prevalence:.4f}, "
          f"배율={ap/prevalence:.2f}x)  n={int(v.sum())}")
    for pct in THRESH_PERCENTILES:
        thr = np.percentile(scores[v], pct)
        pred_pos = scores[v] >= thr
        y = labels[v]
        tp = int((pred_pos & y).sum())
        fp = int((pred_pos & ~y).sum())
        fn = int((~pred_pos & y).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        print(f"    상위{100-pct}%를 양성으로 플래그(threshold={thr:.4f}): "
              f"precision={precision:.3f}  recall={recall:.3f}")
    return ap


def main():
    tickers = ACTIVE_TICKERS
    end_date = date.today().isoformat()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ref, _ = build_merged_dataset_v2(tickers[0], STOCK_INITIAL_LOAD_START, end_date)
    train_end, val_end = compute_global_split_dates(ref.index)
    print(f"fold3 분할 경계: train_end={train_end.date()}  val_end={val_end.date()}")
    assert train_end.date().isoformat() == "2024-10-07", "fold3 경계가 이전 실행과 달라짐"

    print(f"\nGARCH sigma 사전계산({len(tickers)}종목)...")
    garch_cache = precompute_garch(tickers, STOCK_INITIAL_LOAD_START, end_date, train_end)

    build_fn = make_build_fn(train_end, garch_cache)
    splits, feature_cols, ticker_to_id, _ = build_pooled_sequences(
        tickers, STOCK_INITIAL_LOAD_START, end_date, lookback=LOOKBACK, build_fn=build_fn,
    )
    X_train, y_train, tid_train, d_train, _ = splits["train"]
    X_val, y_val, tid_val, d_val, _ = splits["val"]
    # ⚠️ splits["test"]는 여기서 끝 — 어떤 변수에도 담지 않는다.

    scaler = FeatureScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    print("\n급변일 라벨 + GARCH baseline을 val 표본에 정렬 중...")
    n = len(tid_val)
    big_move_val = np.zeros(n, dtype=bool)
    garch_aligned = np.full(n, np.nan)
    for ticker, tid in ticker_to_id.items():
        mask = tid_val == tid
        if mask.sum() == 0:
            continue
        close = load_close_prices(ticker, STOCK_INITIAL_LOAD_START, end_date)
        returns = compute_log_returns_pct(close)
        bm_full = compute_big_move_labels(returns, window=BIG_MOVE_WINDOW, multiplier=BIG_MOVE_MULTIPLIER)
        sigma_full, _ = garch_cache[ticker]
        dates = pd.DatetimeIndex(d_val[mask])
        idx = np.where(mask)[0]
        big_move_val[idx] = bm_full.reindex(dates).fillna(False).values
        garch_aligned[idx] = sigma_full.reindex(dates).values

    prevalence = float(big_move_val.mean())
    print(f"급변일 비율(val 전체, 무작위/유병률 기준선): {prevalence:.4f} ({int(big_move_val.sum())}/{n})")

    val_loader = make_pooled_loader(X_val_s, y_val, tid_val, BATCH_SIZE, shuffle=False)

    print("\n=== baseline(15피처, MSE) 재현 ===")
    result = train_pooled_transformer(
        X_train_s, y_train, tid_train, X_val_s, y_val, tid_val,
        len(feature_cols), LOOKBACK, len(tickers), EMBEDDING_DIM,
        D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT,
        BATCH_SIZE, LR, WEIGHT_DECAY, SMOKE_EPOCHS, MAX_EPOCHS, PATIENCE,
        device, seed=SEED, verbose=True, label="baseline-repro",
    )
    preds_baseline, actuals, _ = evaluate_pooled_predictions(result["model"], val_loader, device)
    base_rmse, _ = rmse_mae(preds_baseline, actuals)
    print(f"재현 확인(전체 val RMSE): {base_rmse:.4f} (이전 실행 참고치: 이 스크립트에서 baseline "
          "①②③④ 분해는 안 했으므로 전체 RMSE로만 재현성 감 잡기)")

    print("\n=== tau=0.8 pinball 재현 ===")
    result_pb = train_pooled_transformer_pinball(
        X_train_s, y_train, tid_train, X_val_s, y_val, tid_val, PINBALL_TAU,
        len(feature_cols), LOOKBACK, len(tickers), EMBEDDING_DIM,
        D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT,
        BATCH_SIZE, LR, WEIGHT_DECAY, SMOKE_EPOCHS, MAX_EPOCHS, PATIENCE,
        device, seed=SEED, verbose=True, label="pinball08-repro",
    )
    preds_pinball, _, _ = evaluate_pooled_predictions(result_pb["model"], val_loader, device)

    print("\n" + "=" * 100)
    print("=== PR-AUC 비교 (label=급변일, score=predicted_volatility 또는 GARCH sigma) ===")
    print("=" * 100)
    all_valid = np.ones(n, dtype=bool)
    ap_base = report_pr("baseline(15피처, MSE)", preds_baseline, big_move_val, all_valid, prevalence)
    ap_pinball = report_pr(f"tau={PINBALL_TAU} pinball", preds_pinball, big_move_val, all_valid, prevalence)
    ap_garch = report_pr("GARCH sigma 단독", garch_aligned, big_move_val, all_valid, prevalence)

    print("\n" + "=" * 100)
    print("=== 판정 ===")
    print("=" * 100)
    results = {"baseline": ap_base, f"pinball tau={PINBALL_TAU}": ap_pinball, "GARCH 단독": ap_garch}
    for name, ap in results.items():
        strong = ap / prevalence >= 2.0
        print(f"{name}: PR-AUC={ap:.4f}  기준선 대비 배율={ap/prevalence:.2f}x  "
              f"확실히 우수(>=2x)={strong}")

    if any(ap / prevalence >= 2.0 for ap in results.values()):
        best_name = max(results, key=results.get)
        print(f"\n결론: {best_name}(PR-AUC={results[best_name]:.4f})가 무작위 대비 확실히 "
              "우수 — 이 모델의 예측값을 탐지용으로 재활용하는 쪽을 제안. 3단계(새 BCE 모델 "
              "학습)는 사람 승인 후 진행.")
    else:
        print("\n결론: 셋 다 무작위 대비 확실한 우위를 보이지 못함 — 이진분류(BCE) 모델을 "
              "새로 학습하는 방향을 다음 지시문으로 요청할 것을 제안.")

    print("\n(주: test(2025-09-26~오늘)는 이번 실행에서 전혀 평가하지 않았다.)")


if __name__ == "__main__":
    main()
