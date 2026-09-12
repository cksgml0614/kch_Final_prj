# test_evaluation_pooled50_hybrid.py — 50종목(시가총액 상위, 2026-09-06 확장) pooled
# 하이브리드(GARCH σ 포함 15피처) Transformer 재학습·평가 (일회성).
#
# 8종목 버전(test_evaluation_pooled_volatility_hybrid.py)과의 차이:
#   - ACTIVE_TICKERS가 이미 50개로 확장됨(constants.py) — 별도 종목 리스트 지정 불필요
#   - 종목 임베딩 차원을 8 -> 16으로 확대(아래 [임베딩 차원 재검토] 참고)
#   - 8종목 시절의 PURE_TF_RESULTS(순수 Transformer) 하드코딩 비교는 50종목에 맞지 않아 생략
#     (요청 범위에 없음 — GARCH/SMA20/Parkinson-SMA20 3종 baseline 비교만 수행)
#   - 예측 분포(mean/std/min/max)를 반드시 출력 — "분산 붕괴가 완화됐는지"가 이번 실험의 핵심 질문

import sys
from datetime import date

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
from 가격예측.garch_baseline import (
    compute_log_returns_pct, compute_sma_baseline,
    load_close_prices, rmse_mae,
)
from 가격예측.검증용.garch_baseline_check import fit_and_forecast_garch
from 가격예측.parkinson_baseline import compute_parkinson_vol_pct, load_high_low, PARK_WINDOW
from 가격예측.pooled_dataset import build_pooled_sequences, compute_global_split_dates
from 가격예측.sequence_dataset import FeatureScaler
from 가격예측.split_dataset import build_merged_dataset_v2, build_merged_dataset_v2_volatility_hybrid
from 가격예측.train_common import evaluate_pooled_predictions, train_pooled_transformer
from 가격예측.검증용.가격예측_통합모델 import (
    BATCH_SIZE, DIM_FEEDFORWARD, DROPOUT, D_MODEL, LOOKBACK, LR,
    MAX_EPOCHS, NHEAD, NUM_LAYERS, PATIENCE, SEED, SMOKE_EPOCHS, WEIGHT_DECAY,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

N_SHUFFLE = 7
SHUFFLE_SEEDS = [0, 1, 7, 123, 2024, 777, 999]

# [임베딩 차원 재검토] 8종목 때는 EMBEDDING_DIM=8(1:1 비율). 50종목으로 늘었으니 그 비율 그대로면
# 지나치게 커져(50) 15개 실제 피처를 종목 정체성 정보가 압도한다. d_model=32 대비 embedding이
# 절반을 넘지 않도록(15피처+임베딩 concat 후 input_proj로 32차원 투영) 16으로 설정 —
# 8종목 대비 2배로, "더 많은 종목을 구분할 여지"와 "실제 피처 비중 유지"의 절충.
EMBEDDING_DIM_50 = 16


def eval_loader(model, X, y, tid, device):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(tid)),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    return evaluate_pooled_predictions(model, loader, device)


def describe(preds):
    return {"mean": float(np.mean(preds)), "std": float(np.std(preds)),
            "min": float(np.min(preds)), "max": float(np.max(preds))}


def main():
    tickers = ACTIVE_TICKERS
    end_date = date.today().isoformat()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"=== Pooled 하이브리드(GARCH+Transformer) 변동성 예측 평가 — {len(tickers)}종목 ===")
    print(f"EMBEDDING_DIM = {EMBEDDING_DIM_50} (8종목 시절 8에서 확대)")

    ref_vol, _ = build_merged_dataset_v2(tickers[0], STOCK_INITIAL_LOAD_START, end_date)
    train_end, val_end = compute_global_split_dates(ref_vol.index)
    print(f"전역 분할 경계: train_end={train_end.date()}  val_end={val_end.date()}")

    def hybrid_build_fn(ticker, s, e):
        return build_merged_dataset_v2_volatility_hybrid(ticker, s, e, train_end)

    splits, feature_cols, ticker_to_id, (train_end2, val_end2) = build_pooled_sequences(
        tickers, STOCK_INITIAL_LOAD_START, end_date, lookback=LOOKBACK, build_fn=hybrid_build_fn,
    )
    assert train_end2 == train_end and val_end2 == val_end
    X_train, y_train, tid_train, d_train, _ = splits["train"]
    X_val, y_val, tid_val, d_val, _ = splits["val"]
    X_test, y_test, tid_test, d_test, _ = splits["test"]
    print(f"feature_cols({len(feature_cols)}개): {feature_cols}")
    print(f"시퀀스 shape: X_train {X_train.shape}  X_val {X_val.shape}  X_test {X_test.shape}")

    scaler = FeatureScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    def train_once(y_train_arr, label, verbose):
        return train_pooled_transformer(
            X_train_s, y_train_arr, tid_train, X_val_s, y_val, tid_val,
            len(feature_cols), LOOKBACK, len(tickers), EMBEDDING_DIM_50,
            D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT,
            BATCH_SIZE, LR, WEIGHT_DECAY, SMOKE_EPOCHS, MAX_EPOCHS, PATIENCE,
            device, seed=SEED, verbose=verbose, label=label,
        )

    print("\n" + "=" * 100)
    print("=== 1) 실제 하이브리드 pooled Transformer(50종목) 학습 + test 평가 ===")
    print("=" * 100)
    result = train_once(y_train, "hybrid50-real", verbose=True)
    real_preds, real_actuals, real_tids = eval_loader(result["model"], X_test_s, y_test, tid_test, device)

    print("\n" + "=" * 100)
    print("=== [핵심] test 예측 분포 — 8종목 시절(mean=1.8383, std=0.2608)과 비교 ===")
    print("=" * 100)
    dist = describe(real_preds)
    print(f"50종목 test 예측 분포: {dist}")
    print(f"실제 test target 분포: mean={y_test.mean():.4f}  std={y_test.std():.4f}  "
          f"min={y_test.min():.4f}  max={y_test.max():.4f}")
    print(f"예측 표준편차가 실제 target 표준편차의 {dist['std']/y_test.std()*100:.1f}% "
          f"(8종목 시절 9.5%였음 — 이 비율이 올라갔으면 분산 붕괴 완화)")

    print("\n" + "=" * 100)
    print("=== 2) baseline 3종 재계산 (GARCH/SMA20/Parkinson-SMA20, 50종목) ===")
    print("=" * 100)
    garch_rows, sma_rows, park_rows = {}, {}, {}
    for ticker in tickers:
        close = load_close_prices(ticker, STOCK_INITIAL_LOAD_START, end_date)
        returns = compute_log_returns_pct(close)
        test_returns = returns[returns.index > val_end]
        sigma_pred, _ = fit_and_forecast_garch(returns, train_end, test_returns.index[0])
        sma_pred = compute_sma_baseline(returns).loc[sigma_pred.index]
        hl = load_high_low(ticker, STOCK_INITIAL_LOAD_START, end_date)
        park_raw = compute_parkinson_vol_pct(hl["high"], hl["low"])
        park_ma20 = park_raw.rolling(PARK_WINDOW, min_periods=PARK_WINDOW).mean().shift(1).loc[sigma_pred.index].dropna()

        realized = test_returns.abs().loc[sigma_pred.index]
        garch_rows[ticker] = rmse_mae(sigma_pred.values, realized.values)
        sma_rows[ticker] = rmse_mae(sma_pred.values, realized.values)
        park_rows[ticker] = rmse_mae(park_ma20.values, realized.loc[park_ma20.index].values)

    avg_g_rmse = np.mean([v[0] for v in garch_rows.values()]); avg_g_mae = np.mean([v[1] for v in garch_rows.values()])
    avg_s_rmse = np.mean([v[0] for v in sma_rows.values()]); avg_s_mae = np.mean([v[1] for v in sma_rows.values()])
    avg_p_rmse = np.mean([v[0] for v in park_rows.values()]); avg_p_mae = np.mean([v[1] for v in park_rows.values()])
    print(f"GARCH(1,1)      : RMSE={avg_g_rmse:.4f}  MAE={avg_g_mae:.4f}")
    print(f"SMA20           : RMSE={avg_s_rmse:.4f}  MAE={avg_s_mae:.4f}")
    print(f"Parkinson-SMA20 : RMSE={avg_p_rmse:.4f}  MAE={avg_p_mae:.4f}")

    def per_ticker_rmse_mae(preds, actuals, tids):
        out = {}
        for ticker, tid in ticker_to_id.items():
            mask = tids == tid
            out[ticker] = rmse_mae(preds[mask], actuals[mask])
        return out, rmse_mae(preds, actuals)

    hybrid_per_ticker, hybrid_overall = per_ticker_rmse_mae(real_preds, real_actuals, real_tids)
    avg_h_rmse = float(np.mean([hybrid_per_ticker[t][0] for t in tickers]))
    avg_h_mae = float(np.mean([hybrid_per_ticker[t][1] for t in tickers]))

    print("\n" + "=" * 100)
    print("=== 3) 종합 비교 (50종목 평균) ===")
    print("=" * 100)
    print(f"{'구분':<20}{'RMSE':>10}{'MAE':>10}")
    print(f"{'하이브리드(50종목)':<20}{avg_h_rmse:>10.4f}{avg_h_mae:>10.4f}")
    print(f"{'GARCH(1,1)':<20}{avg_g_rmse:>10.4f}{avg_g_mae:>10.4f}")
    print(f"{'SMA20':<20}{avg_s_rmse:>10.4f}{avg_s_mae:>10.4f}")
    print(f"{'Parkinson-SMA20':<20}{avg_p_rmse:>10.4f}{avg_p_mae:>10.4f}")
    print(f"(참고: 전체 test 시퀀스 pooling 기준 RMSE/MAE = {hybrid_overall[0]:.4f} / {hybrid_overall[1]:.4f})")

    beats_garch = avg_h_rmse < avg_g_rmse and avg_h_mae < avg_g_mae
    beats_sma = avg_h_rmse < avg_s_rmse and avg_h_mae < avg_s_mae
    beats_park = avg_h_rmse < avg_p_rmse and avg_h_mae < avg_p_mae
    print(f"\nGARCH 대비 개선: {beats_garch}")
    print(f"SMA20 대비 개선: {beats_sma}")
    print(f"Parkinson-SMA20 대비 개선: {beats_park}")

    any_beat = beats_garch or beats_sma or beats_park
    print("\n" + "=" * 100)
    print(f"=== 4) 셔플 테스트(N={N_SHUFFLE}) — {'실행함(baseline 하나 이상 이김)' if any_beat else '생략(baseline 전부 못 이김)'} ===")
    print("=" * 100)
    if not any_beat:
        print("baseline을 하나도 못 이겨 셔플 검증 불필요 — 생략")
    else:
        shuffle_rmse, shuffle_mae = [], []
        for i, seed in enumerate(SHUFFLE_SEEDS):
            rng = np.random.RandomState(seed)
            y_shuf = y_train.copy()
            rng.shuffle(y_shuf)
            r = train_once(y_shuf, f"hybrid50-shuffle{i}", verbose=False)
            preds_s, actuals_s, _ = eval_loader(r["model"], X_test_s, y_test, tid_test, device)
            rmse_s, mae_s = rmse_mae(preds_s, actuals_s)
            shuffle_rmse.append(rmse_s); shuffle_mae.append(mae_s)
            print(f"  [{i+1}/{N_SHUFFLE}] seed={seed}: RMSE={rmse_s:.6f}  MAE={mae_s:.6f}  best_epoch={r['best_epoch']}")
        shuffle_rmse = np.array(shuffle_rmse); shuffle_mae = np.array(shuffle_mae)

        def _p(real_val, arr):
            beat = int(np.sum(arr <= real_val))
            return beat, (beat + 1) / (N_SHUFFLE + 1)

        n_beat_rmse, p_rmse = _p(hybrid_overall[0], shuffle_rmse)
        n_beat_mae, p_mae = _p(hybrid_overall[1], shuffle_mae)
        print(f"\n셔플 분포: RMSE {shuffle_rmse.mean():.6f}±{shuffle_rmse.std():.6f} | 실제 {hybrid_overall[0]:.6f}")
        print(f"셔플 분포: MAE  {shuffle_mae.mean():.6f}±{shuffle_mae.std():.6f} | 실제 {hybrid_overall[1]:.6f}")
        print(f"노이즈가 실제와 같거나 더 좋은 RMSE: {n_beat_rmse}/{N_SHUFFLE} (p={p_rmse:.4f})")
        print(f"노이즈가 실제와 같거나 더 좋은 MAE:  {n_beat_mae}/{N_SHUFFLE} (p={p_mae:.4f})")

    print("\n" + "=" * 100)
    print("=== 최종 판정 ===")
    print("=" * 100)
    print(f"GARCH 대비 개선: {beats_garch}  |  SMA20 대비 개선: {beats_sma}  |  Parkinson-SMA20 대비 개선: {beats_park}")
    print(f"예측 표준편차/target 표준편차 비율: {dist['std']/y_test.std()*100:.1f}% (8종목 시절 9.5%)")


if __name__ == "__main__":
    main()
