# test_evaluation_pooled50_hybrid_shuffle30.py — 50종목 하이브리드 모델의 셔플 검증을
# N=7 -> N=30으로 확장 (2026-09-06, 일회성). 동일 방식(pooled train 전체 y를 순열해 재학습),
# 처음 7개 시드는 이전 실행(test_evaluation_pooled50_hybrid.py)과 동일하게 유지해 N=7 부분
# 결과를 그대로 재현·비교할 수 있게 한다.

import sys
from datetime import date

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
from 가격예측.garch_baseline import rmse_mae
from 가격예측.pooled_dataset import build_pooled_sequences, compute_global_split_dates
from 가격예측.sequence_dataset import FeatureScaler
from 가격예측.split_dataset import build_merged_dataset_v2, build_merged_dataset_v2_volatility_hybrid
from 가격예측.train_common import evaluate_pooled_predictions, train_pooled_transformer
from 가격예측.test_evaluation_pooled50_hybrid import EMBEDDING_DIM_50
from 가격예측.가격예측_통합모델 import (
    BATCH_SIZE, DIM_FEEDFORWARD, DROPOUT, D_MODEL, LOOKBACK, LR,
    MAX_EPOCHS, NHEAD, NUM_LAYERS, PATIENCE, SEED, SMOKE_EPOCHS, WEIGHT_DECAY,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

N_SHUFFLE = 30
# 처음 7개는 이전 N=7 실행과 완전히 동일한 시드 — N=7 부분집합을 그대로 재현하기 위함
SHUFFLE_SEEDS = [0, 1, 7, 123, 2024, 777, 999,
                 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
                 79, 83, 89, 97, 101, 103]
assert len(SHUFFLE_SEEDS) == N_SHUFFLE


def eval_loader(model, X, y, tid, device):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(tid)),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    return evaluate_pooled_predictions(model, loader, device)


def main():
    tickers = ACTIVE_TICKERS
    end_date = date.today().isoformat()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    print(f"=== 50종목 하이브리드 셔플 검증 N={N_SHUFFLE}(N=7 부분집합 포함) ===", flush=True)

    ref_vol, _ = build_merged_dataset_v2(tickers[0], STOCK_INITIAL_LOAD_START, end_date)
    train_end, val_end = compute_global_split_dates(ref_vol.index)

    def hybrid_build_fn(ticker, s, e):
        return build_merged_dataset_v2_volatility_hybrid(ticker, s, e, train_end)

    splits, feature_cols, ticker_to_id, (train_end2, val_end2) = build_pooled_sequences(
        tickers, STOCK_INITIAL_LOAD_START, end_date, lookback=LOOKBACK, build_fn=hybrid_build_fn,
    )
    assert train_end2 == train_end and val_end2 == val_end
    X_train, y_train, tid_train, d_train, _ = splits["train"]
    X_val, y_val, tid_val, d_val, _ = splits["val"]
    X_test, y_test, tid_test, d_test, _ = splits["test"]
    print(f"시퀀스 shape: X_train {X_train.shape}  X_test {X_test.shape}", flush=True)

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

    print("\n=== 실제(real) 모델 재학습 ===", flush=True)
    result = train_once(y_train, "real", verbose=False)
    real_preds, real_actuals, _ = eval_loader(result["model"], X_test_s, y_test, tid_test, device)
    real_rmse, real_mae = rmse_mae(real_preds, real_actuals)
    print(f"real: RMSE={real_rmse:.6f}  MAE={real_mae:.6f}  best_epoch={result['best_epoch']}", flush=True)

    shuffle_rmse, shuffle_mae = [], []
    for i, seed in enumerate(SHUFFLE_SEEDS):
        rng = np.random.RandomState(seed)
        y_shuf = y_train.copy()
        rng.shuffle(y_shuf)
        r = train_once(y_shuf, f"shuffle{i}", verbose=False)
        preds_s, actuals_s, _ = eval_loader(r["model"], X_test_s, y_test, tid_test, device)
        rmse_s, mae_s = rmse_mae(preds_s, actuals_s)
        shuffle_rmse.append(rmse_s); shuffle_mae.append(mae_s)
        print(f"[{i+1}/{N_SHUFFLE}] seed={seed}: RMSE={rmse_s:.6f}  MAE={mae_s:.6f}  best_epoch={r['best_epoch']}", flush=True)

    shuffle_rmse = np.array(shuffle_rmse)
    shuffle_mae = np.array(shuffle_mae)

    def stats(real_val, arr, n):
        beat = int(np.sum(arr <= real_val))
        p = (beat + 1) / (n + 1)
        z = (arr.mean() - real_val) / arr.std() if arr.std() > 0 else float("inf")
        return beat, p, arr.mean(), arr.std(), z

    print("\n" + "=" * 100)
    print("=== N=7 부분집합(처음 7개 시드) vs N=30(전체) 비교 ===")
    print("=" * 100)
    for n, label in [(7, "N=7(부분집합)"), (30, "N=30(전체)")]:
        sub_rmse = shuffle_rmse[:n]
        sub_mae = shuffle_mae[:n]
        beat_r, p_r, mean_r, std_r, z_r = stats(real_rmse, sub_rmse, n)
        beat_m, p_m, mean_m, std_m, z_m = stats(real_mae, sub_mae, n)
        print(f"\n[{label}]")
        print(f"  RMSE: 셔플 {mean_r:.6f}±{std_r:.6f} | 실제 {real_rmse:.6f} | 격차={z_r:.1f}표준편차 | "
              f"{beat_r}/{n} 노이즈가 이김 | p={p_r:.4f}")
        print(f"  MAE : 셔플 {mean_m:.6f}±{std_m:.6f} | 실제 {real_mae:.6f} | 격차={z_m:.1f}표준편차 | "
              f"{beat_m}/{n} 노이즈가 이김 | p={p_m:.4f}")

    print("\n" + "=" * 100)
    print("=== 최종 판정 ===")
    print("=" * 100)
    beat_r30, p_r30, _, _, z_r30 = stats(real_rmse, shuffle_rmse, 30)
    beat_m30, p_m30, _, _, z_m30 = stats(real_mae, shuffle_mae, 30)
    sig = p_r30 < 0.05 and p_m30 < 0.05
    print(f"N=30 기준 p_rmse={p_r30:.4f}  p_mae={p_m30:.4f}  통계적 유의(p<0.05): {sig}")
    print(f"격차(표준편차 단위) — RMSE: {z_r30:.1f}σ, MAE: {z_m30:.1f}σ")
    if sig:
        print(">>> baseline 대비 개선이 통계적으로 유의함 — 노이즈로 재현 불가능한 수준의 격차")
    else:
        print(">>> N=30에서도 형식적 유의성(p<0.05) 미달 — 다만 격차(σ 단위)가 크면 표본크기(N)만의 "
              "해상도 한계일 뿐 실질적 신호로 볼 근거는 있음. 아래 σ 값을 함께 참고할 것")


if __name__ == "__main__":
    main()
