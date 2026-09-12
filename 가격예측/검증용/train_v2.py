# train_v2.py — Task T-1: 피처 재설계 후 재학습 (val만 확인, test는 이 스크립트에서 아예 안 만짐)
#
# diagnose_baseline.py 진단 결과 반영:
#   - OHLC 레벨 -> close_return/hl_range_ratio/open_gap_ratio (정상성), volume -> log1p
#   - 국고채 3년/10년(r=0.989) -> 스프레드(10년-3년) 1개로 축소
#   - 나머지 피처/모델 하이퍼파라미터는 체크포인트4와 동일하게 고정 (피처 재설계 효과만 분리 관찰)
#
# 비교 기준선을 만들기 위해 체크포인트4와 동일한 구(v1) 피처셋도 이 스크립트 안에서 다시 학습한다
# (이미 사람에게 보고된 결과의 재현일 뿐 — test는 v1/v2 둘 다 여기서 전혀 로드하지 않는다).

import sys

import numpy as np
import torch

from constants import STRESS_PERIOD_START
from 가격예측.sequence_dataset import FeatureScaler, build_sequences, split_sequences_by_date
from 가격예측.split_dataset import build_merged_dataset, build_merged_dataset_v2, split_normal_regime
from 가격예측.train_common import (
    directional_accuracy,
    evaluate_predictions,
    majority_baseline_accuracy,
    train_transformer,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TICKER = "005930"
START = "2020-01-02"
END = "2026-07-31"
LOOKBACK = 20
SEED = 42

# 체크포인트4와 동일 (피처 재설계 효과만 분리 관찰하기 위해 고정)
D_MODEL = 32
NHEAD = 2
NUM_LAYERS = 2
DIM_FEEDFORWARD = 64
DROPOUT = 0.3
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
SMOKE_EPOCHS = 2
MAX_EPOCHS = 60
PATIENCE = 8


def prepare_train_val(build_fn, label):
    """test는 만들지도 않는다 — train/val 시퀀스만 반환."""
    merged, meta = build_fn(TICKER, START, END)
    train_df, val_df, test_df, normal, stress = split_normal_regime(merged, STRESS_PERIOD_START)
    feature_cols = [c for c in merged.columns if c != "target"]

    # normal 전체로 시퀀스를 만들되, test 구간 시퀀스는 이 자리에서 바로 버린다 (안 씀 명시)
    X_all, y_all, dates_all = build_sequences(normal, feature_cols, LOOKBACK)
    splits = split_sequences_by_date(X_all, y_all, dates_all, train_df.index.max(), val_df.index.max())
    X_train, y_train, d_train = splits["train"]
    X_val, y_val, d_val = splits["val"]
    del splits["test"]  # test 시퀀스는 참조도 안 만든다

    scaler = FeatureScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    print(f"\n=== [{label}] 데이터 ===")
    print(f"feature_cols({len(feature_cols)}개): {feature_cols}")
    print(f"X_train {X_train.shape}  X_val {X_val.shape}  (test 시퀀스는 생성하지 않음)")

    return X_train_s.astype(np.float32), y_train, X_val_s.astype(np.float32), y_val, len(feature_cols)


def run_variant(build_fn, label, device):
    X_train, y_train, X_val, y_val, n_features = prepare_train_val(build_fn, label)

    result = train_transformer(
        X_train, y_train, X_val, y_val, n_features, LOOKBACK,
        D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT,
        BATCH_SIZE, LR, WEIGHT_DECAY, SMOKE_EPOCHS, MAX_EPOCHS, PATIENCE,
        device, seed=SEED, label=label,
    )

    preds, actuals = evaluate_predictions(result["model"], result["val_loader"], device)
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    mae = float(np.mean(np.abs(preds - actuals)))
    dir_acc = directional_accuracy(preds, actuals)
    base_label, base_acc = majority_baseline_accuracy(y_val)

    return {
        "label": label,
        "n_features": n_features,
        "val_baseline_label": base_label,
        "val_baseline_acc": base_acc,
        "best_epoch": result["best_epoch"],
        "best_val_loss": result["best_val_loss"],
        "last_epoch": result["last_epoch"],
        "last_train_loss": result["last_train_loss"],
        "last_val_loss": result["last_val_loss"],
        "overfit_ratio_at_end": result["overfit_ratio_at_end"],
        "val_rmse": rmse,
        "val_mae": mae,
        "val_dir_acc": dir_acc,
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print("※ 이 스크립트는 test 데이터를 로드하지 않는다 (val까지만 확인, 지시사항 준수)")

    r_v1 = run_variant(build_merged_dataset, "v1(체크포인트4 피처, 재현)", device)
    r_v2 = run_variant(build_merged_dataset_v2, "v2(재설계 피처)", device)

    print("\n" + "=" * 70)
    print("=== val 기준 v1 vs v2 비교 (test는 전혀 사용하지 않음) ===")
    print("=" * 70)
    header = f"{'':<32}{'v1(체크포인트4 피처)':>22}{'v2(재설계 피처)':>18}"
    print(header)
    print(f"{'피처 수':<32}{r_v1['n_features']:>22}{r_v2['n_features']:>18}")
    print(f"{'best epoch':<32}{r_v1['best_epoch']:>22}{r_v2['best_epoch']:>18}")
    print(f"{'best val_loss(MSE)':<32}{r_v1['best_val_loss']:>22.6f}{r_v2['best_val_loss']:>18.6f}")
    print(f"{'last train_loss':<32}{r_v1['last_train_loss']:>22.6f}{r_v2['last_train_loss']:>18.6f}")
    print(f"{'last val_loss':<32}{r_v1['last_val_loss']:>22.6f}{r_v2['last_val_loss']:>18.6f}")
    print(f"{'val/train loss 비(마지막 epoch)':<32}{r_v1['overfit_ratio_at_end']:>22.3f}{r_v2['overfit_ratio_at_end']:>18.3f}"
          f"   (1.0에 가까울수록 과적합 적음)")
    print(f"{'val RMSE':<32}{r_v1['val_rmse']:>22.4f}{r_v2['val_rmse']:>18.4f}")
    print(f"{'val MAE':<32}{r_v1['val_mae']:>22.4f}{r_v2['val_mae']:>18.4f}")
    print(f"{'val 방향성 정확도':<32}{r_v1['val_dir_acc']:>21.2%}{r_v2['val_dir_acc']:>17.2%}")

    print("\n=== 다수클래스 baseline 대비 (val 기준, 각자의 val 구간 y로 개별 계산) ===")
    print(f"v1 val baseline: {r_v1['val_baseline_label']} 정확도={r_v1['val_baseline_acc']:.2%}  "
          f"vs 모델 {r_v1['val_dir_acc']:.2%}  (차이 {r_v1['val_dir_acc']-r_v1['val_baseline_acc']:+.2%})")
    print(f"v2 val baseline: {r_v2['val_baseline_label']} 정확도={r_v2['val_baseline_acc']:.2%}  "
          f"vs 모델 {r_v2['val_dir_acc']:.2%}  (차이 {r_v2['val_dir_acc']-r_v2['val_baseline_acc']:+.2%})")

    improved_dir_acc = r_v2["val_dir_acc"] > r_v1["val_dir_acc"]
    improved_overfit = r_v2["overfit_ratio_at_end"] < r_v1["overfit_ratio_at_end"]
    improved_rmse = r_v2["val_rmse"] < r_v1["val_rmse"]

    print("\n=== 판정 ===")
    print(f"val 방향성 정확도 개선(v2 > v1): {improved_dir_acc}  ({r_v1['val_dir_acc']:.2%} -> {r_v2['val_dir_acc']:.2%})")
    print(f"과적합 개선(v2 val/train 비 < v1): {improved_overfit}  ({r_v1['overfit_ratio_at_end']:.3f} -> {r_v2['overfit_ratio_at_end']:.3f})")
    print(f"val RMSE 개선(v2 < v1): {improved_rmse}  ({r_v1['val_rmse']:.4f} -> {r_v2['val_rmse']:.4f})")
