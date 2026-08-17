# compare_v2_variants.py — Task T-1: v2(재설계) 피처 위에서 3가지 모델 비교
# A) 선형회귀 (마지막 시점 피처만, 진단 단계와 동일 방식)
# B) Transformer 축소판 (d_model=16, nhead=2, layer=1)
# C) Transformer 기존 규모 (d_model=32, nhead=2, layer=2, ~18k 파라미터) — train_v2.py의 v2와 동일 설정 재현
#
# val만 사용, test는 이 스크립트에서 아예 로드하지 않는다.

import sys

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from constants import STRESS_PERIOD_START
from 가격예측.sequence_dataset import FeatureScaler, build_sequences, split_sequences_by_date
from 가격예측.split_dataset import build_merged_dataset_v2, split_normal_regime
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

BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
SMOKE_EPOCHS = 2
MAX_EPOCHS = 60
PATIENCE = 8


def rmse_mae(preds, actuals):
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    mae = float(np.mean(np.abs(preds - actuals)))
    return rmse, mae


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print("※ test 데이터는 이 스크립트에서 로드하지 않는다 (val까지만)")

    merged, meta = build_merged_dataset_v2(TICKER, START, END)
    train_df, val_df, test_df, normal, stress = split_normal_regime(merged, STRESS_PERIOD_START)
    feature_cols = [c for c in merged.columns if c != "target"]

    X_all, y_all, dates_all = build_sequences(normal, feature_cols, LOOKBACK)
    splits = split_sequences_by_date(X_all, y_all, dates_all, train_df.index.max(), val_df.index.max())
    X_train, y_train, d_train = splits["train"]
    X_val, y_val, d_val = splits["val"]
    del splits["test"]  # test 시퀀스는 참조도 안 만든다

    print(f"\nfeature_cols({len(feature_cols)}개): {feature_cols}")
    print(f"X_train {X_train.shape}  X_val {X_val.shape}  (test 시퀀스는 생성하지 않음)")

    scaler = FeatureScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    base_label, base_acc = majority_baseline_accuracy(y_val)
    print(f"\nval 다수클래스 baseline: {base_label}  정확도={base_acc:.2%}")

    n_features = X_train.shape[-1]
    results = {}

    # ============================================================
    # A) 선형회귀 (v2 피처, 마지막 시점=t-1만 — 진단 단계와 동일 방식)
    # ============================================================
    print("\n=== A) 선형회귀 (v2 피처, 마지막 시점만) ===")
    X_train_last = X_train_s[:, -1, :]
    X_val_last = X_val_s[:, -1, :]

    lr_model = LinearRegression()
    lr_model.fit(X_train_last, y_train)
    lr_preds = lr_model.predict(X_val_last)
    lr_rmse, lr_mae = rmse_mae(lr_preds, y_val)
    lr_dir_acc = directional_accuracy(lr_preds, y_val)

    results["A_linear"] = {
        "label": "A) 선형회귀(v2)", "n_params": X_train_last.shape[1] + 1,
        "rmse": lr_rmse, "mae": lr_mae, "dir_acc": lr_dir_acc,
    }
    print(f"val RMSE={lr_rmse:.4f}  MAE={lr_mae:.4f}  방향성정확도={lr_dir_acc:.2%}")

    # ============================================================
    # B) Transformer 축소판 (d_model=16, nhead=2, layer=1)
    # ============================================================
    print("\n=== B) Transformer 축소판 (d_model=16, nhead=2, layer=1, ff=32) ===")
    result_small = train_transformer(
        X_train_s, y_train, X_val_s, y_val, n_features, LOOKBACK,
        d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.3,
        batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY,
        smoke_epochs=SMOKE_EPOCHS, max_epochs=MAX_EPOCHS, patience=PATIENCE,
        device=device, seed=SEED, label="B",
    )
    preds_small, actuals_small = evaluate_predictions(result_small["model"], result_small["val_loader"], device)
    rmse_small, mae_small = rmse_mae(preds_small, actuals_small)
    dir_acc_small = directional_accuracy(preds_small, actuals_small)
    total_p_small = sum(p.numel() for p in result_small["model"].parameters())
    results["B_small"] = {
        "label": "B) Transformer 소형(d16/L1)", "n_params": total_p_small,
        "rmse": rmse_small, "mae": mae_small, "dir_acc": dir_acc_small,
        "best_epoch": result_small["best_epoch"], "overfit_ratio": result_small["overfit_ratio_at_end"],
    }

    # ============================================================
    # C) Transformer 기존 규모 (d_model=32, nhead=2, layer=2) — v2 피처로 재현
    # ============================================================
    print("\n=== C) Transformer 기존 규모 (d_model=32, nhead=2, layer=2, ff=64) v2 피처 ===")
    result_orig = train_transformer(
        X_train_s, y_train, X_val_s, y_val, n_features, LOOKBACK,
        d_model=32, nhead=2, num_layers=2, dim_feedforward=64, dropout=0.3,
        batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY,
        smoke_epochs=SMOKE_EPOCHS, max_epochs=MAX_EPOCHS, patience=PATIENCE,
        device=device, seed=SEED, label="C",
    )
    preds_orig, actuals_orig = evaluate_predictions(result_orig["model"], result_orig["val_loader"], device)
    rmse_orig, mae_orig = rmse_mae(preds_orig, actuals_orig)
    dir_acc_orig = directional_accuracy(preds_orig, actuals_orig)
    total_p_orig = sum(p.numel() for p in result_orig["model"].parameters())
    results["C_orig"] = {
        "label": "C) Transformer 기존(d32/L2)", "n_params": total_p_orig,
        "rmse": rmse_orig, "mae": mae_orig, "dir_acc": dir_acc_orig,
        "best_epoch": result_orig["best_epoch"], "overfit_ratio": result_orig["overfit_ratio_at_end"],
    }

    # ============================================================
    # 비교표
    # ============================================================
    print("\n" + "=" * 82)
    print(f"=== v2 피처 위 3-way 비교 (val 기준, baseline={base_acc:.2%} '{base_label}') ===")
    print("=" * 82)
    print(f"{'':<28}{'파라미터':>10}{'RMSE':>10}{'MAE':>10}{'방향성정확도':>14}{'baseline대비':>14}")
    for key in ["A_linear", "B_small", "C_orig"]:
        r = results[key]
        diff = r["dir_acc"] - base_acc
        print(f"{r['label']:<28}{r['n_params']:>10,}{r['rmse']:>10.4f}{r['mae']:>10.4f}"
              f"{r['dir_acc']:>13.2%}{diff:>+13.2%}")

    beats_baseline = [k for k in results if results[k]["dir_acc"] >= base_acc]
    labels = [results[k]["label"] for k in beats_baseline]
    print(f"\nval baseline({base_acc:.2%}) 이상인 조합: {labels if labels else '없음'}")

    closest = min(results, key=lambda k: abs(results[k]["dir_acc"] - base_acc))
    print(f"baseline에 가장 가까운 조합: {results[closest]['label']} "
          f"({results[closest]['dir_acc']:.2%}, 차이 {results[closest]['dir_acc']-base_acc:+.2%})")
