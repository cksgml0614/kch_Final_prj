# diagnose_down_class.py — Task T-1: 하락 클래스 precision/recall/F1/F2 + threshold sweep
# A(선형회귀)/B(소형 Transformer)/C(기존 Transformer), 전부 v2 피처. val만 사용, test 미사용.
# compare_v2_variants.py와 동일한 데이터/학습 설정을 재현해 예측값을 얻은 뒤,
# "하락"(actual target < 0)을 양성 클래스로 놓고 confusion matrix/PR curve를 계산한다.
#
# 2026-09-06: confusion_at_threshold()는 train_common.py로 이관했다(자동화 파이프라인의
# 배포 게이트가 재사용 — 일회성 진단 스크립트를 import하지 않도록). 이 파일은 값 변경 없음.

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"  # 한글 글리프 누락 방지 (Windows 기본 폰트)
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_recall_curve

from constants import STRESS_PERIOD_START
from 가격예측.sequence_dataset import FeatureScaler, build_sequences, split_sequences_by_date
from 가격예측.split_dataset import build_merged_dataset_v2, split_normal_regime
from 가격예측.train_common import (
    confusion_at_threshold,
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

OUT_DIR = "가격예측/checkpoints"


def find_precision_cliff(precision, recall, min_recall=0.1):
    """recall이 오름차순이 되도록 정렬한 뒤, 한 스텝에서 precision이 가장 급격히 떨어지는
    지점을 찾는다. recall<min_recall 구간은 표본이 1~2개뿐인 극단 임계값이라 노이즈가 커서
    제외한다 (sklearn precision_recall_curve의 recall=0, precision=1 경계점 포함)."""
    order = np.argsort(recall)
    r_sorted = recall[order]
    p_sorted = precision[order]

    mask = r_sorted >= min_recall
    r_sorted, p_sorted = r_sorted[mask], p_sorted[mask]

    drops = -np.diff(p_sorted)  # 양수면 하락
    if len(drops) == 0:
        return None
    idx = int(np.argmax(drops))
    return {"recall_before": float(r_sorted[idx]), "recall_after": float(r_sorted[idx + 1]),
            "precision_before": float(p_sorted[idx]), "precision_after": float(p_sorted[idx + 1]),
            "drop": float(drops[idx])}


if __name__ == "__main__":
    import os

    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print("※ test 데이터는 로드하지 않는다 (val까지만)")

    merged, meta = build_merged_dataset_v2(TICKER, START, END)
    train_df, val_df, test_df, normal, stress = split_normal_regime(merged, STRESS_PERIOD_START)
    feature_cols = [c for c in merged.columns if c != "target"]

    X_all, y_all, dates_all = build_sequences(normal, feature_cols, LOOKBACK)
    splits = split_sequences_by_date(X_all, y_all, dates_all, train_df.index.max(), val_df.index.max())
    X_train, y_train, d_train = splits["train"]
    X_val, y_val, d_val = splits["val"]
    del splits["test"]

    scaler = FeatureScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    n_features = X_train.shape[-1]

    base_label, base_acc = majority_baseline_accuracy(y_val)
    n_down_val = int((y_val < 0).sum())
    print(f"\nval baseline: {base_label} 정확도={base_acc:.2%}  (val {len(y_val)}개 중 하락 {n_down_val}개, "
          f"{n_down_val/len(y_val):.1%})")

    preds_by_model = {}

    # --- A) 선형회귀 ---
    X_train_last = X_train_s[:, -1, :]
    X_val_last = X_val_s[:, -1, :]
    lr_model = LinearRegression().fit(X_train_last, y_train)
    preds_by_model["A) 선형회귀(v2)"] = lr_model.predict(X_val_last)

    # --- B) Transformer 소형 (d16/L1) ---
    result_b = train_transformer(
        X_train_s, y_train, X_val_s, y_val, n_features, LOOKBACK,
        d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.3,
        batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY,
        smoke_epochs=SMOKE_EPOCHS, max_epochs=MAX_EPOCHS, patience=PATIENCE,
        device=device, seed=SEED, label="B", verbose=False,
    )
    preds_b, actuals_b = evaluate_predictions(result_b["model"], result_b["val_loader"], device)
    preds_by_model["B) Transformer 소형(d16/L1)"] = preds_b

    # --- C) Transformer 기존 (d32/L2) ---
    result_c = train_transformer(
        X_train_s, y_train, X_val_s, y_val, n_features, LOOKBACK,
        d_model=32, nhead=2, num_layers=2, dim_feedforward=64, dropout=0.3,
        batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY,
        smoke_epochs=SMOKE_EPOCHS, max_epochs=MAX_EPOCHS, patience=PATIENCE,
        device=device, seed=SEED, label="C", verbose=False,
    )
    preds_c, actuals_c = evaluate_predictions(result_c["model"], result_c["val_loader"], device)
    preds_by_model["C) Transformer 기존(d32/L2)"] = preds_c

    print("모델 3종 학습 완료 (verbose 로그는 이전 turn에서 이미 확인된 것과 동일 설정 재현)")

    # ============================================================
    # 1) threshold=0 (기존 sign 규칙) confusion matrix + precision/recall/F1/F2
    # ============================================================
    print("\n" + "=" * 90)
    print("=== 하락 클래스 confusion matrix (threshold=0, 기존 부호 규칙) ===")
    print("=" * 90)
    print(f"{'모델':<28}{'TP':>6}{'FP':>6}{'FN':>6}{'TN':>6}{'Precision':>11}{'Recall':>9}{'F1':>8}{'F2':>8}")

    metrics_by_model = {}
    for name, preds in preds_by_model.items():
        m = confusion_at_threshold(preds, y_val, threshold=0.0)
        metrics_by_model[name] = m
        print(f"{name:<28}{m['TP']:>6}{m['FP']:>6}{m['FN']:>6}{m['TN']:>6}"
              f"{m['precision']:>11.2%}{m['recall']:>9.2%}{m['f1']:>8.3f}{m['f2']:>8.3f}")

    # ============================================================
    # 2) precision-recall curve (threshold sweep) + 급락 지점
    # ============================================================
    print("\n" + "=" * 90)
    print("=== Precision-Recall curve (threshold sweep, '하락'=양성) ===")
    print("=" * 90)

    plt.figure(figsize=(8, 6))
    colors = {"A) 선형회귀(v2)": "tab:blue", "B) Transformer 소형(d16/L1)": "tab:orange",
              "C) Transformer 기존(d32/L2)": "tab:green"}

    for name, preds in preds_by_model.items():
        y_true = (y_val < 0).astype(int)
        score = -preds  # score가 클수록 '하락' 예측에 가까움
        precision, recall, thresholds = precision_recall_curve(y_true, score)

        plt.plot(recall, precision, label=name, color=colors.get(name))

        cliff = find_precision_cliff(precision, recall)
        print(f"\n[{name}]")
        # recall 체크포인트별 precision (가장 가까운 점)
        for target_r in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            idx = int(np.argmin(np.abs(recall - target_r)))
            print(f"  recall≈{target_r:.1f} 근처(실제 {recall[idx]:.2f}): precision={precision[idx]:.2%}")
        if cliff:
            print(f"  최대 낙폭 지점: recall {cliff['recall_before']:.2f}->{cliff['recall_after']:.2f} 사이, "
                  f"precision {cliff['precision_before']:.2%}->{cliff['precision_after']:.2%} "
                  f"(낙폭 {cliff['drop']:.2%}p)")

    plt.axhline(y=n_down_val / len(y_val), color="gray", linestyle="--", label=f"baseline(무작위, {n_down_val/len(y_val):.1%})")
    plt.xlabel("Recall (하락)")
    plt.ylabel("Precision (하락)")
    plt.title("Precision-Recall curve — 하락 클래스 (val)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    pr_path = f"{OUT_DIR}/pr_curve_down.png"
    plt.savefig(pr_path, dpi=150)
    print(f"\nPR curve 저장: {pr_path}")
