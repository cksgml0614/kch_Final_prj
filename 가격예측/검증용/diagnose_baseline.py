# diagnose_baseline.py — Task T-1 베이스라인 저조 원인 진단
# 모델을 손대기 전에 신호 존재 여부를 먼저 확인한다 (KoBERT 1차 실패 때 TF-IDF baseline으로
# 레이블 신호부터 진단했던 접근과 동일). 여기서는 재학습하지 않는다.
#
# 1) 선형회귀 baseline — 동일 피처/분할, "마지막 시점(=t-1) 피처만" 사용하는 단순 버전 채택.
#    flatten(20*15=300차원)은 train 983개 대비 과적합 위험이 커서 배제.
# 2) 개별 피처 - target_t 피어슨 상관계수 (train 기준, unscaled 원본 값)
# 3) 피처간 상관관계 히트맵(다중공선성) — TASK_T 설계결정 3에서 "학습 후 사후 검증"하기로
#    미뤄뒀던 항목을 지금 실행한다.

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

from constants import STRESS_PERIOD_START
from 가격예측.sequence_dataset import FeatureScaler, build_sequences, split_sequences_by_date
from 가격예측.split_dataset import build_merged_dataset, split_normal_regime
from 가격예측.검증용.train import directional_accuracy, majority_baseline_accuracy

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TICKER = "005930"
START = "2020-01-02"
END = "2026-07-31"
LOOKBACK = 20
OUT_DIR = "가격예측/checkpoints"

# 체크포인트 4(2026-08-17) 결과 — 재학습하지 않고 비교용으로 그대로 인용
TRANSFORMER_RESULT = {"RMSE": 0.032813, "MAE": 0.026662, "방향성 정확도": 0.3953}


def rmse_mae(preds, actuals):
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    mae = float(np.mean(np.abs(preds - actuals)))
    return rmse, mae


if __name__ == "__main__":
    import os

    os.makedirs(OUT_DIR, exist_ok=True)

    merged, meta = build_merged_dataset(TICKER, START, END)
    train_df, val_df, test_df, normal, stress = split_normal_regime(merged, STRESS_PERIOD_START)
    feature_cols = [c for c in merged.columns if c != "target"]

    X_all, y_all, dates_all = build_sequences(normal, feature_cols, LOOKBACK)
    splits = split_sequences_by_date(X_all, y_all, dates_all, train_df.index.max(), val_df.index.max())
    X_train, y_train, d_train = splits["train"]
    X_val, y_val, d_val = splits["val"]
    X_test, y_test, d_test = splits["test"]

    # ============================================================
    # 1) 선형회귀 baseline (마지막 시점=t-1 피처만, 체크포인트 4와 동일 스케일러 방식)
    # ============================================================
    print("=== 1) 선형회귀 baseline (마지막 시점 피처만, 동일 train/val/test) ===")

    scaler = FeatureScaler().fit(X_train)
    X_train_last = scaler.transform(X_train)[:, -1, :]
    X_test_last = scaler.transform(X_test)[:, -1, :]

    lr = LinearRegression()
    lr.fit(X_train_last, y_train)
    lr_preds = lr.predict(X_test_last)

    lr_rmse, lr_mae = rmse_mae(lr_preds, y_test)
    lr_dir_acc = directional_accuracy(lr_preds, y_test)
    base_label, base_acc = majority_baseline_accuracy(y_test)

    print(f"{'모델':<22}{'RMSE':>10}{'MAE':>10}{'방향성정확도':>14}")
    print(f"{'다수클래스 baseline':<22}{'-':>10}{'-':>10}{base_acc:>13.2%}  ({base_label})")
    print(f"{'선형회귀(t-1 피처)':<22}{lr_rmse:>10.4f}{lr_mae:>10.4f}{lr_dir_acc:>13.2%}")
    print(f"{'Transformer(체크포인트4)':<22}{TRANSFORMER_RESULT['RMSE']:>10.4f}"
          f"{TRANSFORMER_RESULT['MAE']:>10.4f}{TRANSFORMER_RESULT['방향성 정확도']:>13.2%}")

    coefs = pd.Series(lr.coef_, index=feature_cols).sort_values(key=np.abs, ascending=False)
    print("\n선형회귀 계수 (|계수| 내림차순, 스케일된 피처 기준이라 상대적 영향력으로 해석):")
    for name, c in coefs.items():
        print(f"  {name:<28}{c:+.6f}")

    # ============================================================
    # 2) 개별 피처 - target_t 피어슨 상관계수 (train 기준, unscaled)
    # ============================================================
    print("\n=== 2) 개별 피처 - target_t 피어슨 상관계수 (train, unscaled) ===")

    train_raw_last = pd.DataFrame(X_train[:, -1, :], columns=feature_cols)
    train_raw_last["target"] = y_train

    corr_with_target = train_raw_last.corr(numeric_only=True)["target"].drop("target")
    corr_sorted = corr_with_target.reindex(corr_with_target.abs().sort_values(ascending=False).index)

    print(f"{'피처':<28}{'상관계수':>10}")
    for name, c in corr_sorted.items():
        print(f"  {name:<28}{c:+.4f}")

    max_abs_corr = corr_sorted.abs().max()
    print(f"\n최대 |상관계수|: {max_abs_corr:.4f} ({corr_sorted.abs().idxmax()})")

    # ============================================================
    # 3) 피처간 상관관계 히트맵 (다중공선성)
    # ============================================================
    print("\n=== 3) 피처간 상관관계 (다중공선성) ===")

    feat_corr = train_raw_last[feature_cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(feat_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                square=True, ax=ax, cbar_kws={"label": "Pearson correlation"})
    ax.set_title(f"Feature correlation heatmap (train, n={len(train_raw_last)})")
    plt.tight_layout()
    heatmap_path = f"{OUT_DIR}/feature_correlation_heatmap.png"
    fig.savefig(heatmap_path, dpi=150)
    print(f"히트맵 저장: {heatmap_path}")

    # |corr| > 0.8인 쌍을 텍스트로도 나열 (이미지 없이도 확인 가능하도록)
    pairs = []
    cols = feat_corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = feat_corr.iloc[i, j]
            if abs(v) > 0.8:
                pairs.append((cols[i], cols[j], v))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"\n|상관계수| > 0.8인 피처 쌍: {len(pairs)}건")
    for a, b, v in pairs:
        print(f"  {a:<26} vs {b:<26} r={v:+.4f}")
