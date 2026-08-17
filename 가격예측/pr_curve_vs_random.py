# pr_curve_vs_random.py — Task T-1: PR curve sanity check
# A/B/C의 하락-클래스 PR curve를, 동일 val셋에 대한 완전 무작위 예측(5시드 평균+band)과
# 나란히 비교한다. A/B/C 곡선이 무작위 band 안에 묻히면 "신호 없음"이 확정되고,
# band 위에 일관되게 있으면 약한 신호가 있다는 근거가 된다. test는 사용하지 않는다.

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_recall_curve

from constants import STRESS_PERIOD_START
from 가격예측.sequence_dataset import FeatureScaler, build_sequences, split_sequences_by_date
from 가격예측.split_dataset import build_merged_dataset_v2, split_normal_regime
from 가격예측.train_common import evaluate_predictions, train_transformer

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TICKER = "005930"
START = "2020-01-02"
END = "2026-07-31"
LOOKBACK = 20
SEED = 42  # A/B/C는 지난 턴과 동일 시드로 재현 (특정 결과를 재현하는 게 목적, 재현성 검증은 별도로 이미 함)

BATCH_SIZE, LR, WEIGHT_DECAY = 32, 1e-3, 1e-4
SMOKE_EPOCHS, MAX_EPOCHS, PATIENCE = 2, 60, 8

RANDOM_SEEDS = [1, 2, 3, 4, 5]
GRID = np.linspace(0, 1, 101)  # recall 0.00 ~ 1.00, 0.01 간격
OUT_DIR = "가격예측/checkpoints"


def interp_pr(precision, recall, grid):
    """recall 오름차순 정렬 후 공통 grid에 보간 (여러 곡선을 같은 x축에서 비교하기 위함)."""
    order = np.argsort(recall)
    r_sorted, p_sorted = recall[order], precision[order]
    return np.interp(grid, r_sorted, p_sorted)


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

    y_true = (y_val < 0).astype(int)
    down_rate = float(y_true.mean())
    print(f"val {len(y_val)}개, 하락 비율(no-skill 기대선) = {down_rate:.2%}")

    # --- A/B/C 예측 (지난 턴과 동일 설정, seed=42 재현) ---
    X_train_last, X_val_last = X_train_s[:, -1, :], X_val_s[:, -1, :]
    preds_A = LinearRegression().fit(X_train_last, y_train).predict(X_val_last)

    result_b = train_transformer(
        X_train_s, y_train, X_val_s, y_val, n_features, LOOKBACK,
        d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.3,
        batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY,
        smoke_epochs=SMOKE_EPOCHS, max_epochs=MAX_EPOCHS, patience=PATIENCE,
        device=device, seed=SEED, label="B", verbose=False,
    )
    preds_B, _ = evaluate_predictions(result_b["model"], result_b["val_loader"], device)

    result_c = train_transformer(
        X_train_s, y_train, X_val_s, y_val, n_features, LOOKBACK,
        d_model=32, nhead=2, num_layers=2, dim_feedforward=64, dropout=0.3,
        batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY,
        smoke_epochs=SMOKE_EPOCHS, max_epochs=MAX_EPOCHS, patience=PATIENCE,
        device=device, seed=SEED, label="C", verbose=False,
    )
    preds_C, _ = evaluate_predictions(result_c["model"], result_c["val_loader"], device)

    model_preds = {
        "A) 선형회귀(v2)": preds_A,
        "B) Transformer 소형(d16/L1)": preds_B,
        "C) Transformer 기존(d32/L2)": preds_C,
    }
    colors = {"A) 선형회귀(v2)": "tab:blue", "B) Transformer 소형(d16/L1)": "tab:orange",
              "C) Transformer 기존(d32/L2)": "tab:green"}

    model_curves = {}
    for name, preds in model_preds.items():
        precision, recall, _ = precision_recall_curve(y_true, -preds)
        model_curves[name] = interp_pr(precision, recall, GRID)

    # --- 무작위 예측 5시드 ---
    print(f"\n무작위 예측 {len(RANDOM_SEEDS)}시드 생성 중: {RANDOM_SEEDS}")
    random_curves = []
    for seed in RANDOM_SEEDS:
        rng = np.random.RandomState(seed)
        rand_scores = rng.randn(len(y_val))  # val과 완전 무관한 순수 잡음
        precision, recall, _ = precision_recall_curve(y_true, rand_scores)
        random_curves.append(interp_pr(precision, recall, GRID))
    random_curves = np.array(random_curves)  # (5, 101)
    random_mean = random_curves.mean(axis=0)
    random_min = random_curves.min(axis=0)
    random_max = random_curves.max(axis=0)
    random_std = random_curves.std(axis=0)

    # ============================================================
    # 그래프
    # ============================================================
    plt.figure(figsize=(9, 7))
    plt.fill_between(GRID, random_min, random_max, color="gray", alpha=0.25,
                      label=f"무작위 예측 범위 ({len(RANDOM_SEEDS)}시드 min~max)")
    plt.plot(GRID, random_mean, color="gray", linestyle="--", label="무작위 예측 평균")
    for name, curve in model_curves.items():
        plt.plot(GRID, curve, label=name, color=colors[name])
    plt.axhline(down_rate, color="black", linestyle=":", linewidth=1,
                label=f"하락 비율(no-skill 기대선, {down_rate:.1%})")
    plt.xlabel("Recall (하락)")
    plt.ylabel("Precision (하락)")
    plt.title("Precision-Recall curve: A/B/C vs 무작위 예측 (val)")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    path = f"{OUT_DIR}/pr_curve_vs_random.png"
    plt.savefig(path, dpi=150)
    print(f"그래프 저장: {path}")

    # ============================================================
    # 표 + 판정
    # ============================================================
    print("\n" + "=" * 100)
    print("=== Precision 비교: 모델 vs 무작위(mean/min~max), recall 체크포인트별 ===")
    print("=" * 100)
    checkpoints = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    header = f"{'recall':<8}" + "".join(f"{'무작위(mean, min~max)':<26}") + "".join(f"{n:<26}" for n in model_curves)
    print(header)
    for r in checkpoints:
        idx = int(round(r * 100))
        row = f"{r:<8.1f}{random_mean[idx]:.1%} ({random_min[idx]:.1%}~{random_max[idx]:.1%})".ljust(8 + 26)
        for name in model_curves:
            row += f"{model_curves[name][idx]:<26.1%}"
        print(row)

    # 중간 구간(recall 0.1~0.9)에서 모델 곡선이 무작위 band 위쪽(max) 밖에 있는 비율
    lo, hi = 10, 90  # GRID index for recall 0.10~0.90
    print("\n=== 판정: recall 0.10~0.90 구간에서 모델이 무작위 band 상단(max)보다 높은 비율 ===")
    verdicts = {}
    for name, curve in model_curves.items():
        above = curve[lo:hi + 1] > random_max[lo:hi + 1]
        pct_above = float(above.mean())
        mean_gap_vs_mean = float((curve[lo:hi + 1] - random_mean[lo:hi + 1]).mean())
        verdicts[name] = (pct_above, mean_gap_vs_mean)
        print(f"  {name:<28}band 상단 초과 비율={pct_above:>6.1%}   평균 대비 평균 gap={mean_gap_vs_mean:+.4f}")

    print("\n=== 최종 판정 ===")
    for name, (pct_above, gap) in verdicts.items():
        if pct_above >= 0.7 and gap > 0:
            verdict = "무작위보다 일관되게 위 — 약한 신호 있음 가능성"
        elif pct_above <= 0.3:
            verdict = "무작위 band 안에 거의 묻힘 — 신호 없음(랜덤과 구분 안 됨)"
        else:
            verdict = "혼재 — 일부 구간만 무작위 위, 명확한 결론 어려움"
        print(f"  {name:<28}{verdict}")
