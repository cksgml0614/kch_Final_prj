# seed_stability_check.py — Task T-1: B(Transformer 소형, d16/L1) 여러 시드 재현성 확인
# val만 사용 (test 미사용). 방향성 정확도뿐 아니라 하락 precision/recall/F2도 시드별로 기록한다.
#
# 게이트+F2 기준 (2026-08-17 확정, 사용자 승인):
#   - precision이 val 하락 비율(baseline) 대비 +2%p 이상이어야 F2 비교 대상으로 인정 (게이트)
#   - recall 최소선 0.5
#   - 게이트를 통과하는 시드 비율로 "이전 턴의 B 결과(52.09% 방향성정확도)가 우연이었는지"를 본다

import sys

import numpy as np
import torch

from constants import STRESS_PERIOD_START
from 가격예측.diagnose_down_class import confusion_at_threshold
from 가격예측.sequence_dataset import FeatureScaler, build_sequences, split_sequences_by_date
from 가격예측.split_dataset import build_merged_dataset_v2, split_normal_regime
from 가격예측.train_common import directional_accuracy, evaluate_predictions, majority_baseline_accuracy, train_transformer

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TICKER = "005930"
START = "2020-01-02"
END = "2026-07-31"
LOOKBACK = 20

SEEDS = [42, 0, 1, 7, 123, 2024, 777]

D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT = 16, 2, 1, 32, 0.3
BATCH_SIZE, LR, WEIGHT_DECAY = 32, 1e-3, 1e-4
SMOKE_EPOCHS, MAX_EPOCHS, PATIENCE = 2, 60, 8

PRECISION_GATE_MARGIN = 0.02  # baseline(하락 비율) 대비 +2%p
RECALL_FLOOR = 0.5


def rmse_mae(preds, actuals):
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    mae = float(np.mean(np.abs(preds - actuals)))
    return rmse, mae


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print("※ test 데이터는 로드하지 않는다 (val까지만)")
    print(f"모델: Transformer 소형(d_model={D_MODEL}, nhead={NHEAD}, layer={NUM_LAYERS}) — B와 동일 설정")
    print(f"시드: {SEEDS}")

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
    down_rate = float((y_val < 0).mean())
    precision_gate = down_rate + PRECISION_GATE_MARGIN
    print(f"\nval baseline: {base_label} 정확도={base_acc:.2%}  (하락 비율={down_rate:.2%})")
    print(f"precision 게이트: >= {precision_gate:.2%} (하락비율+{PRECISION_GATE_MARGIN:.0%}p)  recall 최소선: >= {RECALL_FLOOR:.0%}")

    rows = []
    for seed in SEEDS:
        result = train_transformer(
            X_train_s, y_train, X_val_s, y_val, n_features, LOOKBACK,
            D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT,
            BATCH_SIZE, LR, WEIGHT_DECAY, SMOKE_EPOCHS, MAX_EPOCHS, PATIENCE,
            device, seed=seed, label=f"seed{seed}", verbose=False,
        )
        preds, actuals = evaluate_predictions(result["model"], result["val_loader"], device)
        rmse, mae = rmse_mae(preds, actuals)
        dir_acc = directional_accuracy(preds, actuals)
        m = confusion_at_threshold(preds, actuals, threshold=0.0)

        gate_pass = m["precision"] >= precision_gate
        recall_pass = m["recall"] >= RECALL_FLOOR
        both_pass = gate_pass and recall_pass

        rows.append({
            "seed": seed, "best_epoch": result["best_epoch"], "best_val_loss": result["best_val_loss"],
            "rmse": rmse, "mae": mae, "dir_acc": dir_acc,
            "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "f2": m["f2"],
            "gate_pass": gate_pass, "recall_pass": recall_pass, "both_pass": both_pass,
        })
        print(f"  seed={seed}: best_epoch={result['best_epoch']:>3}  dir_acc={dir_acc:.2%}  "
              f"precision={m['precision']:.2%}  recall={m['recall']:.2%}  F2={m['f2']:.3f}  "
              f"게이트통과={gate_pass}  recall≥0.5={recall_pass}")

    print("\n" + "=" * 92)
    print("=== 시드별 결과 표 ===")
    print("=" * 92)
    print(f"{'seed':<8}{'best_ep':>8}{'RMSE':>9}{'MAE':>9}{'방향성정확도':>13}{'Precision':>11}{'Recall':>9}{'F1':>7}{'F2':>7}{'게이트+recall≥0.5':>18}")
    for r in rows:
        print(f"{r['seed']:<8}{r['best_epoch']:>8}{r['rmse']:>9.4f}{r['mae']:>9.4f}{r['dir_acc']:>12.2%}"
              f"{r['precision']:>11.2%}{r['recall']:>9.2%}{r['f1']:>7.3f}{r['f2']:>7.3f}"
              f"{('통과' if r['both_pass'] else '-'):>18}")

    dir_accs = np.array([r["dir_acc"] for r in rows])
    precisions = np.array([r["precision"] for r in rows])
    recalls = np.array([r["recall"] for r in rows])
    f2s = np.array([r["f2"] for r in rows])
    n_pass = sum(r["both_pass"] for r in rows)

    print("\n=== 시드 간 요약 통계 (mean ± std, n={}개 시드) ===".format(len(SEEDS)))
    print(f"방향성 정확도: {dir_accs.mean():.2%} ± {dir_accs.std():.2%}  (min {dir_accs.min():.2%} / max {dir_accs.max():.2%})")
    print(f"Precision(하락):  {precisions.mean():.2%} ± {precisions.std():.2%}")
    print(f"Recall(하락):     {recalls.mean():.2%} ± {recalls.std():.2%}")
    print(f"F2(하락):        {f2s.mean():.3f} ± {f2s.std():.3f}")
    print(f"\n게이트(precision>=baseline+2%p) AND recall>=0.5 동시 통과 시드: {n_pass}/{len(SEEDS)}")

    print(f"\n(참고) 지난 턴 seed=42 단일 결과: dir_acc=52.09%, baseline=50.70% — "
          f"이번 seed=42 재현값: dir_acc={rows[0]['dir_acc']:.2%}"
          f"{' (일치)' if rows[0]['seed']==42 else ''}")
