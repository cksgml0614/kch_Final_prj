# test_evaluation.py — Task T V2 최초 test 평가 (2026-09-06)
#
# 지금까지 val로만 검증됐던 V2 파이프라인의 test 평가를 최초로 1회 실행한다. 목적은 정확도
# 개선이 아니라 "val 기준 결론(무작위 수준)이 test에서도 재현되는지" 검증 — Task F(뉴스/감성)
# 트랙에서 확립한 엄격 기준을 그대로 적용한다: baseline(random walk/다수클래스) 상회만으로는
# 부족하고, 사전분포를 아는 무작위 예측기 및 레이블 셔플 재학습과 비교해야 신호 유무를 판정할
# 수 있다(결과_TaskF_게이팅검증.md [4]의 방법론적 교훈 — "baseline만 이기는 건 신호의 증거가
# 아니다").
#
# split_full_period(스트레스 분리 없는 전체 기간 70/15/15, 2026-09-06 설계 확정)를 쓴다 —
# split_normal_regime(스트레스 제외)이 아니다. 모델/하이퍼파라미터는 가격예측_공통.py의 운영
# 설정과 완전히 동일(D_MODEL=32/NHEAD=2/NUM_LAYERS=2/DIM_FEEDFORWARD=64/DROPOUT=0.3,
# LOOKBACK=20) — train_v2.py/compare_v2_variants.py의 "C) Transformer 기존(d32/L2)"와 같은
# 구성이다. 이 test 평가는 일회성이라 diagnose_baseline.py 등 기존 Task T-1 진단 스크립트와
# 같은 계열로 취급해 영문 파일명을 유지한다(가격예측_공통.py/가격예측_일일수집.py는 반복
# 실행되는 운영 파이프라인이라 한글명 채택 — 이 파일은 그 대상이 아님).
#
# 검증 절충안(사람 지시, 2026-09-06): 셔플 테스트는 회귀 모델 재학습 비용을 감안해 N=7로
# 축소(seed_stability_check.py의 7시드 규모를 그대로 따름 — 뉴스 트랙의 N=30~200보다 훨씬
# 적음). N=7에서 "실제가 셔플 전부보다 우수"해도 그 경험적 p-value는 1/8=0.125가 최솟값이라
# 해상도가 거칠다 — Task F에서 N=30이 냈던 p=1/31 경계 오판을 N=200 재검증으로 뒤집었던
# 전례가 있으므로, 이번에도 결과를 "N=7 기준"이라는 한계와 함께만 해석한다(과신 금지).

import sys
from datetime import date

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset

from constants import STOCK_INITIAL_LOAD_START
from 가격예측.split_dataset import build_merged_dataset_v2, split_full_period
from 가격예측.sequence_dataset import FeatureScaler, build_sequences, split_sequences_by_date
from 가격예측.train_common import (
    directional_accuracy,
    evaluate_predictions,
    majority_baseline_accuracy,
    train_transformer,
)
from 가격예측.가격예측_공통 import (
    BATCH_SIZE, DIM_FEEDFORWARD, DROPOUT, D_MODEL, LOOKBACK, LR, MAX_EPOCHS,
    NHEAD, NUM_LAYERS, PATIENCE, SEED, SMOKE_EPOCHS, WEIGHT_DECAY,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TICKER = "005930"
START = STOCK_INITIAL_LOAD_START

N_SHUFFLE = 7
SHUFFLE_SEEDS = [0, 1, 7, 123, 2024, 777, 999]  # y_train 순열용 시드(모델 초기화 시드와는 별개)
N_RANDOM_PRIOR = 200


def rmse_mae(preds, actuals):
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    mae = float(np.mean(np.abs(preds - actuals)))
    return rmse, mae


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    end = date.today().isoformat()
    print(f"device: {device}")
    print(f"=== V2 test 평가(최초 실행) — {TICKER}, {START}~{end} ===")

    merged, meta = build_merged_dataset_v2(TICKER, START, end)
    feature_cols = [c for c in merged.columns if c != "target"]
    train_df, val_df, test_df = split_full_period(merged)

    print(
        f"거래일 기준: train {len(train_df)}일({train_df.index.min().date()}~{train_df.index.max().date()}) / "
        f"val {len(val_df)}일({val_df.index.min().date()}~{val_df.index.max().date()}) / "
        f"test {len(test_df)}일({test_df.index.min().date()}~{test_df.index.max().date()})"
    )

    X_all, y_all, dates_all = build_sequences(merged, feature_cols, LOOKBACK)
    splits = split_sequences_by_date(X_all, y_all, dates_all, train_df.index.max(), val_df.index.max())
    X_train, y_train, d_train = splits["train"]
    X_val, y_val, d_val = splits["val"]
    X_test, y_test, d_test = splits["test"]
    print(f"시퀀스 shape: X_train {X_train.shape}  X_val {X_val.shape}  X_test {X_test.shape}")

    scaler = FeatureScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    def eval_on_test(model):
        loader = DataLoader(TensorDataset(torch.from_numpy(X_test_s), torch.from_numpy(y_test)),
                             batch_size=BATCH_SIZE, shuffle=False)
        return evaluate_predictions(model, loader, device)

    def train_once(y_train_arr, seed, label, verbose):
        return train_transformer(
            X_train_s, y_train_arr, X_val_s, y_val, len(feature_cols), LOOKBACK,
            D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT,
            BATCH_SIZE, LR, WEIGHT_DECAY, SMOKE_EPOCHS, MAX_EPOCHS, PATIENCE,
            device, seed=seed, verbose=verbose, label=label,
        )

    # ── 1) 실제 학습(train, val early stopping) + test 최초 평가 ──
    print("\n" + "=" * 90)
    print("=== 1) 실제 모델 학습 + test 평가 (이 test set 최초 사용) ===")
    print("=" * 90)
    result = train_once(y_train, SEED, "real", verbose=True)
    model = result["model"]
    print(f"best_epoch={result['best_epoch']}  best_val_loss={result['best_val_loss']:.6f}")

    real_preds, real_actuals = eval_on_test(model)
    real_rmse, real_mae = rmse_mae(real_preds, real_actuals)
    real_dir_acc = directional_accuracy(real_preds, real_actuals)
    n_test = len(y_test)

    print(f"\n[실제 결과] test(n={n_test}) RMSE={real_rmse:.6f}  MAE={real_mae:.6f}  "
          f"방향성정확도={real_dir_acc:.4f} ({real_dir_acc*100:.2f}%)")
    print(f"실제 분포: mean={real_actuals.mean():+.4%} std={real_actuals.std():.4%} "
          f"min={real_actuals.min():+.4%} max={real_actuals.max():+.4%}")
    print(f"예측 분포: mean={real_preds.mean():+.4%} std={real_preds.std():.4%} "
          f"min={real_preds.min():+.4%} max={real_preds.max():+.4%}")

    # ── 2) baseline 비교 ──
    print("\n" + "=" * 90)
    print("=== 2) Baseline 비교 ===")
    print("=" * 90)

    zero_preds = np.zeros_like(y_test)
    rw_rmse, rw_mae = rmse_mae(zero_preds, y_test)
    print(f"[회귀] Random Walk baseline(예측=0, '직전값 유지'와 동치): RMSE={rw_rmse:.6f}  MAE={rw_mae:.6f}")
    print(f"       실제 모델 대비: RMSE {real_rmse:.6f} ({'개선' if real_rmse < rw_rmse else '악화'} "
          f"{abs(real_rmse-rw_rmse)/rw_rmse*100:.2f}%)  "
          f"MAE {real_mae:.6f} ({'개선' if real_mae < rw_mae else '악화'} {abs(real_mae-rw_mae)/rw_mae*100:.2f}%)")

    base_label, base_acc = majority_baseline_accuracy(y_test)
    print(f"\n[방향성] 다수클래스 baseline: {base_label}  정확도={base_acc:.4f}  (실제 모델 {real_dir_acc:.4f})")

    n_correct = int(round(real_dir_acc * n_test))
    binom_result = stats.binomtest(n_correct, n_test, 0.5, alternative="two-sided")
    print(f"\n[방향성] 50% 동전던지기 대비: 실제 {n_correct}/{n_test} = {real_dir_acc:.4f}, "
          f"이항검정 p-value={binom_result.pvalue:.4f} "
          f"({'유의(p<0.05)' if binom_result.pvalue < 0.05 else '유의 아님(p>=0.05)'})")

    up_rate = float((y_train > 0).mean())
    prior_accs = []
    for i in range(N_RANDOM_PRIOR):
        rng = np.random.RandomState(5000 + i)
        pred_sign = rng.choice([1, -1], size=n_test, p=[up_rate, 1 - up_rate])
        prior_accs.append(float((pred_sign == np.sign(y_test)).mean()))
    prior_accs = np.array(prior_accs)
    print(f"[방향성] train 상승비율({up_rate:.4f}) 아는 무작위 예측기({N_RANDOM_PRIOR}회): "
          f"기대 정확도={prior_accs.mean():.4f}±{prior_accs.std():.4f}  (실제 모델 {real_dir_acc:.4f})")

    # ── 3) 레이블 셔플 테스트 (N=7, 절충) ──
    print("\n" + "=" * 90)
    print(f"=== 3) 레이블 셔플 테스트 (train target 순열 후 재학습, N={N_SHUFFLE}) ===")
    print("=" * 90)
    print("⚠️ 절충: 회귀 모델 재학습 비용 때문에 뉴스 트랙(N=30~200)보다 훨씬 적은 N=7 사용"
          "(seed_stability_check.py의 7시드 규모를 그대로 따름). 모델 초기화 시드는 실제 학습과"
          f" 동일하게 고정(seed={SEED}) — 순열 시드만 다르게 해 'y_train 순서 재배치' 효과만 분리."
          " N=7의 최소 0 초과 경험적 p-value는 1/8=0.125로 해상도가 거칠다 — 경계선 결과는"
          " 그대로 노출하고 과신하지 않는다.")

    shuffle_rmses, shuffle_maes, shuffle_dir_accs = [], [], []
    for i, perm_seed in enumerate(SHUFFLE_SEEDS):
        rng = np.random.RandomState(perm_seed)
        y_train_shuffled = y_train.copy()
        rng.shuffle(y_train_shuffled)

        r = train_once(y_train_shuffled, SEED, f"shuffle{i}", verbose=False)
        preds_s, _ = eval_on_test(r["model"])
        rmse_s, mae_s = rmse_mae(preds_s, y_test)
        dir_acc_s = directional_accuracy(preds_s, y_test)
        shuffle_rmses.append(rmse_s)
        shuffle_maes.append(mae_s)
        shuffle_dir_accs.append(dir_acc_s)
        print(f"  [{i+1}/{N_SHUFFLE}] perm_seed={perm_seed}: RMSE={rmse_s:.6f}  MAE={mae_s:.6f}  "
              f"방향성정확도={dir_acc_s:.4f}  best_epoch={r['best_epoch']}")

    shuffle_rmses = np.array(shuffle_rmses)
    shuffle_maes = np.array(shuffle_maes)
    shuffle_dir_accs = np.array(shuffle_dir_accs)

    print(f"\n셔플 분포(N={N_SHUFFLE}): RMSE {shuffle_rmses.mean():.6f}±{shuffle_rmses.std():.6f} "
          f"(min {shuffle_rmses.min():.6f} / max {shuffle_rmses.max():.6f})  |  실제 {real_rmse:.6f}")
    print(f"셔플 분포(N={N_SHUFFLE}): MAE  {shuffle_maes.mean():.6f}±{shuffle_maes.std():.6f} "
          f"(min {shuffle_maes.min():.6f} / max {shuffle_maes.max():.6f})  |  실제 {real_mae:.6f}")
    print(f"셔플 분포(N={N_SHUFFLE}): 방향성정확도 {shuffle_dir_accs.mean():.4f}±{shuffle_dir_accs.std():.4f} "
          f"(min {shuffle_dir_accs.min():.4f} / max {shuffle_dir_accs.max():.4f})  |  실제 {real_dir_acc:.4f}")

    n_shuffle_beats_real_rmse = int(np.sum(shuffle_rmses <= real_rmse))
    n_shuffle_beats_real_mae = int(np.sum(shuffle_maes <= real_mae))
    n_shuffle_beats_real_dir = int(np.sum(shuffle_dir_accs >= real_dir_acc))
    p_rmse = (n_shuffle_beats_real_rmse + 1) / (N_SHUFFLE + 1)
    p_mae = (n_shuffle_beats_real_mae + 1) / (N_SHUFFLE + 1)
    p_dir = (n_shuffle_beats_real_dir + 1) / (N_SHUFFLE + 1)
    print(f"\n노이즈(셔플)가 실제와 같거나 더 좋은 RMSE를 낸 횟수: {n_shuffle_beats_real_rmse}/{N_SHUFFLE}  (경험적 p={p_rmse:.4f})")
    print(f"노이즈(셔플)가 실제와 같거나 더 좋은 MAE를 낸 횟수:  {n_shuffle_beats_real_mae}/{N_SHUFFLE}  (경험적 p={p_mae:.4f})")
    print(f"노이즈(셔플)가 실제와 같거나 더 좋은 방향성정확도를 낸 횟수: {n_shuffle_beats_real_dir}/{N_SHUFFLE}  (경험적 p={p_dir:.4f})")

    # ── 4) 종합 비교표 + 최종 판정 ──
    print("\n" + "=" * 90)
    print("=== 종합 비교표 ===")
    print("=" * 90)
    print(f"{'지표':<28}{'실제 모델':>14}{'baseline':>16}{'판정':>16}")
    print(f"{'RMSE (vs RW baseline=0)':<28}{real_rmse:>14.6f}{rw_rmse:>16.6f}"
          f"{('모델 우수' if real_rmse < rw_rmse else 'RW가 우수'):>16}")
    print(f"{'MAE (vs RW baseline=0)':<28}{real_mae:>14.6f}{rw_mae:>16.6f}"
          f"{('모델 우수' if real_mae < rw_mae else 'RW가 우수'):>16}")
    print(f"{'방향성정확도 (vs 50%)':<28}{real_dir_acc:>14.4f}{0.5:>16.4f}"
          f"{('p<0.05' if binom_result.pvalue < 0.05 else 'p>=0.05'):>16}")
    print(f"{'방향성정확도 (vs 사전분포)':<28}{real_dir_acc:>14.4f}{prior_accs.mean():>16.4f}"
          f"{('모델 우수' if real_dir_acc > prior_accs.mean() else '무작위가 우수'):>16}")
    print(f"{'RMSE (vs 셔플 N=7)':<28}{real_rmse:>14.6f}{shuffle_rmses.mean():>16.6f}{f'p={p_rmse:.3f}':>16}")
    print(f"{'방향성정확도 (vs 셔플 N=7)':<28}{real_dir_acc:>14.4f}{shuffle_dir_accs.mean():>16.4f}{f'p={p_dir:.3f}':>16}")

    beats_rw = real_rmse < rw_rmse and real_mae < rw_mae
    beats_coinflip = binom_result.pvalue < 0.05 and real_dir_acc > 0.5
    beats_prior = real_dir_acc > prior_accs.mean() + 2 * prior_accs.std()
    beats_shuffle_reg = p_rmse < 0.05 and p_mae < 0.05
    beats_shuffle_dir = p_dir < 0.05

    print("\n" + "=" * 90)
    print("=== 최종 판정 ===")
    print("=" * 90)
    print(f"Random Walk baseline(RMSE/MAE 둘 다) 개선: {beats_rw}")
    print(f"50% 동전던지기 대비 유의(p<0.05): {beats_coinflip} (p={binom_result.pvalue:.4f})")
    print(f"train 상승비율 인지 무작위 예측기 대비 유의(+2σ 상회): {beats_prior}")
    print(f"레이블 셔플(N=7, 회귀 RMSE/MAE) 대비 유의(p<0.05, 해상도 한계 있음): {beats_shuffle_reg} (p_rmse={p_rmse:.3f}, p_mae={p_mae:.3f})")
    print(f"레이블 셔플(N=7, 방향성) 대비 유의(p<0.05, 해상도 한계 있음): {beats_shuffle_dir} (p={p_dir:.3f})")

    all_pass = beats_rw and beats_coinflip and beats_prior and beats_shuffle_reg and beats_shuffle_dir
    if all_pass:
        verdict = "test에서 val 결론이 뒤집힘 — 모든 엄격 기준 통과, 신호 존재 가능성(추가 정밀 검증 권장)"
    else:
        failed = [name for name, ok in [
            ("Random Walk", beats_rw), ("동전던지기", beats_coinflip),
            ("사전분포 무작위", beats_prior), ("셔플(회귀)", beats_shuffle_reg), ("셔플(방향성)", beats_shuffle_dir),
        ] if not ok]
        verdict = f"test에서도 val 결론(무작위 수준) 재현됨 — 신호 없음 (미통과 기준: {', '.join(failed)})"
    print(f"\n>>> {verdict}")


if __name__ == "__main__":
    main()
