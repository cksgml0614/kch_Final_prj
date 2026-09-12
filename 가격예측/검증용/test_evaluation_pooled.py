# test_evaluation_pooled.py — Task T 다종목 pooled 모델 검증 계획(설계+구현, 2026-09-06)
#
# ⚠️ 이 스크립트는 요구사항 6("검증 계획을 미리 설계") 대응 — 코드는 완성했지만 실제 8종목
# pooled 학습·검증은 아직 실행하지 않았다(사람 승인 대기). test_evaluation.py(단일 종목 V2)와
# 동일한 엄격 기준(Random Walk baseline, 50% 동전던지기, train 상승비율 인지 무작위 예측기,
# 레이블 셔플 재학습)을 그대로 적용하되, 이번엔 종목별 + 8종목 전체 평균 두 층위로 보고한다.
#
# 셔플 검증 절충(단일 종목 test_evaluation.py와 동일한 이유로 N=7): pooled train 전체(8종목
# 합쳐진 y)를 한 번에 순열해 재학습 — 종목 경계를 무시하고 전체를 뒤섞는다. "이 pooled 모델이
# 8종목에 걸친 어떤 cross-sectional 관계라도 학습했는가"를 묻는 것이 목적이라, 종목별로
# 따로따로 순열하는 것보다 전체를 한 번에 섞는 쪽이 더 엄격한(더 깨끗한) 귀무가설이다.
# 셔플마다 pooled 모델 하나만 재학습하면 되므로 8종목이어도 재학습 횟수 자체는 단일 종목과
# 동일(N=7) — 종목별 세부 결과는 그 셔플 예측을 ticker_id로 나눠서 얻는다(추가 재학습 비용 없음).
#
# 실행: python -m 가격예측.test_evaluation_pooled [--tickers ...] [--start ...] [--end ...]

import argparse
import sys
from datetime import date

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset

from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
from 가격예측.pooled_dataset import build_pooled_sequences
from 가격예측.sequence_dataset import FeatureScaler
from 가격예측.train_common import evaluate_pooled_predictions, train_pooled_transformer
from 가격예측.검증용.가격예측_통합모델 import (
    BATCH_SIZE, DIM_FEEDFORWARD, DROPOUT, D_MODEL, EMBEDDING_DIM, LOOKBACK, LR,
    MAX_EPOCHS, NHEAD, NUM_LAYERS, PATIENCE, SEED, SMOKE_EPOCHS, WEIGHT_DECAY,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

N_SHUFFLE = 7
SHUFFLE_SEEDS = [0, 1, 7, 123, 2024, 777, 999]
N_RANDOM_PRIOR = 200


def rmse_mae(preds, actuals):
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    mae = float(np.mean(np.abs(preds - actuals)))
    return rmse, mae


def dir_acc(preds, actuals):
    return float((np.sign(preds) == np.sign(actuals)).mean()) if len(actuals) else float("nan")


def compute_metrics_table(preds, actuals, ticker_ids, id_to_ticker, train_actuals_by_ticker=None):
    """종목별 + 전체(overall) RMSE/MAE/방향성정확도 + baseline 비교치를 한 번에 계산한다
    (요구사항 6의 핵심 함수 — 종목별 표와 전체 평균 표를 이 함수 하나로 뽑는다).

    train_actuals_by_ticker: {ticker: y_train_ticker} — train 상승비율 인지 무작위 예측기의
    종목별 기준값 계산용(옵션, 없으면 그 비교는 생략)."""
    def _block(p, a, up_rate_for_prior=None):
        n = len(a)
        if n == 0:
            return None
        rmse, mae = rmse_mae(p, a)
        acc = dir_acc(p, a)
        zero_rmse, zero_mae = rmse_mae(np.zeros_like(a), a)
        n_correct = int(round(acc * n))
        binom_p = stats.binomtest(n_correct, n, 0.5, alternative="two-sided").pvalue if n > 0 else float("nan")

        prior_mean = prior_std = float("nan")
        if up_rate_for_prior is not None:
            prior_accs = []
            for i in range(N_RANDOM_PRIOR):
                rng = np.random.RandomState(5000 + i)
                pred_sign = rng.choice([1, -1], size=n, p=[up_rate_for_prior, 1 - up_rate_for_prior])
                prior_accs.append(float((pred_sign == np.sign(a)).mean()))
            prior_mean, prior_std = float(np.mean(prior_accs)), float(np.std(prior_accs))

        return {
            "n": n, "rmse": rmse, "mae": mae, "dir_acc": acc,
            "rw_rmse": zero_rmse, "rw_mae": zero_mae,
            "beats_rw": rmse < zero_rmse and mae < zero_mae,
            "binom_p": binom_p, "beats_coinflip": binom_p < 0.05 and acc > 0.5,
            "prior_mean": prior_mean, "prior_std": prior_std,
            "beats_prior": (not np.isnan(prior_mean)) and acc > prior_mean + 2 * prior_std,
        }

    overall_up_rate = None
    if train_actuals_by_ticker is not None:
        all_train = np.concatenate(list(train_actuals_by_ticker.values()))
        overall_up_rate = float((all_train > 0).mean())

    overall = _block(preds, actuals, overall_up_rate)

    per_ticker = {}
    for tid in sorted(set(ticker_ids.tolist())):
        ticker = id_to_ticker[tid]
        mask = ticker_ids == tid
        up_rate = None
        if train_actuals_by_ticker is not None and ticker in train_actuals_by_ticker:
            y_tr = train_actuals_by_ticker[ticker]
            up_rate = float((y_tr > 0).mean()) if len(y_tr) else None
        per_ticker[ticker] = _block(preds[mask], actuals[mask], up_rate)

    return {"overall": overall, "per_ticker": per_ticker}


def print_metrics_table(metrics, title):
    print(f"\n=== {title} ===")
    header = (f"{'구분':<12}{'n':>6}{'RMSE':>10}{'MAE':>10}{'방향성':>9}"
              f"{'RW대비':>8}{'동전대비':>10}{'사전분포대비':>12}")
    print(header)
    o = metrics["overall"]
    coinflip_str = f"p={o['binom_p']:.3f}"
    print(f"{'전체(8종목)':<12}{o['n']:>6}{o['rmse']:>10.6f}{o['mae']:>10.6f}{o['dir_acc']:>9.4f}"
          f"{str(o['beats_rw']):>8}{coinflip_str:>10}{str(o['beats_prior']):>12}")
    for ticker, m in metrics["per_ticker"].items():
        if m is None:
            print(f"{ticker:<12}{'(표본없음)':>6}")
            continue
        coinflip_str = f"p={m['binom_p']:.3f}"
        print(f"{ticker:<12}{m['n']:>6}{m['rmse']:>10.6f}{m['mae']:>10.6f}{m['dir_acc']:>9.4f}"
              f"{str(m['beats_rw']):>8}{coinflip_str:>10}{str(m['beats_prior']):>12}")


def main():
    parser = argparse.ArgumentParser(description="Task T pooled 모델 test 평가 — 종목별 + 전체 평균")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--start", type=str, default=STOCK_INITIAL_LOAD_START)
    parser.add_argument("--end", type=str, default=None)
    args = parser.parse_args()

    tickers = args.tickers if args.tickers else ACTIVE_TICKERS
    end_date = args.end if args.end else date.today().isoformat()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"=== Pooled test 평가(최초 실행) — {tickers}, {args.start}~{end_date} ===")

    splits, feature_cols, ticker_to_id, (train_end, val_end) = build_pooled_sequences(
        tickers, args.start, end_date, lookback=LOOKBACK
    )
    id_to_ticker = {v: k for k, v in ticker_to_id.items()}
    X_train, y_train, tid_train, d_train, tname_train = splits["train"]
    X_val, y_val, tid_val, d_val, _ = splits["val"]
    X_test, y_test, tid_test, d_test, _ = splits["test"]
    print(f"전역 분할 경계: train_end={train_end.date()}  val_end={val_end.date()}")
    print(f"시퀀스 shape: X_train {X_train.shape}  X_val {X_val.shape}  X_test {X_test.shape}")

    train_actuals_by_ticker = {
        ticker: y_train[tid_train == tid] for ticker, tid in ticker_to_id.items()
    }

    scaler = FeatureScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    def eval_on_test(model):
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_test_s), torch.from_numpy(y_test), torch.from_numpy(tid_test)),
            batch_size=BATCH_SIZE, shuffle=False,
        )
        return evaluate_pooled_predictions(model, loader, device)

    def train_once(y_train_arr, label, verbose):
        return train_pooled_transformer(
            X_train_s, y_train_arr, tid_train, X_val_s, y_val, tid_val,
            len(feature_cols), LOOKBACK, len(tickers), EMBEDDING_DIM,
            D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT,
            BATCH_SIZE, LR, WEIGHT_DECAY, SMOKE_EPOCHS, MAX_EPOCHS, PATIENCE,
            device, seed=SEED, verbose=verbose, label=label,
        )

    # ── 1) 실제 학습 + test 최초 평가 ──
    print("\n" + "=" * 100)
    print("=== 1) 실제 pooled 모델 학습 + test 평가 (이 test set 최초 사용) ===")
    print("=" * 100)
    result = train_once(y_train, "real", verbose=True)
    real_preds, real_actuals, real_tids = eval_on_test(result["model"])
    real_metrics = compute_metrics_table(real_preds, real_actuals, real_tids, id_to_ticker, train_actuals_by_ticker)
    print_metrics_table(real_metrics, "실제 결과 — 종목별 + 전체")

    # ── 2) 레이블 셔플 테스트 (N=7, pooled train 전체 순열) ──
    print("\n" + "=" * 100)
    print(f"=== 2) 레이블 셔플 테스트 (pooled train target 전체 순열 후 재학습, N={N_SHUFFLE}) ===")
    print("=" * 100)
    print("⚠️ 절충: 단일 종목 test_evaluation.py와 동일하게 N=7(회귀 재학습 비용 감안). 8종목을 "
          "합친 pooled train 전체를 한 번에 순열한다(종목 경계 무시) — 종목별 세부 결과는 같은 "
          "셔플 예측을 ticker_id로 나눠 얻으므로 추가 재학습 비용이 들지 않는다. N=7의 최소 "
          "0 초과 경험적 p-value는 1/8=0.125로 해상도가 거칠다 — 결과 해석 시 과신 금지.")

    shuffle_overall_rmse, shuffle_overall_mae, shuffle_overall_dir = [], [], []
    shuffle_per_ticker = {t: {"rmse": [], "mae": [], "dir_acc": []} for t in tickers}

    for i, perm_seed in enumerate(SHUFFLE_SEEDS):
        rng = np.random.RandomState(perm_seed)
        y_train_shuffled = y_train.copy()
        rng.shuffle(y_train_shuffled)

        r = train_once(y_train_shuffled, f"shuffle{i}", verbose=False)
        preds_s, actuals_s, tids_s = eval_on_test(r["model"])
        rmse_s, mae_s = rmse_mae(preds_s, actuals_s)
        dacc_s = dir_acc(preds_s, actuals_s)
        shuffle_overall_rmse.append(rmse_s)
        shuffle_overall_mae.append(mae_s)
        shuffle_overall_dir.append(dacc_s)

        for ticker, tid in ticker_to_id.items():
            mask = tids_s == tid
            r_rmse, r_mae = rmse_mae(preds_s[mask], actuals_s[mask])
            shuffle_per_ticker[ticker]["rmse"].append(r_rmse)
            shuffle_per_ticker[ticker]["mae"].append(r_mae)
            shuffle_per_ticker[ticker]["dir_acc"].append(dir_acc(preds_s[mask], actuals_s[mask]))

        print(f"  [{i+1}/{N_SHUFFLE}] perm_seed={perm_seed}: 전체 RMSE={rmse_s:.6f}  MAE={mae_s:.6f}  "
              f"방향성정확도={dacc_s:.4f}  best_epoch={r['best_epoch']}")

    shuffle_overall_rmse = np.array(shuffle_overall_rmse)
    shuffle_overall_mae = np.array(shuffle_overall_mae)
    shuffle_overall_dir = np.array(shuffle_overall_dir)

    def _p(real_val, shuffle_arr, higher_is_better):
        if higher_is_better:
            beat = int(np.sum(shuffle_arr >= real_val))
        else:
            beat = int(np.sum(shuffle_arr <= real_val))
        return beat, (beat + 1) / (N_SHUFFLE + 1)

    real_overall = real_metrics["overall"]
    n_beat_rmse, p_rmse = _p(real_overall["rmse"], shuffle_overall_rmse, higher_is_better=False)
    n_beat_mae, p_mae = _p(real_overall["mae"], shuffle_overall_mae, higher_is_better=False)
    n_beat_dir, p_dir = _p(real_overall["dir_acc"], shuffle_overall_dir, higher_is_better=True)

    print(f"\n[전체] 셔플 분포: RMSE {shuffle_overall_rmse.mean():.6f}±{shuffle_overall_rmse.std():.6f} | 실제 {real_overall['rmse']:.6f}")
    print(f"[전체] 셔플 분포: MAE  {shuffle_overall_mae.mean():.6f}±{shuffle_overall_mae.std():.6f} | 실제 {real_overall['mae']:.6f}")
    print(f"[전체] 셔플 분포: 방향성정확도 {shuffle_overall_dir.mean():.4f}±{shuffle_overall_dir.std():.4f} | 실제 {real_overall['dir_acc']:.4f}")
    print(f"[전체] 노이즈가 실제와 같거나 더 좋은 RMSE: {n_beat_rmse}/{N_SHUFFLE} (p={p_rmse:.4f})")
    print(f"[전체] 노이즈가 실제와 같거나 더 좋은 MAE:  {n_beat_mae}/{N_SHUFFLE} (p={p_mae:.4f})")
    print(f"[전체] 노이즈가 실제와 같거나 더 좋은 방향성정확도: {n_beat_dir}/{N_SHUFFLE} (p={p_dir:.4f})")

    print("\n=== 종목별 셔플 검증 요약 (동일 7회 셔플 예측을 종목별로 분리, 추가 재학습 없음) ===")
    header = f"{'ticker':<12}{'실제RMSE':>12}{'셔플RMSE':>14}{'p':>8}{'실제방향성':>10}{'셔플방향성':>14}{'p':>8}"
    print(header)
    per_ticker_verdicts = {}
    for ticker in tickers:
        m = real_metrics["per_ticker"][ticker]
        s_rmse = np.array(shuffle_per_ticker[ticker]["rmse"])
        s_dir = np.array(shuffle_per_ticker[ticker]["dir_acc"])
        _, p_r = _p(m["rmse"], s_rmse, higher_is_better=False)
        _, p_d = _p(m["dir_acc"], s_dir, higher_is_better=True)
        beats_shuffle = p_r < 0.05 and p_d < 0.05
        per_ticker_verdicts[ticker] = {"p_rmse": p_r, "p_dir": p_d, "beats_shuffle": beats_shuffle}
        print(f"{ticker:<12}{m['rmse']:>12.6f}{s_rmse.mean():>14.6f}{p_r:>8.3f}"
              f"{m['dir_acc']:>10.4f}{s_dir.mean():>14.4f}{p_d:>8.3f}")

    # ── 3) 최종 판정 (전체 + 종목별) ──
    print("\n" + "=" * 100)
    print("=== 최종 판정 ===")
    print("=" * 100)
    overall_beats_shuffle = p_rmse < 0.05 and p_mae < 0.05 and p_dir < 0.05
    overall_all_pass = real_overall["beats_rw"] and real_overall["beats_coinflip"] and real_overall["beats_prior"] and overall_beats_shuffle
    print(f"[전체 8종목] Random Walk 개선: {real_overall['beats_rw']}  |  동전던지기 유의: {real_overall['beats_coinflip']}  |  "
          f"사전분포 유의: {real_overall['beats_prior']}  |  셔플 유의: {overall_beats_shuffle}")
    print(f"[전체 8종목] >>> {'신호 존재 가능성(모든 기준 통과)' if overall_all_pass else '신호 없음(무작위 수준) — 단일 종목 test와 동일 결론'}")

    print("\n[종목별]")
    any_ticker_signal = False
    for ticker in tickers:
        m = real_metrics["per_ticker"][ticker]
        v = per_ticker_verdicts[ticker]
        ticker_all_pass = m["beats_rw"] and m["beats_coinflip"] and m["beats_prior"] and v["beats_shuffle"]
        if ticker_all_pass:
            any_ticker_signal = True
        print(f"  {ticker}: RW={m['beats_rw']} 동전={m['beats_coinflip']} 사전분포={m['beats_prior']} "
              f"셔플={v['beats_shuffle']} -> {'신호 가능성' if ticker_all_pass else '무작위 수준'}")

    if any_ticker_signal:
        print("\n⚠️ 일부 종목에서 모든 기준을 통과하는 결과가 나왔습니다 — 다중 비교(8종목 x 4기준)로 "
              "인한 우연적 통과 가능성을 배제할 수 없으니 추가 검증(예: 해당 종목만 별도 셔플 N 확대) 없이는 "
              "'신호 발견'으로 단정하지 말 것.")
    else:
        print("\n8종목 전체와 개별 종목 어디에서도 모든 기준을 통과하는 조합이 없습니다.")


if __name__ == "__main__":
    main()
