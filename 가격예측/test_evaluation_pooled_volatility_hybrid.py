# test_evaluation_pooled_volatility_hybrid.py — Task T 하이브리드(GARCH+Transformer) 변동성
# 예측 검증 (설계+실행, 2026-09-06)
#
# 배경: 순수 pooled Transformer(14피처: V2 13개+recent_vol_ma20)는 GARCH(1,1)/SMA20 baseline을
# 8종목 평균 기준으로 이기지 못했다(test_evaluation_pooled_volatility.py, 2026-09-06). "GARCH를
# 처음부터 재학습하려다 실패한 것"이라는 가설로, GARCH의 조건부 σ 예측치 자체를 15번째 입력
# 피처로 추가한 하이브리드 모델을 시도한다 — GARCH가 이미 잡아낸 persistence 신호를 놓치지
# 않은 채, 그 위에 거시지표·종목 간 정보(임베딩)를 더해 GARCH를 넘어설 수 있는지 확인한다.
#
# 피처: build_merged_dataset_v2_volatility_hybrid()가 만드는 15개(기존 14개 + garch_sigma).
# garch_sigma는 garch_baseline.compute_full_period_sigma()(=fit_and_forecast_garch() 그대로
# 재사용, 코드 수정 없음)로 계산 — train만으로 추정한 고정 파라미터를 전체 구간에 재귀
# 적용한 1-step-ahead 값이라 leak-free다.
#
# 모델: PooledTransformerRegressor 그대로, n_features만 15로 바뀐다(생성자 인자로 자동 대응 —
# 구조 변경 없음, 하이퍼파라미터도 순수 버전과 완전히 동일하게 유지해 "피처 추가 효과"만
# 순수하게 관찰한다).
#
# 비교 대상 4개: 하이브리드 Transformer / 순수 Transformer(이전 실행 결과, 아래 PURE_TF_RESULTS
# 참고 — 동일 시드/설정이라 재실행하지 않고 그대로 인용) / GARCH(1,1) / SMA20.

import sys
from datetime import date

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
from 가격예측.garch_baseline import (
    compute_log_returns_pct,
    compute_sma_baseline,
    fit_and_forecast_garch,
    load_close_prices,
    rmse_mae,
)
from 가격예측.pooled_dataset import build_pooled_sequences, compute_global_split_dates
from 가격예측.sequence_dataset import FeatureScaler
from 가격예측.split_dataset import build_merged_dataset_v2_volatility, build_merged_dataset_v2_volatility_hybrid
from 가격예측.train_common import evaluate_pooled_predictions, train_pooled_transformer
from 가격예측.가격예측_통합모델 import (
    BATCH_SIZE, DIM_FEEDFORWARD, DROPOUT, D_MODEL, EMBEDDING_DIM, LOOKBACK, LR,
    MAX_EPOCHS, NHEAD, NUM_LAYERS, PATIENCE, SEED, SMOKE_EPOCHS, WEIGHT_DECAY,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

N_SHUFFLE = 7
SHUFFLE_SEEDS = [0, 1, 7, 123, 2024, 777, 999]

# 순수 pooled Transformer(14피처) 실제 실행 결과(2026-09-06, test_evaluation_pooled_volatility.py) —
# 동일 종목/동일 split/동일 하이퍼파라미터·시드라 재실행하지 않고 그대로 인용한다.
PURE_TF_RESULTS = {
    "005930": (3.813348, 2.514557), "005380": (3.387840, 2.169334),
    "051910": (3.035854, 2.044057), "105560": (1.830285, 1.217383),
    "207940": (1.911284, 1.212536), "035420": (2.727074, 1.753794),
    "034730": (3.591705, 2.392497), "015760": (2.625946, 1.766172),
}
PURE_TF_AVG = (2.865417, 1.883791)


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
    print(f"device: {device}")
    print(f"=== Pooled 하이브리드(GARCH+Transformer) 변동성 예측 평가 — {tickers} ===")

    # 전역 분할 경계 — non-hybrid 볼륨 데이터셋 기준(하이브리드는 GARCH σ 계산에 train_end가
    # 먼저 필요하므로, 그 경계 자체는 GARCH 의존 없는 이 참조로 구한다).
    ref_vol, _ = build_merged_dataset_v2_volatility(tickers[0], STOCK_INITIAL_LOAD_START, end_date)
    train_end, val_end = compute_global_split_dates(ref_vol.index)
    print(f"전역 분할 경계: train_end={train_end.date()}  val_end={val_end.date()}")

    def hybrid_build_fn(ticker, start_date, end_date):
        return build_merged_dataset_v2_volatility_hybrid(ticker, start_date, end_date, train_end)

    splits, feature_cols, ticker_to_id, (train_end2, val_end2) = build_pooled_sequences(
        tickers, STOCK_INITIAL_LOAD_START, end_date, lookback=LOOKBACK, build_fn=hybrid_build_fn,
    )
    assert train_end2 == train_end and val_end2 == val_end, "분할 경계가 어긋남 — GARCH와 비교 불가"
    id_to_ticker = {v: k for k, v in ticker_to_id.items()}
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
            len(feature_cols), LOOKBACK, len(tickers), EMBEDDING_DIM,
            D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT,
            BATCH_SIZE, LR, WEIGHT_DECAY, SMOKE_EPOCHS, MAX_EPOCHS, PATIENCE,
            device, seed=SEED, verbose=verbose, label=label,
        )

    # ── 1) 실제 학습 + test 평가 ──
    print("\n" + "=" * 100)
    print("=== 1) 실제 하이브리드 pooled Transformer 학습 + test 평가 ===")
    print("=" * 100)
    result = train_once(y_train, "hybrid-real", verbose=True)
    real_preds, real_actuals, real_tids = eval_loader(result["model"], X_test_s, y_test, tid_test, device)

    # ── 2) GARCH/SMA baseline 재계산(동일 test 구간) ──
    print("\n" + "=" * 100)
    print("=== 2) 종목별 GARCH(1,1) / SMA20 baseline 재계산 ===")
    print("=" * 100)
    garch_by_ticker = {}
    for ticker in tickers:
        close = load_close_prices(ticker, STOCK_INITIAL_LOAD_START, end_date)
        returns = compute_log_returns_pct(close)
        test_returns = returns[returns.index > val_end]
        sigma_pred, _ = fit_and_forecast_garch(returns, train_end, test_returns.index[0])
        sma_pred = compute_sma_baseline(returns).loc[sigma_pred.index]
        valid = sigma_pred.index.intersection(sma_pred.dropna().index)
        realized = test_returns.abs().loc[valid]
        g_rmse, g_mae = rmse_mae(sigma_pred.loc[valid].values, realized.values)
        s_rmse, s_mae = rmse_mae(sma_pred.loc[valid].values, realized.values)
        garch_by_ticker[ticker] = {"n": len(valid), "garch_rmse": g_rmse, "garch_mae": g_mae,
                                    "sma_rmse": s_rmse, "sma_mae": s_mae}
        print(f"  {ticker}: n={len(valid)}  GARCH RMSE={g_rmse:.6f} MAE={g_mae:.6f}  |  "
              f"SMA20 RMSE={s_rmse:.6f} MAE={s_mae:.6f}")

    # ── 3) 4-way 종합 비교표 ──
    def per_ticker_rmse_mae(preds, actuals, tids):
        out = {}
        for ticker, tid in ticker_to_id.items():
            mask = tids == tid
            out[ticker] = rmse_mae(preds[mask], actuals[mask])
        return out, rmse_mae(preds, actuals)

    hybrid_per_ticker, hybrid_overall = per_ticker_rmse_mae(real_preds, real_actuals, real_tids)

    print("\n" + "=" * 130)
    print("=== 3) 종합 비교표 — 하이브리드TF vs 순수TF(이전 실행) vs GARCH vs SMA20 ===")
    print("=" * 130)
    header = (f"{'구분':<12}{'하이브리드RMSE':>14}{'하이브리드MAE':>13}{'순수TF RMSE':>12}{'순수TF MAE':>11}"
              f"{'GARCH RMSE':>12}{'GARCH MAE':>11}{'SMA RMSE':>10}{'SMA MAE':>9}")
    print(header)
    for ticker in tickers:
        h_rmse, h_mae = hybrid_per_ticker[ticker]
        p_rmse_v, p_mae_v = PURE_TF_RESULTS[ticker]
        g = garch_by_ticker[ticker]
        print(f"{ticker:<12}{h_rmse:>14.6f}{h_mae:>13.6f}{p_rmse_v:>12.6f}{p_mae_v:>11.6f}"
              f"{g['garch_rmse']:>12.6f}{g['garch_mae']:>11.6f}{g['sma_rmse']:>10.6f}{g['sma_mae']:>9.6f}")

    avg_garch_rmse = float(np.mean([garch_by_ticker[t]["garch_rmse"] for t in tickers]))
    avg_garch_mae = float(np.mean([garch_by_ticker[t]["garch_mae"] for t in tickers]))
    avg_sma_rmse = float(np.mean([garch_by_ticker[t]["sma_rmse"] for t in tickers]))
    avg_sma_mae = float(np.mean([garch_by_ticker[t]["sma_mae"] for t in tickers]))
    avg_hybrid_rmse = float(np.mean([hybrid_per_ticker[t][0] for t in tickers]))
    avg_hybrid_mae = float(np.mean([hybrid_per_ticker[t][1] for t in tickers]))

    print(f"{'평균(주판정)':<12}{avg_hybrid_rmse:>14.6f}{avg_hybrid_mae:>13.6f}{PURE_TF_AVG[0]:>12.6f}"
          f"{PURE_TF_AVG[1]:>11.6f}{avg_garch_rmse:>12.6f}{avg_garch_mae:>11.6f}{avg_sma_rmse:>10.6f}{avg_sma_mae:>9.6f}")
    print(f"(참고: 전체 test 시퀀스 pooling 기준 하이브리드 RMSE/MAE = {hybrid_overall[0]:.6f} / {hybrid_overall[1]:.6f})")

    # ── 4) 순수 TF 대비 개선 여부 (피처 추가 자체의 효과) ──
    improved_vs_pure = avg_hybrid_rmse < PURE_TF_AVG[0] and avg_hybrid_mae < PURE_TF_AVG[1]
    print(f"\n[핵심 확인 1] GARCH σ 피처 추가가 순수 Transformer보다 개선됐는가: {improved_vs_pure}")
    print(f"  RMSE: {PURE_TF_AVG[0]:.6f} -> {avg_hybrid_rmse:.6f} "
          f"({'개선' if avg_hybrid_rmse < PURE_TF_AVG[0] else '악화'} "
          f"{abs(avg_hybrid_rmse-PURE_TF_AVG[0])/PURE_TF_AVG[0]*100:.2f}%)")
    print(f"  MAE : {PURE_TF_AVG[1]:.6f} -> {avg_hybrid_mae:.6f} "
          f"({'개선' if avg_hybrid_mae < PURE_TF_AVG[1] else '악화'} "
          f"{abs(avg_hybrid_mae-PURE_TF_AVG[1])/PURE_TF_AVG[1]*100:.2f}%)")

    beats_garch = avg_hybrid_rmse < avg_garch_rmse and avg_hybrid_mae < avg_garch_mae
    beats_sma = avg_hybrid_rmse < avg_sma_rmse and avg_hybrid_mae < avg_sma_mae
    print(f"\n[핵심 확인 2] 하이브리드 Transformer가 GARCH(1,1)를 넘어서는가: {beats_garch}")
    print(f"[참고] 하이브리드 Transformer가 SMA20을 넘어서는가: {beats_sma}")

    # ── 5) 레이블 셔플 테스트 (N=7) ──
    print("\n" + "=" * 100)
    print(f"=== 5) 레이블 셔플 테스트 (pooled train target 전체 순열 후 재학습, N={N_SHUFFLE}) ===")
    print("=" * 100)
    shuffle_overall_rmse, shuffle_overall_mae = [], []
    for i, perm_seed in enumerate(SHUFFLE_SEEDS):
        rng = np.random.RandomState(perm_seed)
        y_train_shuffled = y_train.copy()
        rng.shuffle(y_train_shuffled)
        r = train_once(y_train_shuffled, f"hybrid-shuffle{i}", verbose=False)
        preds_s, actuals_s, _ = eval_loader(r["model"], X_test_s, y_test, tid_test, device)
        rmse_s, mae_s = rmse_mae(preds_s, actuals_s)
        shuffle_overall_rmse.append(rmse_s)
        shuffle_overall_mae.append(mae_s)
        print(f"  [{i+1}/{N_SHUFFLE}] perm_seed={perm_seed}: 전체 RMSE={rmse_s:.6f}  MAE={mae_s:.6f}  best_epoch={r['best_epoch']}")

    shuffle_overall_rmse = np.array(shuffle_overall_rmse)
    shuffle_overall_mae = np.array(shuffle_overall_mae)

    def _p(real_val, shuffle_arr):
        beat = int(np.sum(shuffle_arr <= real_val))
        return beat, (beat + 1) / (N_SHUFFLE + 1)

    n_beat_rmse, p_rmse = _p(hybrid_overall[0], shuffle_overall_rmse)
    n_beat_mae, p_mae = _p(hybrid_overall[1], shuffle_overall_mae)
    print(f"\n[전체] 셔플 분포: RMSE {shuffle_overall_rmse.mean():.6f}±{shuffle_overall_rmse.std():.6f} | 실제 {hybrid_overall[0]:.6f}")
    print(f"[전체] 셔플 분포: MAE  {shuffle_overall_mae.mean():.6f}±{shuffle_overall_mae.std():.6f} | 실제 {hybrid_overall[1]:.6f}")
    print(f"[전체] 노이즈가 실제와 같거나 더 좋은 RMSE: {n_beat_rmse}/{N_SHUFFLE} (p={p_rmse:.4f})")
    print(f"[전체] 노이즈가 실제와 같거나 더 좋은 MAE:  {n_beat_mae}/{N_SHUFFLE} (p={p_mae:.4f})")
    beats_shuffle = p_rmse < 0.05 and p_mae < 0.05

    # ── 6) 최종 판정 ──
    print("\n" + "=" * 100)
    print("=== 최종 판정 ===")
    print("=" * 100)
    print(f"순수 Transformer 대비 개선: {improved_vs_pure}")
    print(f"GARCH(1,1) 대비 개선: {beats_garch}")
    print(f"SMA20 대비 개선: {beats_sma}")
    print(f"셔플(N=7) 대비 유의: {beats_shuffle} (p_rmse={p_rmse:.3f}, p_mae={p_mae:.3f})")

    if beats_garch:
        final = "하이브리드가 GARCH를 넘어섰다"
    elif improved_vs_pure:
        final = "GARCH는 못 넘었지만 순수 Transformer보다는 개선됐다"
    else:
        final = "순수 Transformer보다도 개선되지 않았다 — GARCH 피처 추가 자체가 효과 없음"
    print(f"\n>>> {final}")
    if beats_garch and not beats_shuffle:
        print("⚠️ GARCH를 이긴 것으로 보이나 셔플 검증을 통과하지 못했습니다 — 노이즈로도 재현 가능한 "
              "결과일 수 있어 '넘어섰다'고 단정하지 않습니다.")

    print("\n[종목별 참고]")
    for ticker in tickers:
        h_rmse, h_mae = hybrid_per_ticker[ticker]
        g = garch_by_ticker[ticker]
        p_rmse_v, p_mae_v = PURE_TF_RESULTS[ticker]
        vs_garch = "GARCH 이김" if (h_rmse < g["garch_rmse"] and h_mae < g["garch_mae"]) else "GARCH 못이김"
        vs_pure = "순수TF 대비 개선" if (h_rmse < p_rmse_v and h_mae < p_mae_v) else "순수TF 대비 악화/동일"
        print(f"  {ticker}: {vs_garch} / {vs_pure}")


if __name__ == "__main__":
    main()
