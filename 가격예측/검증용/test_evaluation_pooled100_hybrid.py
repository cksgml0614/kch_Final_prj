# test_evaluation_pooled100_hybrid.py — 50→100종목(시가총액 상위, 2026-09-07 확장) pooled
# 하이브리드(GARCH σ 포함 15피처) Transformer 재학습·평가 (일회성).
#
# 목적: 어제(2026-09-06) 8→50종목 확장에서 분산 붕괴 완화 효과가 확인됐고, 오늘(2026-09-07)
# 종목평균 제거(demean) 검증에서도 baseline 대비 진짜 시계열 예측력이 살아있음이 확인됐다.
# 표본을 50→100으로 더 늘려도 이 두 효과(분산 붕괴 완화, demean 후에도 baseline 우위)가
# 이어지는지 "방향 확인"이 목적이다 — 완벽한 통계적 확정이 아니라 셔플 N=5로 빠르게 본다
# (p-value 해상도 최소 1/6, 그 대신 격차가 표준편차 단위로 여전히 큰지를 함께 본다).
#
# 50종목 버전(test_evaluation_pooled50_hybrid.py) + 종목평균 제거 검증(test_evaluation_
# pooled50_decompose.py) 두 스크립트를 하나로 합쳤다 — baseline(GARCH/SMA20/Parkinson-SMA20)
# 계산이 종목당 GARCH fit을 포함해 비용이 크므로, 별도 스크립트로 나누면 baseline을 두 번
# 계산하게 돼 비효율적이다.
#
# [임베딩 차원 재검토, 100종목]: 50종목 확장 때 EMBEDDING_DIM을 8->16으로 올리며 "d_model=32
# 대비 embedding이 절반을 넘지 않도록"이라는 원칙을 세웠다(16 = d_model의 정확히 절반). 100종목은
# 50종목 대비 카테고리 수가 2배로만 늘어난다(8종목→50종목의 6.25배 증가와는 규모가 다르다) —
# nn.Embedding은 연속 공간이라 100개를 구분하는 데 16차원이 정보이론적으로 부족한 것은 아니지만
# (16차원 연속 벡터는 100개보다 훨씬 많은 점을 선형분리 가능하게 배치할 수 있음), 카테고리가
# 늘어난 만큼 학습 중 각 종목이 받는 그래디언트 신호가 더 희석되므로 표현 여유를 약간 더 준다 —
# d_model은 그대로 32로 유지(50종목 결과와 아키텍처를 최대한 동일하게 유지해 비교 가능하게 함)
# 하고 EMBEDDING_DIM만 16->20으로 소폭 증가시켰다(20/32=62.5%, "절반을 넘지 않는다" 원칙은
# 깨지지만 d_model을 함께 키우면 50종목 결과와 아키텍처가 달라져 비교가 어려워지므로 이쪽을
# 선택). 이 값 자체를 정밀 튜닝하지는 않았다 — 오늘 목적은 "방향 확인"이지 하이퍼파라미터
# 최적화가 아니다.

import sys
from datetime import date

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
from 가격예측.garch_baseline import (
    compute_log_returns_pct, compute_sma_baseline,
    load_close_prices, rmse_mae,
)
from 가격예측.검증용.garch_baseline_check import fit_and_forecast_garch
from 가격예측.parkinson_baseline import compute_parkinson_vol_pct, load_high_low, PARK_WINDOW
from 가격예측.pooled_dataset import build_pooled_sequences, compute_global_split_dates
from 가격예측.sequence_dataset import FeatureScaler
from 가격예측.split_dataset import build_merged_dataset_v2, build_merged_dataset_v2_volatility_hybrid
from 가격예측.train_common import evaluate_pooled_predictions, train_pooled_transformer
from 가격예측.검증용.가격예측_통합모델 import (
    BATCH_SIZE, DIM_FEEDFORWARD, DROPOUT, D_MODEL, LOOKBACK, LR,
    MAX_EPOCHS, NHEAD, NUM_LAYERS, PATIENCE, SEED, SMOKE_EPOCHS, WEIGHT_DECAY,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EMBEDDING_DIM_100 = 20  # 위 [임베딩 차원 재검토, 100종목] 참고 (50종목 시절 16에서 소폭 증가)

N_SHUFFLE = 5
SHUFFLE_SEEDS = [0, 1, 7, 123, 2024]  # 50종목 N=7/N=30 셔플과 앞 3개 시드 공유(재현성 참고용)


def eval_loader(model, X, y, tid, device):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(tid)),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    return evaluate_pooled_predictions(model, loader, device)


def describe(preds):
    return {"mean": float(np.mean(preds)), "std": float(np.std(preds)),
            "min": float(np.min(preds)), "max": float(np.max(preds))}


def pearson_r(pred, actual):
    if len(pred) < 2 or np.std(pred) == 0 or np.std(actual) == 0:
        return float("nan")
    return float(np.corrcoef(pred, actual)[0, 1])


def demean_by_ticker(values, ticker_labels):
    out = np.empty_like(values, dtype=np.float64)
    for t in np.unique(ticker_labels):
        mask = ticker_labels == t
        out[mask] = values[mask] - values[mask].mean()
    return out


def main():
    tickers = ACTIVE_TICKERS
    end_date = date.today().isoformat()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    print(f"=== Pooled 하이브리드(GARCH+Transformer) 변동성 예측 평가 — {len(tickers)}종목 ===", flush=True)
    print(f"EMBEDDING_DIM = {EMBEDDING_DIM_100} (50종목 시절 16에서 소폭 확대, d_model={D_MODEL} 유지)", flush=True)

    # ── 0) 전역 분할 경계 + pooled 시퀀스 구성 ──
    ref_vol, _ = build_merged_dataset_v2(tickers[0], STOCK_INITIAL_LOAD_START, end_date)
    train_end, val_end = compute_global_split_dates(ref_vol.index)
    print(f"전역 분할 경계: train_end={train_end.date()}  val_end={val_end.date()}", flush=True)

    def hybrid_build_fn(ticker, s, e):
        return build_merged_dataset_v2_volatility_hybrid(ticker, s, e, train_end)

    splits, feature_cols, ticker_to_id, (train_end2, val_end2) = build_pooled_sequences(
        tickers, STOCK_INITIAL_LOAD_START, end_date, lookback=LOOKBACK, build_fn=hybrid_build_fn,
    )
    assert train_end2 == train_end and val_end2 == val_end
    X_train, y_train, tid_train, d_train, _ = splits["train"]
    X_val, y_val, tid_val, d_val, _ = splits["val"]
    X_test, y_test, tid_test, d_test, tname_test = splits["test"]
    print(f"feature_cols({len(feature_cols)}개): {feature_cols}", flush=True)
    print(f"시퀀스 shape: X_train {X_train.shape}  X_val {X_val.shape}  X_test {X_test.shape}", flush=True)

    n_train_by_ticker = {t: int((tid_train == tid).sum()) for t, tid in ticker_to_id.items()}
    zero_train = [t for t, n in n_train_by_ticker.items() if n == 0]
    if zero_train:
        print(f"⚠️ train 시퀀스 0개인 종목 {len(zero_train)}개(임베딩이 학습되지 않음): {zero_train}", flush=True)
    else:
        print("모든 종목이 train 시퀀스 1개 이상 보유 (임베딩 학습 가능)", flush=True)

    scaler = FeatureScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    def train_once(y_train_arr, label, verbose):
        return train_pooled_transformer(
            X_train_s, y_train_arr, tid_train, X_val_s, y_val, tid_val,
            len(feature_cols), LOOKBACK, len(tickers), EMBEDDING_DIM_100,
            D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT,
            BATCH_SIZE, LR, WEIGHT_DECAY, SMOKE_EPOCHS, MAX_EPOCHS, PATIENCE,
            device, seed=SEED, verbose=verbose, label=label,
        )

    # ── 1) 실제 하이브리드 pooled Transformer(100종목) 학습 + test 평가 ──
    print("\n" + "=" * 100)
    print("=== 1) 실제 하이브리드 pooled Transformer(100종목) 학습 + test 평가 ===")
    print("=" * 100)
    result = train_once(y_train, "hybrid100-real", verbose=True)
    hybrid_preds, hybrid_actuals, hybrid_tids = eval_loader(result["model"], X_test_s, y_test, tid_test, device)
    print(f"하이브리드 학습 완료: best_epoch={result['best_epoch']}", flush=True)

    print("\n" + "=" * 100)
    print("=== [핵심] test 예측 분포 — 8종목(mean=1.8383, std=0.2608, 9.5%) / "
          "50종목(30.8%)과 비교 ===")
    print("=" * 100)
    dist = describe(hybrid_preds)
    print(f"100종목 test 예측 분포: {dist}")
    print(f"실제 test target 분포: mean={y_test.mean():.4f}  std={y_test.std():.4f}  "
          f"min={y_test.min():.4f}  max={y_test.max():.4f}")
    std_ratio = dist["std"] / y_test.std() * 100
    print(f"예측 표준편차가 실제 target 표준편차의 {std_ratio:.1f}% "
          f"(8종목 9.5% → 50종목 30.8% → 100종목 {std_ratio:.1f}%, 방향: "
          f"{'개선(더 완화)' if std_ratio > 30.8 else '악화(다시 붕괴 쪽)'})")

    id_to_ticker = {v: k for k, v in ticker_to_id.items()}
    hybrid_pred_by_ticker = {}
    hybrid_actual_by_ticker = {}
    for tid in np.unique(hybrid_tids):
        ticker = id_to_ticker[int(tid)]
        mask = hybrid_tids == tid
        dates = pd.DatetimeIndex(d_test[mask])
        hybrid_pred_by_ticker[ticker] = pd.Series(hybrid_preds[mask], index=dates)
        hybrid_actual_by_ticker[ticker] = pd.Series(hybrid_actuals[mask], index=dates)

    # ── 2) baseline 3종 계산 + 하이브리드 test 날짜 기준 정렬 (원본+demean 동시 준비) ──
    print("\n" + "=" * 100)
    print("=== 2) baseline 3종 계산(GARCH/SMA20/Parkinson-SMA20, 100종목) + 날짜 정렬 ===")
    print("=" * 100)
    methods = ["hybrid", "garch", "sma20", "parkinson_sma20"]
    pooled = {m: {"pred": [], "actual": [], "ticker": []} for m in methods}
    garch_rows, sma_rows, park_rows = {}, {}, {}
    common_dates_by_ticker = {}  # 5)의 셔플 재평가가 GARCH/Parkinson을 다시 계산하지 않도록 재사용

    skipped_dates_total = 0
    for ticker in tickers:
        close = load_close_prices(ticker, STOCK_INITIAL_LOAD_START, end_date)
        returns = compute_log_returns_pct(close)
        test_returns = returns[returns.index > val_end]
        sigma_pred, _ = fit_and_forecast_garch(returns, train_end, test_returns.index[0])
        sma_pred = compute_sma_baseline(returns).loc[sigma_pred.index]
        hl = load_high_low(ticker, STOCK_INITIAL_LOAD_START, end_date)
        park_raw = compute_parkinson_vol_pct(hl["high"], hl["low"])
        park_ma20 = park_raw.rolling(PARK_WINDOW, min_periods=PARK_WINDOW).mean().shift(1)

        realized_full = test_returns.abs().loc[sigma_pred.index]
        garch_rows[ticker] = rmse_mae(sigma_pred.values, realized_full.values)
        sma_valid = sma_pred.dropna()
        common_sg = sigma_pred.index.intersection(sma_valid.index)
        sma_rows[ticker] = rmse_mae(sma_valid.loc[common_sg].values, realized_full.loc[common_sg].values)
        park_valid = park_ma20.dropna()
        common_sp = sigma_pred.index.intersection(park_valid.index)
        park_rows[ticker] = rmse_mae(park_valid.loc[common_sp].values, realized_full.loc[common_sp].values)

        if ticker not in hybrid_pred_by_ticker:
            print(f"  ⚠️ {ticker}: 하이브리드 test 시퀀스가 없음(신규 상장 등) — decompose 비교에서 제외")
            continue
        h_pred = hybrid_pred_by_ticker[ticker]
        h_actual = hybrid_actual_by_ticker[ticker]

        common = h_pred.index
        common = common.intersection(sigma_pred.index)
        common = common.intersection(sma_pred.dropna().index)
        common = common.intersection(park_ma20.dropna().index)
        skipped = len(h_pred.index) - len(common)
        skipped_dates_total += skipped
        if len(common) == 0:
            print(f"  ⚠️ {ticker}: 공통 날짜 0 — decompose 비교에서 스킵")
            continue
        common = common.sort_values()
        common_dates_by_ticker[ticker] = common

        actual_hybrid_v = h_actual.loc[common].values
        n = len(common)
        pooled["hybrid"]["pred"].append(h_pred.loc[common].values)
        pooled["hybrid"]["actual"].append(actual_hybrid_v)
        pooled["hybrid"]["ticker"].append(np.full(n, ticker, dtype=object))

        pooled["garch"]["pred"].append(sigma_pred.loc[common].values)
        pooled["garch"]["actual"].append(actual_hybrid_v)
        pooled["garch"]["ticker"].append(np.full(n, ticker, dtype=object))

        pooled["sma20"]["pred"].append(sma_pred.loc[common].values)
        pooled["sma20"]["actual"].append(actual_hybrid_v)
        pooled["sma20"]["ticker"].append(np.full(n, ticker, dtype=object))

        pooled["parkinson_sma20"]["pred"].append(park_ma20.loc[common].values)
        pooled["parkinson_sma20"]["actual"].append(actual_hybrid_v)
        pooled["parkinson_sma20"]["ticker"].append(np.full(n, ticker, dtype=object))

    print(f"\ndecompose 비교용 정렬 실패로 제외된 (종목,날짜) 쌍: {skipped_dates_total}개", flush=True)
    for m in methods:
        pooled[m]["pred"] = np.concatenate(pooled[m]["pred"])
        pooled[m]["actual"] = np.concatenate(pooled[m]["actual"])
        pooled[m]["ticker"] = np.concatenate(pooled[m]["ticker"])
    n_common_total = len(pooled["hybrid"]["pred"])
    print(f"decompose 비교용 공통 정렬 후 pooled 표본 수: {n_common_total} "
          f"(종목 {len(tickers)}개 평균 {n_common_total/len(tickers):.1f}일)", flush=True)

    avg_g_rmse = float(np.mean([v[0] for v in garch_rows.values()])); avg_g_mae = float(np.mean([v[1] for v in garch_rows.values()]))
    avg_s_rmse = float(np.mean([v[0] for v in sma_rows.values()])); avg_s_mae = float(np.mean([v[1] for v in sma_rows.values()]))
    avg_p_rmse = float(np.mean([v[0] for v in park_rows.values()])); avg_p_mae = float(np.mean([v[1] for v in park_rows.values()]))

    def per_ticker_rmse_mae(preds, actuals, tids):
        out = {}
        for ticker, tid in ticker_to_id.items():
            mask = tids == tid
            if mask.sum() == 0:
                continue
            out[ticker] = rmse_mae(preds[mask], actuals[mask])
        return out, rmse_mae(preds, actuals)

    hybrid_per_ticker, hybrid_overall = per_ticker_rmse_mae(hybrid_preds, hybrid_actuals, hybrid_tids)
    avg_h_rmse = float(np.mean([v[0] for v in hybrid_per_ticker.values()]))
    avg_h_mae = float(np.mean([v[1] for v in hybrid_per_ticker.values()]))

    print("\n" + "=" * 100)
    print("=== 3) 종합 비교 (100종목 평균, 종목별 독립계산 기준 — 50종목 스크립트와 동일 방식) ===")
    print("=" * 100)
    print(f"{'구분':<20}{'RMSE':>10}{'MAE':>10}")
    print(f"{'하이브리드(100종목)':<18}{avg_h_rmse:>10.4f}{avg_h_mae:>10.4f}")
    print(f"{'GARCH(1,1)':<20}{avg_g_rmse:>10.4f}{avg_g_mae:>10.4f}")
    print(f"{'SMA20':<20}{avg_s_rmse:>10.4f}{avg_s_mae:>10.4f}")
    print(f"{'Parkinson-SMA20':<20}{avg_p_rmse:>10.4f}{avg_p_mae:>10.4f}")
    print(f"(참고: 전체 test 시퀀스 pooling 기준 RMSE/MAE = {hybrid_overall[0]:.4f} / {hybrid_overall[1]:.4f})")

    beats_garch = avg_h_rmse < avg_g_rmse and avg_h_mae < avg_g_mae
    beats_sma = avg_h_rmse < avg_s_rmse and avg_h_mae < avg_s_mae
    beats_park = avg_h_rmse < avg_p_rmse and avg_h_mae < avg_p_mae
    print(f"\nGARCH 대비 개선: {beats_garch}  |  SMA20 대비 개선: {beats_sma}  |  Parkinson-SMA20 대비 개선: {beats_park}")

    # ── 4) (a) 원본 vs (b) 종목평균 제거 후 — decompose 정렬 표본 기준 ──
    rows = []
    for m in methods:
        pred = pooled[m]["pred"]; actual = pooled[m]["actual"]; tlabel = pooled[m]["ticker"]
        r_raw = pearson_r(pred, actual)
        rmse_raw, mae_raw = rmse_mae(pred, actual)
        pred_dm = demean_by_ticker(pred, tlabel)
        actual_dm = demean_by_ticker(actual, tlabel)
        r_dm = pearson_r(pred_dm, actual_dm)
        rmse_dm, mae_dm = rmse_mae(pred_dm, actual_dm)
        rows.append({"method": m, "n": len(pred), "r_raw": r_raw, "rmse_raw": rmse_raw, "mae_raw": mae_raw,
                     "r_demeaned": r_dm, "rmse_demeaned": rmse_dm, "mae_demeaned": mae_dm})
    by_method = {row["method"]: row for row in rows}

    print("\n" + "=" * 100)
    print("=== 4) (a) 원본 vs (b) 종목평균 제거 후 — 상관계수·RMSE·MAE (decompose 정렬 표본) ===")
    print("=" * 100)
    print(f"{'method':<18}{'n':>6}{'r(raw)':>10}{'r(demean)':>12}{'RMSE(raw)':>12}{'RMSE(dm)':>11}{'MAE(raw)':>10}{'MAE(dm)':>9}")
    for row in rows:
        print(f"{row['method']:<18}{row['n']:>6}{row['r_raw']:>10.4f}{row['r_demeaned']:>12.4f}"
              f"{row['rmse_raw']:>12.4f}{row['rmse_demeaned']:>11.4f}{row['mae_raw']:>10.4f}{row['mae_demeaned']:>9.4f}")

    hy = by_method["hybrid"]
    print("\n=== baseline 대비 하이브리드 우위 — 원본 vs 종목평균 제거 후 (decompose 정렬 표본) ===")
    for m in ["garch", "sma20", "parkinson_sma20"]:
        b = by_method[m]
        print(f"\nvs {m}:")
        print(f"  (a) 원본        : RMSE 개선={hy['rmse_raw'] < b['rmse_raw']} ({hy['rmse_raw']:.4f} vs {b['rmse_raw']:.4f})  "
              f"MAE 개선={hy['mae_raw'] < b['mae_raw']} ({hy['mae_raw']:.4f} vs {b['mae_raw']:.4f})")
        print(f"  (b) 종목평균 제거: RMSE 개선={hy['rmse_demeaned'] < b['rmse_demeaned']} "
              f"({hy['rmse_demeaned']:.4f} vs {b['rmse_demeaned']:.4f})  "
              f"MAE 개선={hy['mae_demeaned'] < b['mae_demeaned']} ({hy['mae_demeaned']:.4f} vs {b['mae_demeaned']:.4f})")
        print(f"  상관계수: (a) r={hy['r_raw']:.4f} vs {b['r_raw']:.4f}  |  (b) r={hy['r_demeaned']:.4f} vs {b['r_demeaned']:.4f}")

    # ── 5) 셔플 검증 N=5 (원본 + 종목평균 제거 후 동시) ──
    print("\n" + "=" * 100)
    print(f"=== 5) 셔플 테스트(N={N_SHUFFLE}) — 원본·종목평균 제거 후 지표 동시 산출 ===")
    print("=" * 100)
    print("주의: N=5는 통계적 확정이 아니라 방향 확인용. p-value 최소 해상도 1/6.")

    # common_dates_by_ticker는 2)에서 이미 계산해뒀다(셔플과 무관하게 날짜 자체는 동일 —
    # lookback 구조상 test 시퀀스 날짜는 y만 셔플해 재학습해도 바뀌지 않는다) — 재계산하지 않음.

    def hybrid_decomposed_metrics(preds_test, actuals_test, tids_test):
        pred_by_t, actual_by_t = {}, {}
        for tid in np.unique(tids_test):
            ticker = id_to_ticker[int(tid)]
            mask = tids_test == tid
            dates = pd.DatetimeIndex(d_test[mask])
            pred_by_t[ticker] = pd.Series(preds_test[mask], index=dates)
            actual_by_t[ticker] = pd.Series(actuals_test[mask], index=dates)

        preds_c, actuals_c, tlabels_c = [], [], []
        for ticker, common in common_dates_by_ticker.items():
            if ticker not in pred_by_t:
                continue
            preds_c.append(pred_by_t[ticker].loc[common].values)
            actuals_c.append(actual_by_t[ticker].loc[common].values)
            tlabels_c.append(np.full(len(common), ticker, dtype=object))
        preds_c = np.concatenate(preds_c); actuals_c = np.concatenate(actuals_c); tlabels_c = np.concatenate(tlabels_c)

        rmse_raw, mae_raw = rmse_mae(preds_c, actuals_c)
        preds_dm = demean_by_ticker(preds_c, tlabels_c)
        actuals_dm = demean_by_ticker(actuals_c, tlabels_c)
        rmse_dm, mae_dm = rmse_mae(preds_dm, actuals_dm)
        return rmse_raw, mae_raw, rmse_dm, mae_dm

    real_rmse_raw, real_mae_raw, real_rmse_dm, real_mae_dm = hybrid_decomposed_metrics(
        hybrid_preds, hybrid_actuals, hybrid_tids
    )
    print(f"\n실제(real) 모델: RMSE(raw)={real_rmse_raw:.6f}  MAE(raw)={real_mae_raw:.6f}  "
          f"RMSE(demean)={real_rmse_dm:.6f}  MAE(demean)={real_mae_dm:.6f}", flush=True)

    shuf_rmse_raw, shuf_mae_raw, shuf_rmse_dm, shuf_mae_dm = [], [], [], []
    for i, seed in enumerate(SHUFFLE_SEEDS):
        rng = np.random.RandomState(seed)
        y_shuf = y_train.copy()
        rng.shuffle(y_shuf)
        r = train_once(y_shuf, f"hybrid100-shuffle{i}", verbose=False)
        preds_s, actuals_s, tids_s = eval_loader(r["model"], X_test_s, y_test, tid_test, device)
        rr, mr, rd, md = hybrid_decomposed_metrics(preds_s, actuals_s, tids_s)
        shuf_rmse_raw.append(rr); shuf_mae_raw.append(mr); shuf_rmse_dm.append(rd); shuf_mae_dm.append(md)
        print(f"  [{i+1}/{N_SHUFFLE}] seed={seed}: RMSE(raw)={rr:.6f}  MAE(raw)={mr:.6f}  "
              f"RMSE(dm)={rd:.6f}  MAE(dm)={md:.6f}  best_epoch={r['best_epoch']}", flush=True)

    shuf_rmse_raw = np.array(shuf_rmse_raw); shuf_mae_raw = np.array(shuf_mae_raw)
    shuf_rmse_dm = np.array(shuf_rmse_dm); shuf_mae_dm = np.array(shuf_mae_dm)

    def stats(real_val, arr, n):
        beat = int(np.sum(arr <= real_val))
        p = (beat + 1) / (n + 1)
        z = (arr.mean() - real_val) / arr.std() if arr.std() > 0 else float("inf")
        return beat, p, arr.mean(), arr.std(), z

    print("\n" + "=" * 100)
    print(f"=== 셔플 검증 결과 요약 (N={N_SHUFFLE}) ===")
    print("=" * 100)
    for label, real_val, arr in [
        ("RMSE(원본)", real_rmse_raw, shuf_rmse_raw),
        ("MAE(원본)", real_mae_raw, shuf_mae_raw),
        ("RMSE(종목평균제거)", real_rmse_dm, shuf_rmse_dm),
        ("MAE(종목평균제거)", real_mae_dm, shuf_mae_dm),
    ]:
        beat, p, mean_, std_, z = stats(real_val, arr, N_SHUFFLE)
        print(f"{label:<20}: 셔플 {mean_:.6f}±{std_:.6f} | 실제 {real_val:.6f} | "
              f"격차={z:.2f}표준편차 | {beat}/{N_SHUFFLE} 노이즈가 이김 | p={p:.4f}")

    print("\n" + "=" * 100)
    print("=== 최종 판정 ===")
    print("=" * 100)
    print(f"GARCH 대비 개선(원본 RMSE/MAE): {beats_garch}  |  SMA20 대비: {beats_sma}  |  Parkinson-SMA20 대비: {beats_park}")
    print(f"예측 표준편차/target 표준편차 비율: {std_ratio:.1f}% (8종목 9.5% → 50종목 30.8% → 100종목 {std_ratio:.1f}%)")
    print(f"종목평균 제거 후 상관계수: hybrid r={hy['r_demeaned']:.4f} "
          f"(50종목 결과 참고치: r=0.2556)")
    r_beats_any_dm = any(hy['rmse_demeaned'] < by_method[m]['rmse_demeaned'] and hy['mae_demeaned'] < by_method[m]['mae_demeaned']
                          for m in ["garch", "sma20", "parkinson_sma20"])
    print(f"종목평균 제거 후에도 baseline 중 하나 이상을 이기는가: {r_beats_any_dm}")
    print("(주: 5)의 셔플 검증은 N=5로 통계적 확정이 아니라 방향 확인 — 격차(σ) 크기를 함께 볼 것)")


if __name__ == "__main__":
    main()
