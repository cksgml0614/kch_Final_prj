# test_evaluation_pooled50_decompose.py — 50종목 하이브리드 모델의 예측력이 "종목 간 평균
# 수준 차이"에서 오는지 "종목 내부 날짜별 변동"(진짜 시계열 예측)에서 오는지 분해 검증
# (2026-09-07, 일회성 — 사용자가 "분산이 커진 게 종목 평균 수준 차이 반영일 뿐일 수 있다"는
# 우려를 제기해 실행).
#
# 방법: test_evaluation_pooled50_hybrid.py와 완전히 동일한 조건(같은 시드/하이퍼파라미터)으로
# 하이브리드 pooled Transformer를 재학습해 test 예측을 얻는다. baseline 3종(GARCH/SMA20/
# Parkinson-SMA20)도 동일 종목·동일 구간에서 각자 예측을 계산한다. 네 방법 모두에 대해:
#   (a) 원본 예측 vs 원본 실제값의 Pearson 상관계수 (지금까지 본 것)
#   (b) 종목별 평균을 뺀("demean") 잔차끼리의 Pearson 상관계수 — 종목 고유 수준 차이를
#       걷어내고 날짜별 변동만 남긴 것 = 진짜 시계열 예측력
# 네 방법의 "실제값"은 전부 동일한 정의(|log(close_t/close_(t-1))x100|, dataset_builder.
# build_base_dataset_v2_volatility의 target)를 쓰므로, 하이브리드 모델의 test 시퀀스 날짜에
# baseline 예측을 정렬시켜 비교 대상을 완전히 동일하게 맞춘다(하이브리드 쪽 실제값을
# 공통 ground truth로 사용 — baseline 쪽 realized와 정의가 같음을 사후에 교차 검증).

import sys
from datetime import date

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
from 가격예측.garch_baseline import (
    compute_log_returns_pct, compute_sma_baseline, fit_and_forecast_garch,
    load_close_prices, rmse_mae,
)
from 가격예측.parkinson_volatility_check import compute_parkinson_vol_pct, load_high_low, PARK_WINDOW
from 가격예측.pooled_dataset import build_pooled_sequences, compute_global_split_dates
from 가격예측.sequence_dataset import FeatureScaler
from 가격예측.split_dataset import build_merged_dataset_v2, build_merged_dataset_v2_volatility_hybrid
from 가격예측.train_common import evaluate_pooled_predictions, train_pooled_transformer
from 가격예측.test_evaluation_pooled50_hybrid import EMBEDDING_DIM_50
from 가격예측.가격예측_통합모델 import (
    BATCH_SIZE, DIM_FEEDFORWARD, DROPOUT, D_MODEL, LOOKBACK, LR,
    MAX_EPOCHS, NHEAD, NUM_LAYERS, PATIENCE, SEED, SMOKE_EPOCHS, WEIGHT_DECAY,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def eval_loader(model, X, y, tid, device):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(tid)),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    return evaluate_pooled_predictions(model, loader, device)


def pearson_r(pred, actual):
    if len(pred) < 2 or np.std(pred) == 0 or np.std(actual) == 0:
        return float("nan")
    return float(np.corrcoef(pred, actual)[0, 1])


def demean_by_ticker(values, ticker_labels):
    """ticker_labels(같은 길이의 종목 식별 배열) 기준으로 각 종목의 평균을 빼 잔차를 반환."""
    out = np.empty_like(values, dtype=np.float64)
    for t in np.unique(ticker_labels):
        mask = ticker_labels == t
        out[mask] = values[mask] - values[mask].mean()
    return out


def main():
    tickers = ACTIVE_TICKERS
    end_date = date.today().isoformat()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"=== 50종목 하이브리드 모델 — 종목간 vs 종목내 예측력 분해 ===")

    # ── 1) 하이브리드 pooled Transformer 재학습 (test_evaluation_pooled50_hybrid.py와 동일 조건) ──
    ref_vol, _ = build_merged_dataset_v2(tickers[0], STOCK_INITIAL_LOAD_START, end_date)
    train_end, val_end = compute_global_split_dates(ref_vol.index)
    print(f"전역 분할 경계: train_end={train_end.date()}  val_end={val_end.date()}")

    def hybrid_build_fn(ticker, s, e):
        return build_merged_dataset_v2_volatility_hybrid(ticker, s, e, train_end)

    splits, feature_cols, ticker_to_id, (train_end2, val_end2) = build_pooled_sequences(
        tickers, STOCK_INITIAL_LOAD_START, end_date, lookback=LOOKBACK, build_fn=hybrid_build_fn,
    )
    assert train_end2 == train_end and val_end2 == val_end
    X_train, y_train, tid_train, d_train, _ = splits["train"]
    X_val, y_val, tid_val, d_val, _ = splits["val"]
    X_test, y_test, tid_test, d_test, tname_test = splits["test"]
    print(f"시퀀스 shape: X_train {X_train.shape}  X_test {X_test.shape}")

    scaler = FeatureScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    result = train_pooled_transformer(
        X_train_s, y_train, tid_train, X_val_s, y_val, tid_val,
        len(feature_cols), LOOKBACK, len(tickers), EMBEDDING_DIM_50,
        D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT,
        BATCH_SIZE, LR, WEIGHT_DECAY, SMOKE_EPOCHS, MAX_EPOCHS, PATIENCE,
        device, seed=SEED, verbose=True, label="hybrid50-decompose",
    )
    hybrid_preds, hybrid_actuals, hybrid_tids = eval_loader(result["model"], X_test_s, y_test, tid_test, device)
    print(f"하이브리드 재학습 완료: best_epoch={result['best_epoch']}")

    # 종목별 (날짜 -> 값) 시리즈로 정리 (baseline과의 날짜 정렬용)
    id_to_ticker = {v: k for k, v in ticker_to_id.items()}
    hybrid_pred_by_ticker = {}
    hybrid_actual_by_ticker = {}
    for tid in np.unique(hybrid_tids):
        ticker = id_to_ticker[int(tid)]
        mask = hybrid_tids == tid
        dates = pd.DatetimeIndex(d_test[mask])
        hybrid_pred_by_ticker[ticker] = pd.Series(hybrid_preds[mask], index=dates)
        hybrid_actual_by_ticker[ticker] = pd.Series(hybrid_actuals[mask], index=dates)

    # ── 2) baseline 3종 계산 (GARCH/SMA20/Parkinson-SMA20), 하이브리드 test 날짜에 정렬 ──
    print("\n" + "=" * 100)
    print("=== baseline 계산 + 하이브리드 test 날짜 기준 정렬 ===")
    print("=" * 100)

    methods = ["hybrid", "garch", "sma20", "parkinson_sma20"]
    pooled = {m: {"pred": [], "actual": [], "ticker": []} for m in methods}

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

        h_pred = hybrid_pred_by_ticker[ticker]
        h_actual = hybrid_actual_by_ticker[ticker]

        # 네 방법 전부에서 값이 존재하는 날짜만 사용 (완전히 동일한 비교 대상 확보)
        common = h_pred.index
        common = common.intersection(sigma_pred.index)
        common = common.intersection(sma_pred.dropna().index)
        common = common.intersection(park_ma20.dropna().index)
        skipped = len(h_pred.index) - len(common)
        skipped_dates_total += skipped
        if skipped > 0:
            print(f"  {ticker}: 하이브리드 test {len(h_pred.index)}일 중 {skipped}일 baseline 정렬 실패로 제외")
        if len(common) == 0:
            print(f"  ⚠️ {ticker}: 공통 날짜 0 — 스킵")
            continue
        common = common.sort_values()

        # 실제값 정의 일치 사후 검증 (하이브리드 target vs baseline realized)
        actual_hybrid_v = h_actual.loc[common].values
        actual_baseline_v = test_returns.abs().loc[common].values
        max_diff = float(np.max(np.abs(actual_hybrid_v - actual_baseline_v))) if len(common) else float("nan")
        if max_diff > 1e-6:
            print(f"  ⚠️ {ticker}: 실제값 정의 불일치 감지 (max|Δ|={max_diff:.8f}) — 하이브리드 target 값을 ground truth로 사용")

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

    print(f"\n전체 baseline 정렬 실패로 제외된 (종목,날짜) 쌍: {skipped_dates_total}개")

    for m in methods:
        pooled[m]["pred"] = np.concatenate(pooled[m]["pred"])
        pooled[m]["actual"] = np.concatenate(pooled[m]["actual"])
        pooled[m]["ticker"] = np.concatenate(pooled[m]["ticker"])

    n_common_total = len(pooled["hybrid"]["pred"])
    print(f"공통 정렬 후 pooled 표본 수: {n_common_total} (종목 {len(tickers)}개 평균 {n_common_total/len(tickers):.1f}일)")

    # ── 3) (a) 원본 상관계수, (b) 종목평균 제거 후 상관계수/RMSE/MAE ──
    rows = []
    for m in methods:
        pred = pooled[m]["pred"]
        actual = pooled[m]["actual"]
        tlabel = pooled[m]["ticker"]

        r_raw = pearson_r(pred, actual)
        rmse_raw, mae_raw = rmse_mae(pred, actual)

        pred_dm = demean_by_ticker(pred, tlabel)
        actual_dm = demean_by_ticker(actual, tlabel)
        r_dm = pearson_r(pred_dm, actual_dm)
        rmse_dm, mae_dm = rmse_mae(pred_dm, actual_dm)

        rows.append({
            "method": m, "n": len(pred),
            "r_raw": r_raw, "rmse_raw": rmse_raw, "mae_raw": mae_raw,
            "r_demeaned": r_dm, "rmse_demeaned": rmse_dm, "mae_demeaned": mae_dm,
        })

    print("\n" + "=" * 100)
    print("=== (a) 원본 vs (b) 종목평균 제거 후 — 상관계수·RMSE·MAE ===")
    print("=" * 100)
    print(f"{'method':<18}{'n':>6}{'r(raw)':>10}{'r(demean)':>12}{'RMSE(raw)':>12}{'RMSE(dm)':>11}{'MAE(raw)':>10}{'MAE(dm)':>9}")
    for row in rows:
        print(f"{row['method']:<18}{row['n']:>6}{row['r_raw']:>10.4f}{row['r_demeaned']:>12.4f}"
              f"{row['rmse_raw']:>12.4f}{row['rmse_demeaned']:>11.4f}{row['mae_raw']:>10.4f}{row['mae_demeaned']:>9.4f}")

    # ── 4) baseline 대비 비교 (원본 vs demean 각각) ──
    print("\n" + "=" * 100)
    print("=== baseline 대비 하이브리드 우위 — 원본 vs 종목평균 제거 후 ===")
    print("=" * 100)
    by_method = {row["method"]: row for row in rows}
    hy = by_method["hybrid"]
    for m in ["garch", "sma20", "parkinson_sma20"]:
        b = by_method[m]
        print(f"\nvs {m}:")
        print(f"  (a) 원본        : hybrid RMSE={hy['rmse_raw']:.4f} vs {m} RMSE={b['rmse_raw']:.4f}  "
              f"(하이브리드 개선: {hy['rmse_raw'] < b['rmse_raw']})   "
              f"hybrid MAE={hy['mae_raw']:.4f} vs {m} MAE={b['mae_raw']:.4f} (개선: {hy['mae_raw'] < b['mae_raw']})")
        print(f"  (b) 종목평균 제거: hybrid RMSE={hy['rmse_demeaned']:.4f} vs {m} RMSE={b['rmse_demeaned']:.4f}  "
              f"(하이브리드 개선: {hy['rmse_demeaned'] < b['rmse_demeaned']})   "
              f"hybrid MAE={hy['mae_demeaned']:.4f} vs {m} MAE={b['mae_demeaned']:.4f} (개선: {hy['mae_demeaned'] < b['mae_demeaned']})")
        print(f"  상관계수: (a) hybrid r={hy['r_raw']:.4f} vs {m} r={b['r_raw']:.4f}  |  "
              f"(b) hybrid r={hy['r_demeaned']:.4f} vs {m} r={b['r_demeaned']:.4f}")

    # ── 5) 결론 판정 ──
    print("\n" + "=" * 100)
    print("=== 결론 ===")
    print("=" * 100)
    r_drop = hy["r_raw"] - hy["r_demeaned"]
    print(f"하이브리드 상관계수: 원본 r={hy['r_raw']:.4f}  ->  종목평균 제거 후 r={hy['r_demeaned']:.4f}  (하락폭={r_drop:.4f})")
    if hy["r_demeaned"] < 0.1 or np.isnan(hy["r_demeaned"]):
        verdict = "종목평균 제거 후 상관계수가 거의 0에 가까움 — 우려대로 '종목 간 차이만 맞히고 진짜 시계열 예측은 약함'"
    elif r_drop > 0.3:
        verdict = "종목평균 제거 후 상관계수가 크게 하락 — 원본 상관관계 상당 부분이 종목 간 평균 수준 차이에서 왔을 가능성이 큼(우려가 상당 부분 맞음)"
    else:
        verdict = "종목평균 제거 후에도 상관계수가 의미 있게 유지됨 — 종목 간 차이를 걷어내고도 날짜별 예측력이 살아있음(우려가 기각됨)"
    print(f"판정: {verdict}")

    beats_any_demeaned = any(hy["rmse_demeaned"] < by_method[m]["rmse_demeaned"] and
                              hy["mae_demeaned"] < by_method[m]["mae_demeaned"]
                              for m in ["garch", "sma20", "parkinson_sma20"])
    print(f"\n종목평균 제거 후에도 baseline(GARCH/SMA20/Parkinson-SMA20) 중 하나 이상을 이기는가: {beats_any_demeaned}")
    print("(주: 이 스크립트는 상관계수/RMSE 분해만 수행 — 유의성(셔플) 검증은 범위 밖)")


if __name__ == "__main__":
    main()
