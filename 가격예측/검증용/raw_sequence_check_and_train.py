# raw_sequence_check_and_train.py — "원본에 가까운 시퀀스"를 Transformer에 직접 주는 실험
# (2026-09-06, 일회성). 압축 지속성 피처(SMA20/recent_vol_ma20/Parkinson-SMA20/momentum)를
# 전부 빼고, close_return/hl_range_ratio/open_gap_ratio/log_volume(원본에 가까운 일별 값)
# + GARCH σ 1개 + 거시지표 8개 + 종목 임베딩만으로 재구성한다.
#
# 0) 날짜 매핑 사전 검증을 최우선으로 실행하고, 하나라도 어긋나면 학습을 진행하지 않고 즉시
#    중단한다(오늘 GARCH 라벨링 버그를 뒤늦게 발견한 사고 재발 방지).

import sys
from datetime import date

import numpy as np
import pandas as pd
import torch

from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
from db_manager import get_db_connection
from 가격예측.dataset_builder import build_base_dataset_v2_volatility
from 가격예측.garch_baseline import (
    compute_log_returns_pct, compute_sma_baseline,
    load_close_prices, rmse_mae,
)
from 가격예측.검증용.garch_baseline_check import fit_and_forecast_garch
from 가격예측.pooled_dataset import build_pooled_sequences, compute_global_split_dates
from 가격예측.sequence_dataset import FeatureScaler
from 가격예측.split_dataset import build_merged_dataset_v2
from 가격예측.train_common import evaluate_pooled_predictions, train_pooled_transformer
from 가격예측.parkinson_baseline import compute_parkinson_vol_pct, load_high_low, PARK_WINDOW
from 가격예측.검증용.가격예측_통합모델 import (
    BATCH_SIZE, DIM_FEEDFORWARD, DROPOUT, D_MODEL, EMBEDDING_DIM, LOOKBACK, LR,
    MAX_EPOCHS, NHEAD, NUM_LAYERS, PATIENCE, SEED, SMOKE_EPOCHS, WEIGHT_DECAY,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

N_SHUFFLE = 7
SHUFFLE_SEEDS = [0, 1, 7, 123, 2024, 777, 999]
VERIFY_TICKERS = [("005930", 60), ("105560", 130), ("035420", 200)]  # 8종목 중 임의 3종목, D 오프셋도 다르게

HYBRID15_AVG = (2.709781, 1.833777)


def build_raw_sequence_dataset(ticker, start_date, end_date, train_end, precomputed_sigma=None, garch_params=None):
    """12개 base(V2, recent_vol_ma20 제외) + garch_sigma 1개 = 13피처. momentum/recent_vol_ma20
    전부 제외 — '원본에 가까운' 시퀀스만 남긴다."""
    merged, meta = build_base_dataset_v2_volatility(ticker, start_date, end_date)
    merged = merged.drop(columns=["recent_vol_ma20"])

    if precomputed_sigma is not None:
        sigma_full, params = precomputed_sigma, garch_params
    else:
        from 가격예측.검증용.garch_baseline_check import compute_full_period_sigma
        sigma_full, params = compute_full_period_sigma(ticker, start_date, end_date, train_end)

    merged = merged.copy()
    merged["garch_sigma"] = sigma_full.reindex(merged.index)
    before = len(merged)
    merged = merged.dropna(how="any")
    meta = dict(meta)
    meta["garch_params"] = params
    meta["rows_before_garch_dropna"] = before
    meta["rows_after_garch_dropna"] = len(merged)
    return merged, meta


def load_raw_ohlcv(ticker, start_date, end_date):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT date, open, high, low, close, volume FROM daily_stock_prices WHERE ticker=%s AND date BETWEEN %s AND %s ORDER BY date",
                (ticker, start_date, end_date),
            )
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"]).astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": float})
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def verify_date_alignment(ticker, train_end, val_end, D_offset=100):
    """[0] 날짜 매핑 사전 검증 — 실제 원본 값으로 손 계산해서 대조한다(코드 리뷰 아님).
    거시지표(KOSPI 등)의 '이중 시프트'는 별도 세션에서 이미 실제값 대조로 확인·판정 완료된
    사안(누수 아님, 최신성만 하루 손해)이라 이번 게이트에서는 재검증 대상에서 제외한다 —
    이번 요청은 close_return/hl_range_ratio/open_gap_ratio/log_volume/garch_sigma/target에
    집중."""
    problems = []
    end_date = date.today().isoformat()
    close = load_close_prices(ticker, STOCK_INITIAL_LOAD_START, end_date)
    returns = compute_log_returns_pct(close)
    ohlcv = load_raw_ohlcv(ticker, STOCK_INITIAL_LOAD_START, end_date)

    merged, meta = build_raw_sequence_dataset(ticker, STOCK_INITIAL_LOAD_START, end_date, train_end)
    test_dates = merged.index[merged.index > val_end]
    D = test_dates[min(D_offset, len(test_dates) - 1)]
    pos_all = merged.index.get_loc(D)
    if pos_all < LOOKBACK - 1:
        problems.append(f"{ticker}: D={D.date()}가 lookback 확보 전 위치")
        return False, problems, None

    window = merged.iloc[pos_all - LOOKBACK + 1: pos_all + 1]
    print(f"\n--- {ticker}  D={D.date()}  (윈도우 {window.index[0].date()} ~ {window.index[-1].date()}) ---")

    # 원본 OHLCV로 4개 raw 피처를 직접 손 계산 (build_stationary_features 공식 그대로,
    # 코드 호출 없이 pandas 연산만 사용)
    ohlcv_close_return = ohlcv["close"].pct_change()
    ohlcv_hl_range = (ohlcv["high"] - ohlcv["low"]) / ohlcv["close"]
    ohlcv_open_gap = (ohlcv["open"] - ohlcv["close"].shift(1)) / ohlcv["close"].shift(1)
    ohlcv_log_volume = np.log1p(ohlcv["volume"])

    print(f"{'행라벨(D상대)':<12}{'대응실현일':<12}{'close_return':>13}{'hl_range':>10}{'open_gap':>10}{'log_vol':>9}{'garch_σ':>10}{'전부일치':>9}")
    all_ok = True
    for row_label in window.index:
        pos = ohlcv.index.get_loc(row_label)
        prev_day = ohlcv.index[pos - 1]  # 이 행의 4개 raw 피처가 실제로 실현된 날짜
        exp_cr = ohlcv_close_return.loc[prev_day]
        exp_hl = ohlcv_hl_range.loc[prev_day]
        exp_og = ohlcv_open_gap.loc[prev_day]
        exp_lv = ohlcv_log_volume.loc[prev_day]

        act_cr = window.loc[row_label, "close_return"]
        act_hl = window.loc[row_label, "hl_range_ratio"]
        act_og = window.loc[row_label, "open_gap_ratio"]
        act_lv = window.loc[row_label, "log_volume"]
        act_garch = window.loc[row_label, "garch_sigma"]

        ok_cr = np.isclose(act_cr, exp_cr, atol=1e-6)
        ok_hl = np.isclose(act_hl, exp_hl, atol=1e-6)
        ok_og = np.isclose(act_og, exp_og, atol=1e-6)
        ok_lv = np.isclose(act_lv, exp_lv, atol=1e-6)
        row_ok = ok_cr and ok_hl and ok_og and ok_lv
        all_ok = all_ok and row_ok

        rel = (row_label - D).days
        print(f"{row_label.date().isoformat()+f'(D{rel:+d})':<12}{prev_day.date().isoformat():<12}"
              f"{act_cr:>13.6f}{act_hl:>10.6f}{act_og:>10.6f}{act_lv:>9.4f}{act_garch:>10.4f}{str(row_ok):>9}")

    if not all_ok:
        problems.append(f"{ticker}: close_return/hl_range_ratio/open_gap_ratio/log_volume 중 하나 이상 원본 대조 불일치")

    # D 당일 값이 섞였는지 명시 확인 — 윈도우 마지막 행(=D)의 4개 값이 'D 당일 실현값'과
    # 같은 정의로 재계산했을 때 다른지(=D-1 값이어야 정상) 직접 비교
    D_own_cr = ohlcv_close_return.loc[D]
    D_own_hl = ohlcv_hl_range.loc[D]
    D_row_cr = window.loc[D, "close_return"]
    D_row_hl = window.loc[D, "hl_range_ratio"]
    d_leak = np.isclose(D_row_cr, D_own_cr, atol=1e-6) or np.isclose(D_row_hl, D_own_hl, atol=1e-6)
    print(f"\nD={D.date()} 행: close_return={D_row_cr:.6f} (D당일 실제값={D_own_cr:.6f}, 같으면 리크) / "
          f"hl_range_ratio={D_row_hl:.6f} (D당일 실제값={D_own_hl:.6f}, 같으면 리크)")
    print(f"D 당일 값 섞임 여부: {d_leak} (False여야 정상)")
    if d_leak:
        problems.append(f"{ticker}: D행에 D 당일 원본 값이 섞임(리크)")

    # target(D) 검증 — |D일 실현 수익률(로그x100)|과 원본에서 직접 재계산해 대조
    target_D = merged.loc[D, "target"]
    pos_D_ret = returns.index.get_loc(D)
    manual_target_D = abs(returns.loc[D])
    target_ok = np.isclose(target_D, manual_target_D, atol=1e-6)
    print(f"target(D)={target_D:.6f}  원본 재계산 |D일 로그수익률x100|={manual_target_D:.6f}  일치: {target_ok}")
    if not target_ok:
        problems.append(f"{ticker}: target(D)이 원본 재계산값과 불일치")

    # garch_sigma 재검증 — 수동 재귀식(arch 미사용)으로 D 및 윈도우 전체 재계산해 대조
    from arch import arch_model
    train_returns = returns[returns.index <= train_end]
    am = arch_model(train_returns.values, mean="Constant", vol="Garch", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    mu, omega, alpha, beta = res.params["mu"], res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"]
    eps = returns.values - mu
    n = len(eps)
    sigma2_manual = np.empty(n)
    sigma2_manual[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        sigma2_manual[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2_manual[t - 1]

    garch_all_ok = True
    for row_label in window.index:
        pos_r = returns.index.get_loc(row_label)
        manual_sigma_r = np.sqrt(sigma2_manual[pos_r])
        func_sigma_r = window.loc[row_label, "garch_sigma"]
        ok = np.isclose(manual_sigma_r, func_sigma_r, atol=1e-6)
        garch_all_ok = garch_all_ok and ok
    manual_sigma_D = np.sqrt(sigma2_manual[pos_D_ret])
    func_sigma_D = merged.loc[D, "garch_sigma"]
    print(f"D행 garch_sigma: 수동재귀식={manual_sigma_D:.8f}  merged값={func_sigma_D:.8f}  일치: "
          f"{np.isclose(manual_sigma_D, func_sigma_D, atol=1e-6)}  (윈도우 20행 전부 일치: {garch_all_ok})")
    if not garch_all_ok:
        problems.append(f"{ticker}: 윈도우 내 garch_sigma가 수동 재귀식과 불일치하는 행 있음")

    return (len(problems) == 0), problems, merged


def main():
    end_date = date.today().isoformat()
    reference_merged, _ = build_merged_dataset_v2(ACTIVE_TICKERS[0], STOCK_INITIAL_LOAD_START, end_date)
    train_end, val_end = compute_global_split_dates(reference_merged.index)
    print(f"전역 분할 경계: train_end={train_end.date()}  val_end={val_end.date()}")

    print("\n" + "=" * 100)
    print("=== [0] 날짜 매핑 사전 검증 (8종목 중 임의 3종목, 실제 값 대조) ===")
    print("=" * 100)
    all_pass = True
    all_problems = []
    for ticker, offset in VERIFY_TICKERS:
        ok, problems, _ = verify_date_alignment(ticker, train_end, val_end, D_offset=offset)
        all_pass = all_pass and ok
        all_problems.extend(problems)

    print("\n" + "=" * 100)
    print(f"=== [0] 검증 결과: {'전부 통과' if all_pass else '실패 — 학습 중단'} ===")
    print("=" * 100)
    if not all_pass:
        print("발견된 문제:")
        for p in all_problems:
            print(f"  - {p}")
        print("\n>>> 검증 실패 — 학습을 진행하지 않고 종료합니다.")
        return

    print("검증 통과 — 학습 진행")

    # ── [1]~[2] 13피처(원본에 가까운 4개 + garch_sigma + 거시 8개) 하이브리드 재구성 ──
    # ⚠️ raw_build_fn은 momentum(60일 버퍼) 없이 build_base_dataset_v2_volatility만 쓰므로
    # build_merged_dataset_v2보다 인덱스가 더 일찍 시작한다 — build_pooled_sequences가 내부적으로
    # 이 함수의 결과만으로 전역 분할 경계를 재계산하면 다른 실험(GARCH/15피처 하이브리드 등)의
    # train_end/val_end와 어긋난다(실측: AssertionError로 확인). reference_merged.index[0](=
    # build_merged_dataset_v2 기준 시작일, 8종목이 캘린더를 공유함은 pooled_dataset.py 상단
    # 전제로 이미 확인됨)로 앞부분을 잘라 행 수를 맞춰, 동일 비율 분할이 동일 날짜로 떨어지게
    # 한다.
    raw_seq_start = reference_merged.index[0]

    def raw_build_fn(ticker, s, e):
        merged, meta = build_raw_sequence_dataset(ticker, s, e, train_end)
        merged = merged[merged.index >= raw_seq_start]
        return merged, meta

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\ndevice: {device}")
    splits, feature_cols, ticker_to_id, (train_end2, val_end2) = build_pooled_sequences(
        ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START, end_date, lookback=LOOKBACK, build_fn=raw_build_fn,
    )
    assert train_end2 == train_end and val_end2 == val_end
    print(f"feature_cols({len(feature_cols)}개): {feature_cols}")
    X_train, y_train, tid_train, d_train, _ = splits["train"]
    X_val, y_val, tid_val, d_val, _ = splits["val"]
    X_test, y_test, tid_test, d_test, _ = splits["test"]
    print(f"시퀀스 shape: X_train {X_train.shape}  X_test {X_test.shape}")

    scaler = FeatureScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    def eval_loader(model, X, y, tid):
        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(tid)),
                             batch_size=BATCH_SIZE, shuffle=False)
        return evaluate_pooled_predictions(model, loader, device)

    def train_once(y_arr, label, verbose):
        return train_pooled_transformer(
            X_train_s, y_arr, tid_train, X_val_s, y_val, tid_val,
            len(feature_cols), LOOKBACK, len(ACTIVE_TICKERS), EMBEDDING_DIM,
            D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT,
            BATCH_SIZE, LR, WEIGHT_DECAY, SMOKE_EPOCHS, MAX_EPOCHS, PATIENCE,
            device, seed=SEED, verbose=verbose, label=label,
        )

    result = train_once(y_train, "rawseq-real", verbose=True)
    real_preds, real_actuals, real_tids = eval_loader(result["model"], X_test_s, y_test, tid_test)

    def per_ticker_rmse_mae(preds, actuals, tids):
        out = {}
        for ticker, tid in ticker_to_id.items():
            mask = tids == tid
            out[ticker] = rmse_mae(preds[mask], actuals[mask])
        return out, rmse_mae(preds, actuals)

    raw_per_ticker, raw_overall = per_ticker_rmse_mae(real_preds, real_actuals, real_tids)
    avg_raw_rmse = float(np.mean([raw_per_ticker[t][0] for t in ACTIVE_TICKERS]))
    avg_raw_mae = float(np.mean([raw_per_ticker[t][1] for t in ACTIVE_TICKERS]))

    # ── baseline 3종 계산 (GARCH, SMA20, Parkinson-SMA20) ──
    print("\n" + "=" * 100)
    print("=== baseline 3종 재계산 ===")
    print("=" * 100)
    baseline_rows = {}
    for ticker in ACTIVE_TICKERS:
        close = load_close_prices(ticker, STOCK_INITIAL_LOAD_START, end_date)
        returns = compute_log_returns_pct(close)
        test_returns = returns[returns.index > val_end]
        sigma_pred, _ = fit_and_forecast_garch(returns, train_end, test_returns.index[0])
        sma20 = compute_sma_baseline(returns).loc[sigma_pred.index]
        hl = load_high_low(ticker, STOCK_INITIAL_LOAD_START, end_date)
        park_raw = compute_parkinson_vol_pct(hl["high"], hl["low"])
        park_ma20 = park_raw.rolling(PARK_WINDOW, min_periods=PARK_WINDOW).mean().shift(1).loc[sigma_pred.index]
        realized = test_returns.abs().loc[sigma_pred.index]

        g_rmse, g_mae = rmse_mae(sigma_pred.values, realized.values)
        s_rmse, s_mae = rmse_mae(sma20.values, realized.values)
        p_rmse, p_mae = rmse_mae(park_ma20.dropna().values, realized.loc[park_ma20.dropna().index].values)
        baseline_rows[ticker] = {"g_rmse": g_rmse, "g_mae": g_mae, "s_rmse": s_rmse, "s_mae": s_mae, "p_rmse": p_rmse, "p_mae": p_mae}

    avg_g_rmse = np.mean([baseline_rows[t]["g_rmse"] for t in ACTIVE_TICKERS])
    avg_g_mae = np.mean([baseline_rows[t]["g_mae"] for t in ACTIVE_TICKERS])
    avg_s_rmse = np.mean([baseline_rows[t]["s_rmse"] for t in ACTIVE_TICKERS])
    avg_s_mae = np.mean([baseline_rows[t]["s_mae"] for t in ACTIVE_TICKERS])
    avg_p_rmse = np.mean([baseline_rows[t]["p_rmse"] for t in ACTIVE_TICKERS])
    avg_p_mae = np.mean([baseline_rows[t]["p_mae"] for t in ACTIVE_TICKERS])

    print("\n" + "=" * 130)
    print("=== 종합 비교 — 원본시퀀스(13피처) vs 15피처하이브리드 vs GARCH vs SMA20 vs Parkinson-SMA20 ===")
    print("=" * 130)
    header = f"{'ticker':<10}{'원본시퀀스RMSE':>15}{'원본시퀀스MAE':>14}{'GARCH RMSE':>12}{'SMA20 RMSE':>11}{'ParkSMA20 RMSE':>15}"
    print(header)
    for ticker in ACTIVE_TICKERS:
        r = raw_per_ticker[ticker]
        b = baseline_rows[ticker]
        print(f"{ticker:<10}{r[0]:>15.4f}{r[1]:>14.4f}{b['g_rmse']:>12.4f}{b['s_rmse']:>11.4f}{b['p_rmse']:>15.4f}")
    print(f"{'평균':<10}{avg_raw_rmse:>15.4f}{avg_raw_mae:>14.4f}{avg_g_rmse:>12.4f}{avg_s_rmse:>11.4f}{avg_p_rmse:>15.4f}")
    print(f"(참고) 15피처 하이브리드 평균: RMSE={HYBRID15_AVG[0]:.4f}  MAE={HYBRID15_AVG[1]:.4f}")

    beats_15 = avg_raw_rmse < HYBRID15_AVG[0] and avg_raw_mae < HYBRID15_AVG[1]
    beats_garch = avg_raw_rmse < avg_g_rmse and avg_raw_mae < avg_g_mae
    beats_sma = avg_raw_rmse < avg_s_rmse and avg_raw_mae < avg_s_mae
    beats_park = avg_raw_rmse < avg_p_rmse and avg_raw_mae < avg_p_mae
    print(f"\n[핵심 확인 1] 15피처 하이브리드보다 개선: {beats_15}")
    print(f"[핵심 확인 2] GARCH(1,1) 대비 개선: {beats_garch}")
    print(f"[핵심 확인 3] SMA20 대비 개선: {beats_sma}")
    print(f"[핵심 확인 4] Parkinson-SMA20 대비 개선: {beats_park}")

    # ── 셔플 테스트 N=7 ──
    print("\n" + "=" * 100)
    print(f"=== 레이블 셔플 테스트 (N={N_SHUFFLE}) ===")
    print("=" * 100)
    shuffle_rmse, shuffle_mae = [], []
    for i, seed in enumerate(SHUFFLE_SEEDS):
        rng = np.random.RandomState(seed)
        y_shuf = y_train.copy()
        rng.shuffle(y_shuf)
        r = train_once(y_shuf, f"rawseq-shuffle{i}", verbose=False)
        preds_s, actuals_s, _ = eval_loader(r["model"], X_test_s, y_test, tid_test)
        rmse_s, mae_s = rmse_mae(preds_s, actuals_s)
        shuffle_rmse.append(rmse_s); shuffle_mae.append(mae_s)
        print(f"  [{i+1}/{N_SHUFFLE}] seed={seed}: RMSE={rmse_s:.6f}  MAE={mae_s:.6f}  best_epoch={r['best_epoch']}")

    shuffle_rmse = np.array(shuffle_rmse); shuffle_mae = np.array(shuffle_mae)

    def _p(real_val, arr):
        beat = int(np.sum(arr <= real_val))
        return beat, (beat + 1) / (N_SHUFFLE + 1)

    n_beat_rmse, p_rmse = _p(raw_overall[0], shuffle_rmse)
    n_beat_mae, p_mae = _p(raw_overall[1], shuffle_mae)
    print(f"\n셔플 분포: RMSE {shuffle_rmse.mean():.6f}±{shuffle_rmse.std():.6f} | 실제 {raw_overall[0]:.6f}")
    print(f"셔플 분포: MAE  {shuffle_mae.mean():.6f}±{shuffle_mae.std():.6f} | 실제 {raw_overall[1]:.6f}")
    print(f"노이즈가 실제와 같거나 더 좋은 RMSE: {n_beat_rmse}/{N_SHUFFLE} (p={p_rmse:.4f})")
    print(f"노이즈가 실제와 같거나 더 좋은 MAE:  {n_beat_mae}/{N_SHUFFLE} (p={p_mae:.4f})")

    print("\n" + "=" * 100)
    print("=== 최종 판정 ===")
    print("=" * 100)
    print(f"15피처 하이브리드 대비 개선: {beats_15}")
    print(f"GARCH 대비 개선: {beats_garch}")
    print(f"SMA20 대비 개선: {beats_sma}")
    print(f"Parkinson-SMA20 대비 개선: {beats_park}")
    print(f"셔플(N=7) 대비 유의: {p_rmse < 0.05 and p_mae < 0.05} (p_rmse={p_rmse:.3f}, p_mae={p_mae:.3f})")


if __name__ == "__main__":
    main()
