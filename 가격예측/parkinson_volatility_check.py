# parkinson_volatility_check.py — Parkinson 변동성 추정량이 GARCH σ/SMA20/recent_vol_ma20보다
# 유용한 정보를 주는지 검증 (2026-09-06, 일회성 분석 — 자동화/운영 코드 아님).
#
# 배경: 학계 GARCH-신경망 하이브리드 연구는 대개 고빈도(5분봉 등) 실현변동성을 핵심 피처로
# 쓰는데, 이 프로젝트는 그 데이터가 없다. 대안으로, 일중 고가-저가 범위 기반 Parkinson(1980)
# 추정량이 종가 기반 변동성 추정보다 통계적으로 더 효율적(같은 정보량 대비 분산이 작음)이라는
# 것이 알려져 있다 — hl_range_ratio가 이미 피처에 있지만 "당일 원시값"으로만 쓰이고 있어,
# "직전 20일 Parkinson 변동성 평균"이라는 명시적 지속성 피처로 재가공해볼 가치가 있다.
#
# ⚠️ 리크 방지 설계: Parkinson 추정량 자체(parkinson_vol(t), high(t)/low(t) 사용)는 t일
# 자신의 장중 정보라 target(t)(=|종가수익률(t)|)을 예측하는 피처로 쓰면 리크다. 이 스크립트는
# 두 가지를 명확히 구분한다:
#   (a) "당일 동시성 상관관계" — parkinson_vol(t) vs target(t): Parkinson이 얼마나 좋은
#       "그날 자체의" 변동성 추정량인지 보는 순수 진단용. 예측 피처로 쓸 수 없는 값이다.
#   (b) "예측용 상관관계/RMSE" — GARCH σ(t)/SMA20(t)/recent_vol_ma20(t)와 동일하게, t-1까지의
#       Parkinson 값만으로 만든 시차 피처("Parkinson-SMA20", 직전 20일 Parkinson 평균을
#       shift(1)한 것)를 target(t)과 비교. 이것만이 실제로 16번째 피처 후보가 될 수 있다.

import sys
from datetime import date

import numpy as np
import pandas as pd
import torch

from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
from db_manager import get_db_connection
from 가격예측.garch_baseline import (
    compute_log_returns_pct, compute_sma_baseline, fit_and_forecast_garch,
    load_close_prices, rmse_mae,
)
from 가격예측.pooled_dataset import build_pooled_sequences, compute_global_split_dates
from 가격예측.dataset_builder import build_base_dataset_v2_volatility
from 가격예측.sequence_dataset import FeatureScaler
from 가격예측.split_dataset import build_merged_dataset_v2, build_merged_dataset_v2_volatility_hybrid
from 가격예측.train_common import evaluate_pooled_predictions, train_pooled_transformer
from 가격예측.가격예측_통합모델 import (
    BATCH_SIZE, DIM_FEEDFORWARD, DROPOUT, D_MODEL, EMBEDDING_DIM, LOOKBACK, LR,
    MAX_EPOCHS, NHEAD, NUM_LAYERS, PATIENCE, SEED, SMOKE_EPOCHS, WEIGHT_DECAY,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PARK_WINDOW = 20
N_SHUFFLE = 7
SHUFFLE_SEEDS = [0, 1, 7, 123, 2024, 777, 999]

# test_evaluation_pooled_volatility_hybrid.py 실제 실행 결과(2026-09-06, 라벨링 버그 수정 후) —
# 동일 종목/split/시드라 재실행하지 않고 그대로 인용.
HYBRID15_RESULTS = {
    "005930": (3.540248, 2.329369), "005380": (3.191598, 2.078374),
    "051910": (2.892820, 2.001431), "105560": (1.756112, 1.273551),
    "207940": (1.819776, 1.260878), "035420": (2.603423, 1.730854),
    "034730": (3.385972, 2.280838), "015760": (2.488303, 1.714924),
}
HYBRID15_AVG = (2.709781, 1.833777)
HYBRID15_SHUFFLE = {"rmse_mean": 2.950011, "rmse_std": 0.028239, "mae_mean": 1.904412, "mae_std": 0.008861}


def load_high_low(ticker, start_date, end_date):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT date, high, low FROM daily_stock_prices WHERE ticker=%s AND date BETWEEN %s AND %s ORDER BY date",
                (ticker, start_date, end_date),
            )
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["date", "high", "low"]).astype({"high": float, "low": float})
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def compute_parkinson_vol_pct(high, low):
    """parkinson_vol(t) = sqrt(1/(4 ln2) * ln(high/low)^2), x100(%) 스케일 통일.
    당일 H/L을 쓰므로 그 자체는 t일 정보 — 리크 방지 주석 참고."""
    raw = np.sqrt((np.log(high / low) ** 2) / (4 * np.log(2)))
    return raw * 100


def build_test_frame(ticker, train_end, val_end):
    """티커별 target/GARCH/SMA20/recent_vol_ma20/parkinson 계열을 한 데이터프레임에 정렬."""
    close = load_close_prices(ticker, STOCK_INITIAL_LOAD_START, end_date)
    returns = compute_log_returns_pct(close)
    hl = load_high_low(ticker, STOCK_INITIAL_LOAD_START, end_date)
    parkinson_raw = compute_parkinson_vol_pct(hl["high"], hl["low"])  # t일 자신의 정보(동시성 전용)

    sigma_pred, _ = fit_and_forecast_garch(returns, train_end, returns.index[returns.index > val_end][0])
    sma20 = compute_sma_baseline(returns)

    merged_vol, _ = build_base_dataset_v2_volatility(ticker, STOCK_INITIAL_LOAD_START, end_date)
    recent_vol_ma20 = merged_vol["recent_vol_ma20"] * 100  # 스케일 통일(x100)
    target = merged_vol["target"]

    # 예측용(leak-free) Parkinson-SMA20: t-1까지 20일 평균, recent_vol_ma20/SMA20과 동일한 관례
    parkinson_ma20 = parkinson_raw.rolling(PARK_WINDOW, min_periods=PARK_WINDOW).mean().shift(1)

    df = pd.DataFrame({
        "target": target,
        "parkinson_same_day": parkinson_raw,       # (a) 동시성 전용, 피처 불가
        "parkinson_ma20": parkinson_ma20,           # (b) 예측용(leak-free)
        "garch_sigma": sigma_pred,
        "sma20": sma20,
        "recent_vol_ma20": recent_vol_ma20,
    })
    return df.dropna(how="any")


def main():
    global end_date
    end_date = date.today().isoformat()

    reference_merged, _ = build_merged_dataset_v2(ACTIVE_TICKERS[0], STOCK_INITIAL_LOAD_START, end_date)
    train_end, val_end = compute_global_split_dates(reference_merged.index)
    print(f"전역 분할 경계: train_end={train_end.date()}  val_end={val_end.date()}")

    per_ticker = {}
    for ticker in ACTIVE_TICKERS:
        df = build_test_frame(ticker, train_end, val_end)
        test_df = df[df.index > val_end]
        per_ticker[ticker] = test_df

    # ── [1] 상관관계 비교 ──
    print("\n" + "=" * 110)
    print("=== [1] 상관계수 비교 (test 구간, target=|당일 수익률|) ===")
    print("=" * 110)
    print("(a) 동시성(당일 H/L, 리크) — Parkinson 자체가 얼마나 좋은 '그날' 변동성 추정량인지 참고용")
    print("(b) 예측용(t-1까지 정보) — 실제 다음날 예측에 쓸 수 있는 값들 간 공정 비교")
    print(f"\n{'ticker':<10}{'(a)parkinson_당일':>18}{'(b)parkinson_ma20':>18}{'(b)garch_sigma':>15}{'(b)sma20':>10}{'(b)recent_vol_ma20':>19}")
    corr_rows = []
    for ticker in ACTIVE_TICKERS:
        d = per_ticker[ticker]
        c_same = d["parkinson_same_day"].corr(d["target"])
        c_park = d["parkinson_ma20"].corr(d["target"])
        c_garch = d["garch_sigma"].corr(d["target"])
        c_sma = d["sma20"].corr(d["target"])
        c_recent = d["recent_vol_ma20"].corr(d["target"])
        corr_rows.append((ticker, c_same, c_park, c_garch, c_sma, c_recent))
        print(f"{ticker:<10}{c_same:>18.4f}{c_park:>18.4f}{c_garch:>15.4f}{c_sma:>10.4f}{c_recent:>19.4f}")

    avg = np.mean(np.array([r[1:] for r in corr_rows]), axis=0)
    print(f"{'평균(8종목)':<10}{avg[0]:>18.4f}{avg[1]:>18.4f}{avg[2]:>15.4f}{avg[3]:>10.4f}{avg[4]:>19.4f}")

    # ── [2] Parkinson-SMA20 vs GARCH vs SMA20 RMSE 비교 ──
    print("\n" + "=" * 110)
    print("=== [2] 예측 baseline RMSE/MAE 비교 (Parkinson-SMA20 신규 추가) ===")
    print("=" * 110)
    print(f"{'ticker':<10}{'n':>5}{'Park-SMA20 RMSE':>16}{'Park-SMA20 MAE':>15}{'GARCH RMSE':>12}{'GARCH MAE':>11}{'SMA20 RMSE':>11}{'SMA20 MAE':>10}")
    rmse_rows = []
    for ticker in ACTIVE_TICKERS:
        d = per_ticker[ticker]
        p_rmse, p_mae = rmse_mae(d["parkinson_ma20"].values, d["target"].values)
        g_rmse, g_mae = rmse_mae(d["garch_sigma"].values, d["target"].values)
        s_rmse, s_mae = rmse_mae(d["sma20"].values, d["target"].values)
        rmse_rows.append((ticker, len(d), p_rmse, p_mae, g_rmse, g_mae, s_rmse, s_mae))
        print(f"{ticker:<10}{len(d):>5}{p_rmse:>16.4f}{p_mae:>15.4f}{g_rmse:>12.4f}{g_mae:>11.4f}{s_rmse:>11.4f}{s_mae:>10.4f}")

    avg_p_rmse = np.mean([r[2] for r in rmse_rows]); avg_p_mae = np.mean([r[3] for r in rmse_rows])
    avg_g_rmse = np.mean([r[4] for r in rmse_rows]); avg_g_mae = np.mean([r[5] for r in rmse_rows])
    avg_s_rmse = np.mean([r[6] for r in rmse_rows]); avg_s_mae = np.mean([r[7] for r in rmse_rows])
    print(f"{'평균(8종목)':<10}{'-':>5}{avg_p_rmse:>16.4f}{avg_p_mae:>15.4f}{avg_g_rmse:>12.4f}{avg_g_mae:>11.4f}{avg_s_rmse:>11.4f}{avg_s_mae:>10.4f}")
    park_beats_sma = avg_p_rmse < avg_s_rmse and avg_p_mae < avg_s_mae
    park_beats_garch = avg_p_rmse < avg_g_rmse and avg_p_mae < avg_g_mae
    print(f"\nParkinson-SMA20이 SMA20을 이기는가: {park_beats_sma}")
    print(f"Parkinson-SMA20이 GARCH를 이기는가: {park_beats_garch}")

    # ── [3] 기존 피처와의 상관관계 (중복 정보 여부) ──
    print("\n" + "=" * 110)
    print("=== [3] Parkinson-SMA20과 기존 지속성 피처 간 상관관계 (중복도 확인) ===")
    print("=" * 110)
    for ticker in ACTIVE_TICKERS[:3] + ["평균(8종목)"]:
        pass
    corr_with_recent, corr_with_garch, corr_with_sma = [], [], []
    for ticker in ACTIVE_TICKERS:
        d = per_ticker[ticker]
        corr_with_recent.append(d["parkinson_ma20"].corr(d["recent_vol_ma20"]))
        corr_with_garch.append(d["parkinson_ma20"].corr(d["garch_sigma"]))
        corr_with_sma.append(d["parkinson_ma20"].corr(d["sma20"]))
    print(f"parkinson_ma20 vs recent_vol_ma20: 평균 상관계수 = {np.mean(corr_with_recent):.4f}")
    print(f"parkinson_ma20 vs garch_sigma    : 평균 상관계수 = {np.mean(corr_with_garch):.4f}")
    print(f"parkinson_ma20 vs sma20          : 평균 상관계수 = {np.mean(corr_with_sma):.4f}")

    redundant = np.mean(corr_with_recent) > 0.95
    print(f"\nrecent_vol_ma20과 거의 동일한 정보인가(상관 > 0.95): {redundant}")

    if redundant:
        print("\n>>> [4] 판정: recent_vol_ma20과 사실상 같은 정보 — 하이브리드 재학습 생략, 여기서 종료")
        return

    print("\n>>> [4-전] 판정: 기존 피처와 충분히 다른 정보로 판단 — 16번째 피처로 추가해 재학습 진행")

    # ── [4] 16번째 피처로 추가해 하이브리드 재학습 ──
    print("\n" + "=" * 110)
    print("=== [5] 16피처 하이브리드(GARCH σ + Parkinson-SMA20) 재학습 + test 평가 ===")
    print("=" * 110)

    def hybrid16_build_fn(ticker, s, e):
        merged, meta = build_merged_dataset_v2_volatility_hybrid(ticker, s, e, train_end)
        hl = load_high_low(ticker, s, e)
        parkinson_raw = compute_parkinson_vol_pct(hl["high"], hl["low"])
        parkinson_ma20 = parkinson_raw.rolling(PARK_WINDOW, min_periods=PARK_WINDOW).mean().shift(1)
        merged = merged.copy()
        merged["parkinson_ma20"] = parkinson_ma20.reindex(merged.index)
        before = len(merged)
        merged = merged.dropna(how="any")
        meta = dict(meta)
        meta["rows_before_parkinson_dropna"] = before
        meta["rows_after_parkinson_dropna"] = len(merged)
        return merged, meta

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    splits, feature_cols, ticker_to_id, (train_end2, val_end2) = build_pooled_sequences(
        ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START, end_date, lookback=LOOKBACK, build_fn=hybrid16_build_fn,
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

    result = train_once(y_train, "hybrid16-real", verbose=True)
    real_preds, real_actuals, real_tids = eval_loader(result["model"], X_test_s, y_test, tid_test)

    def per_ticker_rmse_mae(preds, actuals, tids):
        out = {}
        for ticker, tid in ticker_to_id.items():
            mask = tids == tid
            out[ticker] = rmse_mae(preds[mask], actuals[mask])
        return out, rmse_mae(preds, actuals)

    hybrid16_per_ticker, hybrid16_overall = per_ticker_rmse_mae(real_preds, real_actuals, real_tids)
    avg16_rmse = float(np.mean([hybrid16_per_ticker[t][0] for t in ACTIVE_TICKERS]))
    avg16_mae = float(np.mean([hybrid16_per_ticker[t][1] for t in ACTIVE_TICKERS]))

    print("\n" + "=" * 130)
    print("=== [6] 종합 비교 — 16피처 하이브리드 vs 15피처 하이브리드 vs GARCH vs SMA20 ===")
    print("=" * 130)
    header = f"{'ticker':<10}{'16피처RMSE':>12}{'16피처MAE':>11}{'15피처RMSE':>12}{'15피처MAE':>11}{'GARCH RMSE':>12}{'SMA20 RMSE':>11}"
    print(header)
    for ticker in ACTIVE_TICKERS:
        h16 = hybrid16_per_ticker[ticker]
        h15 = HYBRID15_RESULTS[ticker]
        g = rmse_rows[[r[0] for r in rmse_rows].index(ticker)]
        print(f"{ticker:<10}{h16[0]:>12.4f}{h16[1]:>11.4f}{h15[0]:>12.4f}{h15[1]:>11.4f}{g[4]:>12.4f}{g[6]:>11.4f}")
    print(f"{'평균':<10}{avg16_rmse:>12.4f}{avg16_mae:>11.4f}{HYBRID15_AVG[0]:>12.4f}{HYBRID15_AVG[1]:>11.4f}{avg_g_rmse:>12.4f}{avg_s_rmse:>11.4f}")

    improved_vs_15 = avg16_rmse < HYBRID15_AVG[0] and avg16_mae < HYBRID15_AVG[1]
    beats_garch16 = avg16_rmse < avg_g_rmse and avg16_mae < avg_g_mae
    beats_sma16 = avg16_rmse < avg_s_rmse and avg16_mae < avg_s_mae
    print(f"\n[핵심 확인 1] 16피처가 15피처(하이브리드)보다 개선됐는가: {improved_vs_15}")
    print(f"[핵심 확인 2] 16피처가 GARCH(1,1)를 넘어서는가: {beats_garch16}")
    print(f"[핵심 확인 3] 16피처가 SMA20을 넘어서는가: {beats_sma16}")

    # ── 셔플 테스트 N=7 ──
    print("\n" + "=" * 100)
    print(f"=== [7] 레이블 셔플 테스트 (N={N_SHUFFLE}) ===")
    print("=" * 100)
    shuffle_rmse, shuffle_mae = [], []
    for i, seed in enumerate(SHUFFLE_SEEDS):
        rng = np.random.RandomState(seed)
        y_shuf = y_train.copy()
        rng.shuffle(y_shuf)
        r = train_once(y_shuf, f"hybrid16-shuffle{i}", verbose=False)
        preds_s, actuals_s, _ = eval_loader(r["model"], X_test_s, y_test, tid_test)
        rmse_s, mae_s = rmse_mae(preds_s, actuals_s)
        shuffle_rmse.append(rmse_s); shuffle_mae.append(mae_s)
        print(f"  [{i+1}/{N_SHUFFLE}] seed={seed}: RMSE={rmse_s:.6f}  MAE={mae_s:.6f}  best_epoch={r['best_epoch']}")

    shuffle_rmse = np.array(shuffle_rmse); shuffle_mae = np.array(shuffle_mae)

    def _p(real_val, arr):
        beat = int(np.sum(arr <= real_val))
        return beat, (beat + 1) / (N_SHUFFLE + 1)

    n_beat_rmse, p_rmse = _p(hybrid16_overall[0], shuffle_rmse)
    n_beat_mae, p_mae = _p(hybrid16_overall[1], shuffle_mae)
    print(f"\n셔플 분포: RMSE {shuffle_rmse.mean():.6f}±{shuffle_rmse.std():.6f} | 실제 {hybrid16_overall[0]:.6f}")
    print(f"셔플 분포: MAE  {shuffle_mae.mean():.6f}±{shuffle_mae.std():.6f} | 실제 {hybrid16_overall[1]:.6f}")
    print(f"노이즈가 실제와 같거나 더 좋은 RMSE: {n_beat_rmse}/{N_SHUFFLE} (p={p_rmse:.4f})")
    print(f"노이즈가 실제와 같거나 더 좋은 MAE:  {n_beat_mae}/{N_SHUFFLE} (p={p_mae:.4f})")

    print("\n" + "=" * 100)
    print("=== [8] 최종 판정 ===")
    print("=" * 100)
    print(f"15피처 하이브리드 대비 개선: {improved_vs_15}")
    print(f"GARCH(1,1) 대비 개선: {beats_garch16}")
    print(f"SMA20 대비 개선: {beats_sma16}")
    print(f"셔플(N=7) 대비 유의: {p_rmse < 0.05 and p_mae < 0.05} (p_rmse={p_rmse:.3f}, p_mae={p_mae:.3f})")


if __name__ == "__main__":
    main()
