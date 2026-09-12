# 가격예측_변동성_공통.py
# Task T 100종목 검증된 하이브리드(GARCH+Transformer) 변동성 예측 자동화 파이프라인 —
# 데이터 로드~피처 생성~학습~배포 게이트~다음 거래일 예측 공용 로직(2026-09-07 신규 작성).
#
# 배경: 결과_TaskT_변동성예측_최종.md의 100종목 검증(원본·종목평균 제거 지표 둘 다 GARCH·
# SMA20·Parkinson-SMA20 셋을 이김, N=5 셔플 격차 90~120σ)을 그대로 운영 파이프라인으로
# 옮긴다. 하이퍼파라미터·게이트 기준의 정본은 가격예측/test_evaluation_pooled100_hybrid.py다
# — 이 파일의 상수들은 그 검증 스크립트 값을 그대로 가져온 것이며, 검증 재현성을 위해
# 임의로 바꾸지 않는다.
#
# 방향예측 파이프라인(쓰래기통/가격예측_공통.py, 가격예측_통합모델.py — 둘 다 폐기)과의 관계:
#   - "공통/일일수집" 파일 분리, run_isolated를 통한 종목별 예측 격리, next_weekday() 근사,
#     save_checkpoint 버전 태깅은 그대로 계승했다.
#   - pooled(전 종목 통합) 학습이라 종목별 격리가 원천적으로 불가능하다는 점도 가격예측_
#     통합모델.py의 설계를 그대로 따른다(학습 실패=전체 파이프라인 실패, 예측 단계만 격리).
#   - GARCH σ가 피처로 들어가는 하이브리드 구조라 baseline 3종(GARCH/SMA20/Parkinson-SMA20)을
#     매일 다시 계산해 배포 게이트 판정 + model_predictions 감사 컬럼에 채운다는 점이
#     방향예측 파이프라인과 다르다.
#
# GARCH 파라미터 캐싱(월 1회 재추정): split_dataset.build_merged_dataset_v2_volatility_hybrid의
# precomputed_sigma/garch_params 인자, split_dataset.build_next_day_merged_window_volatility_hybrid의
# garch_params 인자는 2026-09-06에 정확히 이 캐싱 용도로 이미 만들어져 있었다(당시엔 실제
# 호출부가 없었음 — 이 파일이 최초 호출부다). 종목당 GARCH MLE 적합(arch_model.fit())은
# 비용이 커서 매일 100종목분을 재추정하면 낭비이므로, (omega, alpha, beta)만 파일 캐시에
# 남기고 재귀적 σ 계산(forecast_with_fixed_params, 비용이 훨씬 작음)은 매일 최신 가격 이력
# 전체에 대해 새로 돌린다.

import json
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch
from arch import arch_model

from db_manager import get_db_connection
from 가격예측.garch_baseline import (
    compute_log_returns_pct, compute_sma_baseline, forecast_next_day_sigma,
    forecast_with_fixed_params, load_close_prices, rmse_mae,
)
from 가격예측.parkinson_baseline import compute_parkinson_vol_pct, load_high_low, PARK_WINDOW
from 가격예측.pooled_dataset import build_pooled_sequences, compute_global_split_dates
from 가격예측.sequence_dataset import FeatureScaler
from 가격예측.split_dataset import (
    build_merged_dataset_v2,
    build_merged_dataset_v2_volatility_hybrid,
    build_next_day_merged_window_volatility_hybrid,
)
from 가격예측.train_common import (
    evaluate_pooled_predictions,
    passes_deployment_gate_volatility,
    run_isolated,
    save_checkpoint,
    train_pooled_transformer,
)

LOOKBACK = 20

# test_evaluation_pooled100_hybrid.py(100종목 검증 정본)와 완전히 동일한 하이퍼파라미터 —
# 배포 모델이 검증된 설정과 달라지면 검증 결과가 더 이상 이 모델을 대변하지 못하므로 값을
# 절대 바꾸지 않고 그대로 가져온다.
D_MODEL = 32
NHEAD = 2
NUM_LAYERS = 2
DIM_FEEDFORWARD = 64
DROPOUT = 0.3
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
SMOKE_EPOCHS = 2
MAX_EPOCHS = 60
PATIENCE = 8
SEED = 42
EMBEDDING_DIM = 20  # test_evaluation_pooled100_hybrid.EMBEDDING_DIM_100과 동일값.
# ⚠️ 종목 수(현재 100)가 크게 달라지면(종목_월간갱신.py 누적 결과) 이 값의 적정성도 사람이
# 재검토해야 한다 — 자동으로 재조정되지 않는다.

# GARCH 파라미터를 이 일수 이내에 이미 재추정했으면 캐시를 그대로 재사용(월 1회 재추정 원칙,
# 위 헤더 설명 참고).
GARCH_REFIT_MAX_AGE_DAYS = 30

# 이 pooled 모델이 종목 하나를 학습에 포함시키기 위한 최소 train 시퀀스 수. 2026-09-07 100종목
# 확장 세션 실측 기준: 443060(HD현대마린솔루션, 실제 train 시퀀스 13개)은 그대로 포함해도
# 문제가 없었고, 0126Z0/064400(train 0개)은 애초에 종목 리스트에서 제외했다 — "13은 포함,
# 0은 제외"라는 그 경계 사이에서 여유를 두고 10으로 정했다. 종목_월간갱신.py의 신규 진입
# 게이팅이 이 상수를 그대로 가져다 쓴다(단일 출처 유지).
MIN_TRAIN_SEQUENCES = 10

_HERE = os.path.dirname(__file__)
CHECKPOINT_DIR = os.path.join(_HERE, "checkpoints", "pooled_volatility_hybrid")
# ⚠️ 8종목 시절 방향예측 pooled 모델의 checkpoints/pooled/ 와는 다른 별도 디렉터리다 — 같은
# 곳을 쓰면 model_class는 같지만(PooledTransformerRegressor) feature_cols(13 vs 15)와
# embedding_dim이 달라 load_checkpoint()가 latest.txt로 엉뚱한 모델을 집어올 위험이 있다.
GARCH_PARAM_CACHE_DIR = os.path.join(_HERE, "checkpoints", "garch_params")
LOG_DIR = os.path.join(_HERE, "logs")


def _model_kwargs(num_stocks):
    return dict(
        lookback=LOOKBACK, num_stocks=num_stocks, embedding_dim=EMBEDDING_DIM,
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT,
    )


# ── GARCH 파라미터 캐싱 ──────────────────────────────────────────────────────────

def _garch_cache_path(ticker):
    return os.path.join(GARCH_PARAM_CACHE_DIR, f"{ticker}.json")


def _load_garch_cache(ticker):
    path = _garch_cache_path(ticker)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_garch_cache(ticker, params, train_end):
    os.makedirs(GARCH_PARAM_CACHE_DIR, exist_ok=True)
    payload = {
        "params": {k: float(v) for k, v in params.items()},
        "fitted_at": date.today().isoformat(),
        "train_end": str(pd.Timestamp(train_end).date()),
    }
    with open(_garch_cache_path(ticker), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def get_or_fit_garch_params(ticker, returns, train_end, max_age_days=GARCH_REFIT_MAX_AGE_DAYS):
    """캐시가 있고 max_age_days 이내면 재사용, 없거나 오래됐으면 train_end까지 데이터로
    새로 MLE 적합해 캐시를 갱신한다. 반환: (params: pd.Series, refit_happened: bool)."""
    cached = _load_garch_cache(ticker)
    if cached is not None:
        fitted_at = date.fromisoformat(cached["fitted_at"])
        age_days = (date.today() - fitted_at).days
        if age_days <= max_age_days:
            return pd.Series(cached["params"]), False

    train_returns = returns[returns.index <= train_end]
    am_train = arch_model(train_returns.values, mean="Constant", vol="Garch", p=1, q=1, dist="normal")
    res_train = am_train.fit(disp="off")
    params = res_train.params
    _save_garch_cache(ticker, params, train_end)
    return params, True


def compute_full_period_sigma_cached(returns, params):
    """검증용/garch_baseline_check.compute_full_period_sigma()와 동일한 결과를 내되, 이미
    가진(캐시된) params로 재적합 없이 재귀 계산만 한다 — compute_full_period_sigma가
    내부적으로 하는 forecast_with_fixed_params(start_pos=0) 호출과 완전히 동일, fit() 단계만
    스킵한 것(2026-09-12: compute_full_period_sigma는 검증용/garch_baseline_check.py로 이동됨)."""
    return forecast_with_fixed_params(returns, params, 0)


# ── baseline 3종: val 구간 RMSE(게이트용) + 다음 거래일 점예측(감사 컬럼용) ──────────────

def compute_ticker_baselines(ticker, start_date, end_date, train_end, val_end):
    """종목 1개의 GARCH 파라미터를 캐시에서 가져오거나 새로 적합하고, 그 파라미터로:
      (a) 전체 구간 σ(하이브리드 피처용 — build_merged_dataset_v2_volatility_hybrid의
          precomputed_sigma로 그대로 재사용, 내부에서 다시 적합하지 않음)
      (b) val 구간 GARCH/SMA20/Parkinson-SMA20 RMSE(배포 게이트 판정용)
      (c) 다음 거래일 GARCH/SMA20/Parkinson-SMA20 baseline 점예측(model_predictions 감사 컬럼용)
    을 한 번에 계산한다 — 세 용도를 따로 호출하면 종목당 GARCH 적합/재귀계산을 최대 3번
    반복하게 되어 여기서 합쳤다.

    반환 dict: params, sigma_full, val_rmse(garch/sma20/parkinson dict), n_val_valid,
    next_day(garch/sma20/parkinson dict)."""
    close = load_close_prices(ticker, start_date, end_date)
    returns = compute_log_returns_pct(close)
    params, refit = get_or_fit_garch_params(ticker, returns, train_end)
    sigma_full = compute_full_period_sigma_cached(returns, params)

    sma_full = compute_sma_baseline(returns)
    hl = load_high_low(ticker, start_date, end_date)
    park_raw = compute_parkinson_vol_pct(hl["high"], hl["low"])
    park_ma20 = park_raw.rolling(PARK_WINDOW, min_periods=PARK_WINDOW).mean().shift(1)

    val_mask = (returns.index > train_end) & (returns.index <= val_end)
    realized_val = returns.abs()[val_mask]
    sigma_val = sigma_full.reindex(realized_val.index)
    sma_val = sma_full.reindex(realized_val.index)
    park_val = park_ma20.reindex(realized_val.index)
    valid = sigma_val.notna() & sma_val.notna() & park_val.notna()

    garch_rmse, _ = rmse_mae(sigma_val[valid].values, realized_val[valid].values)
    sma_rmse, _ = rmse_mae(sma_val[valid].values, realized_val[valid].values)
    park_rmse, _ = rmse_mae(park_val[valid].values, realized_val[valid].values)

    next_garch = forecast_next_day_sigma(returns, params)
    next_sma = float(returns.abs().tail(PARK_WINDOW).mean())
    next_park = float(park_raw.tail(PARK_WINDOW).mean())

    return {
        "params": params, "sigma_full": sigma_full, "refit": refit,
        "val_rmse": {"garch": garch_rmse, "sma20": sma_rmse, "parkinson": park_rmse},
        "n_val_valid": int(valid.sum()),
        "next_day": {"garch": next_garch, "sma20": next_sma, "parkinson": next_park},
    }


# ── pooled 하이브리드 학습 ────────────────────────────────────────────────────────

def train_daily_pooled_model(tickers, start_date, end_date, seed=SEED):
    """전 종목을 하나의 pooled 데이터셋으로 묶어 학습한다 — "매일 전체 재학습" 원칙(쓰래기통/
    가격예측_공통.py의 run_daily_pipeline 주석과 동일 근거: 파라미터 규모가 작아 재학습
    자체가 저렴하고, 조건부 재학습 트리거는 새로운 버그 지점만 늘린다).

    반환: model, scaler, feature_cols, ticker_to_id, device, gate, baselines_by_ticker,
    train_end/val_end, n_train/n_val, n_train_by_ticker."""
    ref, _ = build_merged_dataset_v2(tickers[0], start_date, end_date)
    train_end, val_end = compute_global_split_dates(ref.index)

    baselines_by_ticker = {}
    for ticker in tickers:
        baselines_by_ticker[ticker] = compute_ticker_baselines(ticker, start_date, end_date, train_end, val_end)

    def hybrid_build_fn(ticker, s, e):
        b = baselines_by_ticker[ticker]
        return build_merged_dataset_v2_volatility_hybrid(
            ticker, s, e, train_end, precomputed_sigma=b["sigma_full"], garch_params=b["params"],
        )

    splits, feature_cols, ticker_to_id, (train_end2, val_end2) = build_pooled_sequences(
        tickers, start_date, end_date, lookback=LOOKBACK, build_fn=hybrid_build_fn,
    )
    assert train_end2 == train_end and val_end2 == val_end
    X_train, y_train, tid_train, d_train, _ = splits["train"]
    X_val, y_val, tid_val, d_val, _ = splits["val"]

    n_train_by_ticker = {t: int((tid_train == tid).sum()) for t, tid in ticker_to_id.items()}
    thin = {t: n for t, n in n_train_by_ticker.items() if n < MIN_TRAIN_SEQUENCES}
    if thin:
        print(f"⚠️ train 시퀀스가 {MIN_TRAIN_SEQUENCES}개 미만인 종목(임베딩이 사실상 학습 안 될 위험): {thin}")
        print("   — 종목_월간갱신.py 게이팅을 통과한 종목이라면 나타나면 안 되는 상태다. 재검토 필요.")

    scaler = FeatureScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = train_pooled_transformer(
        X_train_s, y_train, tid_train, X_val_s, y_val, tid_val,
        len(feature_cols), LOOKBACK, len(tickers), EMBEDDING_DIM,
        D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT,
        BATCH_SIZE, LR, WEIGHT_DECAY, SMOKE_EPOCHS, MAX_EPOCHS, PATIENCE,
        device, seed=seed, label="pooled-volatility-hybrid",
    )

    val_preds, val_actuals, val_tids = evaluate_pooled_predictions(result["model"], result["val_loader"], device)
    hybrid_val_rmse, _ = rmse_mae(val_preds, val_actuals)

    avg_garch_rmse = float(np.mean([b["val_rmse"]["garch"] for b in baselines_by_ticker.values()]))
    avg_sma_rmse = float(np.mean([b["val_rmse"]["sma20"] for b in baselines_by_ticker.values()]))
    avg_park_rmse = float(np.mean([b["val_rmse"]["parkinson"] for b in baselines_by_ticker.values()]))

    gate = passes_deployment_gate_volatility(hybrid_val_rmse, avg_garch_rmse, avg_sma_rmse, avg_park_rmse)

    return {
        "model": result["model"], "scaler": scaler, "feature_cols": feature_cols,
        "ticker_to_id": ticker_to_id, "device": device, "gate": gate,
        "baselines_by_ticker": baselines_by_ticker,
        "train_end": train_end, "val_end": val_end,
        "n_train": len(X_train), "n_val": len(X_val),
        "n_train_by_ticker": n_train_by_ticker,
    }


# ── 다음 거래일 예측 + 저장 ───────────────────────────────────────────────────────

def next_weekday(d):
    """쓰래기통/가격예측_공통.py·가격예측_통합모델.py에서 쓰던 근사를 값 변경 없이 그대로
    가져옴 — 실제 거래소 캘린더가 없어 공휴일은 반영 못 한다. target_date가 실제 거래일과
    어긋나면 actual_volatility 백필 시 그냥 매치되는 실제 거래일이 없어 NULL로 남는다
    (조용히 틀린 값이 채워지는 것보다 안전한 실패 모드)."""
    wd = d.weekday()  # Mon=0 .. Sun=6
    if wd == 4:  # 금요일 -> 월요일
        return d + timedelta(days=3)
    if wd == 5:  # 토요일 -> 월요일
        return d + timedelta(days=2)
    return d + timedelta(days=1)


def _trailing_predicted_volatility_mean(ticker, before_target_date, n=60):
    """model_predictions에서 이 종목의 직전 n개 target_date(< before_target_date)
    predicted_volatility 평균을 가져온다. is_early_warning 계산용.

    ⚠️ 이 규칙(예측σ > 직전60일 평균의 1.5배) 자체의 조기경보 "능력"은 결과_변동성_조기경보_
    검증.md 원 검증이 GARCH 라벨링 버그로 무효화된 뒤 재검증에서 우연 수준으로 확인됐다
    (CLAUDE.md 참고). 여기서 채우는 is_early_warning은 감사/기록용 descriptive 값일 뿐,
    검증된 운영 신호로 취급하면 안 된다."""
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패 — _trailing_predicted_volatility_mean")
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT predicted_volatility FROM model_predictions
                WHERE ticker = %s AND target_date < %s
                ORDER BY target_date DESC LIMIT %s
                """,
                (ticker, before_target_date, n),
            )
            rows = cur.fetchall()
    if len(rows) < 10:  # 이력이 너무 적으면 판단을 보류(과거 데이터 부족 -> 조기경보 False로 안전하게)
        return None
    return float(np.mean([r[0] for r in rows]))


def save_prediction(ticker, target_date, prediction_date, predicted_volatility,
                     garch_baseline, sma20_baseline, parkinson_sma20_baseline,
                     is_early_warning, model_version, gate):
    """model_predictions에 UPSERT((ticker, target_date) 기준) — 재실행 시 최신 값으로 덮어씀."""
    query = """
        INSERT INTO model_predictions
            (ticker, target_date, prediction_date, predicted_volatility,
             garch_baseline, sma20_baseline, parkinson_sma20_baseline,
             is_early_warning, model_version,
             gate_passed, gate_vs_garch, gate_vs_sma20, gate_vs_parkinson)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker, target_date) DO UPDATE
            SET prediction_date = EXCLUDED.prediction_date,
                predicted_volatility = EXCLUDED.predicted_volatility,
                garch_baseline = EXCLUDED.garch_baseline,
                sma20_baseline = EXCLUDED.sma20_baseline,
                parkinson_sma20_baseline = EXCLUDED.parkinson_sma20_baseline,
                is_early_warning = EXCLUDED.is_early_warning,
                model_version = EXCLUDED.model_version,
                gate_passed = EXCLUDED.gate_passed,
                gate_vs_garch = EXCLUDED.gate_vs_garch,
                gate_vs_sma20 = EXCLUDED.gate_vs_sma20,
                gate_vs_parkinson = EXCLUDED.gate_vs_parkinson
    """
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패 — save_prediction")
        with conn.cursor() as cur:
            cur.execute(query, (
                ticker, target_date, prediction_date, float(predicted_volatility),
                float(garch_baseline), float(sma20_baseline), float(parkinson_sma20_baseline),
                bool(is_early_warning), model_version,
                bool(gate["passed"]), bool(gate["beats_garch"]), bool(gate["beats_sma"]), bool(gate["beats_parkinson"]),
            ))
        conn.commit()


def predict_and_save_for_ticker(ticker, start_date, end_date, model, scaler, feature_cols,
                                 ticker_to_id, device, version, gate, baselines_by_ticker):
    """pooled 하이브리드 모델로 한 종목의 다음 거래일 변동성을 예측하고 model_predictions에
    저장한다. run_isolated로 감싸 호출하는 것을 전제로 예외를 그대로 던진다(격리 책임은
    호출부에 있음)."""
    params = baselines_by_ticker[ticker]["params"]
    window_df, last_confirmed_date = build_next_day_merged_window_volatility_hybrid(
        ticker, start_date, end_date, LOOKBACK, params,
    )
    X_next = window_df[feature_cols].values[np.newaxis, :, :].astype(np.float32)
    X_next_s = scaler.transform(X_next)
    tid = np.array([ticker_to_id[ticker]], dtype=np.int64)

    model.eval()
    with torch.no_grad():
        pred = model(
            torch.from_numpy(X_next_s).to(device), torch.from_numpy(tid).to(device)
        ).cpu().numpy()[0]

    prediction_date = pd.Timestamp(last_confirmed_date).date()
    target_date = next_weekday(prediction_date)

    trailing_mean = _trailing_predicted_volatility_mean(ticker, target_date)
    is_early_warning = bool(trailing_mean is not None and float(pred) > 1.5 * trailing_mean)

    next_day = baselines_by_ticker[ticker]["next_day"]
    save_prediction(
        ticker, target_date, prediction_date, float(pred),
        next_day["garch"], next_day["sma20"], next_day["parkinson"],
        is_early_warning, version, gate,
    )

    return {
        "predicted_volatility": float(pred),
        "garch_baseline": next_day["garch"], "sma20_baseline": next_day["sma20"],
        "parkinson_sma20_baseline": next_day["parkinson"],
        "is_early_warning": is_early_warning,
        "prediction_date": str(prediction_date), "target_date": str(target_date),
    }


def run_pooled_volatility_pipeline(tickers, start_date, end_date):
    """전체 파이프라인: baseline 3종 계산(GARCH 캐시 포함) -> pooled 하이브리드 학습(전 종목
    통합) -> 체크포인트 저장 -> 종목별 다음 거래일 예측(run_isolated로 종목별 격리) ->
    model_predictions 저장.

    학습 자체는 격리하지 않는다(가격예측_통합모델.py와 동일 근거 — 모델이 하나라 학습 실패는
    전체 실패, 단일 종목 파이프라인처럼 "한 종목만 실패, 나머지는 계속"이 구조적으로 불가능).
    예측 단계만 종목별로 run_isolated로 감싼다."""
    train_out = train_daily_pooled_model(tickers, start_date, end_date)
    model, scaler, feature_cols = train_out["model"], train_out["scaler"], train_out["feature_cols"]
    ticker_to_id, device, gate = train_out["ticker_to_id"], train_out["device"], train_out["gate"]
    baselines_by_ticker = train_out["baselines_by_ticker"]

    version = save_checkpoint(
        model, scaler, feature_cols, _model_kwargs(len(tickers)), CHECKPOINT_DIR,
        model_class_name="PooledTransformerRegressor",
        extra_meta={
            "tickers": tickers, "ticker_to_id": ticker_to_id,
            "gate": {k: v for k, v in gate.items()},
            "n_train": train_out["n_train"], "n_val": train_out["n_val"],
            "n_train_by_ticker": train_out["n_train_by_ticker"],
            "train_start": start_date, "train_end": end_date,
            "split_train_end": str(train_out["train_end"].date()),
            "split_val_end": str(train_out["val_end"].date()),
        },
    )

    results = {}
    for ticker in tickers:
        r = run_isolated(
            predict_and_save_for_ticker, ticker, start_date, end_date,
            model, scaler, feature_cols, ticker_to_id, device, version, gate, baselines_by_ticker,
            label=ticker, log_dir=LOG_DIR,
        )
        results[ticker] = r["result"] if r["status"] == "success" else f"실패 ({r['error']})"

    gate_str = "통과" if gate["passed"] else "미통과"
    print(f"\n✅ pooled 하이브리드 변동성 모델 학습 완료(버전 {version}, "
          f"train={train_out['n_train']}/val={train_out['n_val']}) — 배포 게이트 {gate_str}")
    print(f"   val RMSE: hybrid={gate['hybrid_rmse']:.4f}  GARCH={gate['garch_rmse']:.4f}  "
          f"SMA20={gate['sma_rmse']:.4f}  Parkinson-SMA20={gate['parkinson_rmse']:.4f}")
    print(f"   개별 판정: vs GARCH={gate['beats_garch']}  vs SMA20={gate['beats_sma']}  "
          f"vs Parkinson-SMA20={gate['beats_parkinson']}")

    n_failed = sum(1 for v in results.values() if isinstance(v, str) and v.startswith("실패"))
    print(f"\n=== 종목별 예측 결과 ({len(tickers) - n_failed}/{len(tickers)} 성공) ===")
    for ticker, r in results.items():
        print(f"  {ticker}: {r}")
    if n_failed:
        print(f"⚠️ {n_failed}개 종목 예측 실패 — 가격예측/logs/ 에서 상세 로그를 확인하세요.")

    return {"version": version, "gate": gate, "predictions": results}
