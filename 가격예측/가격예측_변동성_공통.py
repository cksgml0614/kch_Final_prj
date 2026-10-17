# 가격예측_변동성_공통.py
# 하이브리드(GARCH σ 피처 + pooled LSTM) 변동성 예측 일일 파이프라인의 오케스트레이터와 설정(2026-09-07 신규 작성).
#
# 단계(2026-10-17 정리): 수집(주가데이터/·시장지표/, 이 파일 밖) -> 입력 점검(이 파일: 가격 정렬·지수 신선도)
# -> 피처(dataset_builder·momentum_feature·split_dataset, GARCH σ·baseline은 이 파일의 compute_ticker_baselines)
# -> 학습(train_common.train_model) -> 평가(evaluate.evaluate_gate) -> 저장+예측(store: 체크포인트·registry·
# model_predictions·MLflow). run_pooled_volatility_pipeline은 이 단계를 순서대로 부르는 얇은 함수다.
#
# 배경: pre-cleanup 태그 시점의 결과_TaskT_변동성예측_최종.md의 100종목 검증(원본·종목평균 제거 지표 둘 다 GARCH·
# SMA20·Parkinson-SMA20 셋을 이김, N=5 셔플 격차 90~120σ)을 그대로 운영 파이프라인으로
# 옮겼다(검증 정본은 pre-cleanup 태그 시점의 가격예측/검증용/test_evaluation_pooled100_hybrid.py).
#
# 방향예측 파이프라인(pre-cleanup 태그 시점의 가격예측_공통.py·가격예측_통합모델.py — 둘 다 폐기)과의 관계:
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
# 남기고 재귀적 σ 계산(garch_sigma_series, 비용이 훨씬 작음)은 매일 최신 가격 이력
# 전체에 대해 새로 돌린다.

import json
import os
import time
from datetime import date

import numpy as np
import pandas as pd
import torch
from arch import arch_model

from constants import MIN_TRAIN_SEQUENCES
from db_manager import get_db_connection
from 가격예측 import store
from 가격예측.evaluate import evaluate_gate
from 가격예측.garch_baseline import (
    SMA_WINDOW, compute_log_returns_pct, compute_sma_baseline, forecast_next_day_sigma,
    garch_sigma_series, load_close_prices, rmse_mae,
)
from 가격예측.momentum_feature import load_true_kospi
from 가격예측.parkinson_baseline import compute_parkinson_vol_pct, load_high_low, PARK_WINDOW
from 가격예측.pooled_dataset import build_pooled_sequences, compute_global_split_dates
from 가격예측.sequence_dataset import FeatureScaler
from 가격예측.split_dataset import (
    build_merged_dataset_v2_volatility_hybrid,
    valid_feature_dates,
)
from 가격예측.model import MODEL_CLASS_NAME
from 가격예측.train_common import TrainConfig, train_model
from 시장지표.feature_loader import load_indicator_cache, load_price_cache

LOOKBACK = 20

# 학습 하이퍼파라미터는 train_common.TrainConfig(2026-10-17, 모듈 상수에서 설정 객체로). 기본값은 운영값 그대로다
# (hidden 32·2층·dropout 0.3·embedding 20·batch 32·lr 1e-3·wd 1e-4·smoke 2·max 60·patience 8·seed 42).
# 쓰지 않던 Transformer 전용 값(NHEAD=2, DIM_FEEDFORWARD=64)은 지웠다.
# ⚠️ 종목 수(현재 100)가 크게 달라지면(종목_월간갱신.py 누적 결과) embedding_dim의 적정성도 사람이
# 재검토해야 한다 — 자동으로 재조정되지 않는다.
DEFAULT_TRAIN_CONFIG = TrainConfig()

# GARCH 파라미터를 이 일수 이내에 이미 재추정했으면 캐시를 그대로 재사용(월 1회 재추정 원칙,
# 위 헤더 설명 참고).
GARCH_REFIT_MAX_AGE_DAYS = 30


# MIN_TRAIN_SEQUENCES(최소 train 시퀀스 수)는 2026-10-17부터 constants.py에 있다 — 종목_월간갱신.py가 이 상수
# 하나 때문에 이 모듈(torch·mlflow)을 import하지 않게 하려고 옮겼다. 값과 근거는 constants.py 참고.

_HERE = os.path.dirname(__file__)
CHECKPOINT_DIR = os.path.join(_HERE, "checkpoints", "pooled_volatility_hybrid")
# (8종목 시절 방향예측 pooled 모델의 checkpoints/pooled/ 와는 다른 디렉터리다. 운영 경로는 run_pooled_volatility_pipeline의
# checkpoint_dir·garch_cache_dir·log_dir 인자로 바꿀 수 있다 — 기본값이 아래 세 상수다.)
GARCH_PARAM_CACHE_DIR = os.path.join(_HERE, "checkpoints", "garch_params")
LOG_DIR = os.path.join(_HERE, "logs")


def _model_kwargs(num_stocks, config):
    # 체크포인트 meta의 model_kwargs — PooledLSTMRegressor 생성자가 받는 인자만 담는다(2026-09-27).
    return dict(
        lookback=LOOKBACK, num_stocks=num_stocks, embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout,
    )


# ── GARCH 파라미터 캐싱 ──────────────────────────────────────────────────────────

def _garch_cache_path(ticker, cache_dir=None):
    return os.path.join(cache_dir or GARCH_PARAM_CACHE_DIR, f"{ticker}.json")


def _load_garch_cache(ticker, cache_dir=None):
    path = _garch_cache_path(ticker, cache_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_garch_cache(ticker, params, train_end, cache_dir=None):
    os.makedirs(cache_dir or GARCH_PARAM_CACHE_DIR, exist_ok=True)
    payload = {
        "params": {k: float(v) for k, v in params.items()},
        "fitted_at": date.today().isoformat(),
        "train_end": str(pd.Timestamp(train_end).date()),
    }
    with open(_garch_cache_path(ticker, cache_dir), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def get_or_fit_garch_params(ticker, returns, train_end, max_age_days=GARCH_REFIT_MAX_AGE_DAYS, cache_dir=None,
                            read_only=False):
    """캐시가 있고 max_age_days 이내면 재사용, 없거나 오래됐으면 train_end까지 데이터로
    새로 MLE 적합해 캐시를 갱신한다. 반환: (params: pd.Series, refit_happened: bool).
    cache_dir(2026-10-17): 기본 GARCH_PARAM_CACHE_DIR. read_only=True면 캐시가 없거나 만료됐을 때 적합·저장하지 않고
    예외를 낸다(같은 params로 비교해야 하는 점검·백테스트용). ⚠️ 캐시 재사용은 fitted_at만 보고 train_end는 보지 않는다
    (CLAUDE.md "GARCH 파라미터 캐시의 시점 무결성 문제")."""
    cached = _load_garch_cache(ticker, cache_dir)
    if cached is not None:
        fitted_at = date.fromisoformat(cached["fitted_at"])
        age_days = (date.today() - fitted_at).days
        if age_days <= max_age_days:
            return pd.Series(cached["params"]), False
    if read_only:
        raise RuntimeError(f"[GARCH-READ-ONLY] {ticker}: 캐시가 없거나 {max_age_days}일 지나 만료됨 — 읽기 전용이라 적합하지 않음")

    train_returns = returns[returns.index <= train_end]
    am_train = arch_model(train_returns.values, mean="Constant", vol="Garch", p=1, q=1, dist="normal")
    # cov_type="classic" — 표준오차(샌드위치 공분산)는 안 쓰고 params 점추정치만 쓰므로
    # 기본값 "robust"의 추가 계산 비용을 뺀다(2026-09-20).
    res_train = am_train.fit(disp="off", cov_type="classic")
    params = res_train.params
    _save_garch_cache(ticker, params, train_end, cache_dir)
    return params, True


def compute_full_period_sigma_cached(returns, params):
    """이미 가진(캐시된) params로 재적합 없이 전체 구간 GARCH σ를 재귀 계산한다
    (garch_baseline.garch_sigma_series — 서빙 창과 같은 경로). 이름의 "cached"는 params가 캐시에서 온다는 뜻이다."""
    return garch_sigma_series(returns, params)


# ── baseline 3종: val 구간 RMSE(게이트용) + 다음 거래일 점예측(감사 컬럼용) ──────────────

def compute_ticker_baselines(ticker, start_date, end_date, train_end, val_end, prices=None, garch_cache_dir=None,
                             garch_read_only=False):
    """종목 1개의 GARCH 파라미터를 캐시에서 가져오거나 새로 적합하고, 그 파라미터로:
      (a) 전체 구간 σ(하이브리드 피처용 — build_merged_dataset_v2_volatility_hybrid의
          precomputed_sigma로 그대로 재사용, 내부에서 다시 적합하지 않음)
      (b) val 구간 GARCH/SMA20/Parkinson-SMA20 RMSE(배포 게이트 판정용)
      (c) 다음 거래일 GARCH/SMA20/Parkinson-SMA20 baseline 점예측(model_predictions 감사 컬럼용)
    을 한 번에 계산한다 — 세 용도를 따로 호출하면 종목당 GARCH 적합/재귀계산을 최대 3번
    반복하게 되어 여기서 합쳤다.

    prices(선택, 2026-10-05): 시장지표.feature_loader.load_price_cache(...)[ticker]. 주면
    load_close_prices/load_high_low 대신 여기서 close·high·low를 꺼낸다 — 두 함수와 같은 테이블·
    같은 구간(ticker, date BETWEEN start_date AND end_date, ORDER BY date)·같은 float 변환이다.

    garch_cache_dir/garch_read_only(2026-10-17): get_or_fit_garch_params에 그대로 전달.

    반환 dict: params, sigma_full, val_rmse(garch/sma20/parkinson dict), n_val_valid,
    next_day(garch/sma20/parkinson dict)."""
    if prices is not None:
        prices_by_date = prices.set_index("date")
        close = prices_by_date["close"]
        hl = prices_by_date[["high", "low"]]
    else:
        close = load_close_prices(ticker, start_date, end_date)
        hl = load_high_low(ticker, start_date, end_date)
    returns = compute_log_returns_pct(close)
    params, refit = get_or_fit_garch_params(ticker, returns, train_end, cache_dir=garch_cache_dir,
                                            read_only=garch_read_only)
    sigma_full = compute_full_period_sigma_cached(returns, params)

    sma_full = compute_sma_baseline(returns)
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
    next_sma = float(returns.abs().tail(SMA_WINDOW).mean())
    next_park = float(park_raw.tail(PARK_WINDOW).mean())

    return {
        "params": params, "sigma_full": sigma_full, "refit": refit,
        "val_rmse": {"garch": garch_rmse, "sma20": sma_rmse, "parkinson": park_rmse},
        "n_val_valid": int(valid.sum()),
        "next_day": {"garch": next_garch, "sma20": next_sma, "parkinson": next_park},
    }


# ── pooled 하이브리드 학습 ────────────────────────────────────────────────────────

def load_shared_cache(start_date, end_date):
    """종목과 무관한 거시지표 9종·KOSPI 원자료를 한 번 읽어 학습·예측이 함께 쓰게 한다(2026-10-05,
    Neon 전송량 절감). 반환 dict의 키는 split_dataset 피처 함수들의 키워드 인자 이름과 같다."""
    return {
        "precomputed_indicators": load_indicator_cache(end_date, start_date),
        "true_kospi": load_true_kospi(start_date, end_date),
    }


def compute_split_dates(ref_ticker, start_date, end_date, shared_cache, prices):
    """전역 train/val 분할 경계(2026-10-17 분리). 기준 종목(운영: ACTIVE_TICKERS[0])의 유효 피처 날짜
    (split_dataset.valid_feature_dates — GARCH σ 전 변동성 frame 인덱스)로 compute_global_split_dates를 부른다.
    예전에는 방향 예측 frame(build_merged_dataset_v2) 인덱스를 썼다(정리 골든 비교로 같은 경계 확인)."""
    dates = valid_feature_dates(ref_ticker, start_date, end_date, **shared_cache, precomputed_prices=prices)
    return compute_global_split_dates(dates)


def train_daily_pooled_model(tickers, start_date, end_date, train_config=DEFAULT_TRAIN_CONFIG, shared_cache=None,
                             prices_by_ticker=None, garch_cache_dir=None, garch_read_only=False):
    """전 종목을 하나의 pooled 데이터셋으로 묶어 학습한다 — "매일 전체 재학습" 원칙(파라미터 규모가
    작아 재학습 자체가 저렴하고, 조건부 재학습 트리거는 새로운 버그 지점만 늘린다).

    shared_cache(선택, 2026-10-05): load_shared_cache(start_date, end_date) 결과. None이면 여기서 로드한다.
    prices_by_ticker(선택, 2026-10-05): load_price_cache(tickers, start_date, end_date) 결과. None이면
    여기서 한 번에 로드한다 — 예전에는 종목마다 baseline용(종가, 고가·저가)과 피처용(OHLCV)으로
    같은 구간을 세 번 읽었다.

    반환: model, scaler, feature_cols, ticker_to_id, device, gate, baselines_by_ticker,
    train_end/val_end, n_train/n_val, n_train_by_ticker, history/best_epoch/best_val_loss/
    overfit_ratio_at_end(2026-09-14, MLflow 로깅용)."""

    if shared_cache is None:
        shared_cache = load_shared_cache(start_date, end_date)
    if prices_by_ticker is None:
        prices_by_ticker = load_price_cache(tickers, start_date, end_date)

    train_end, val_end = compute_split_dates(tickers[0], start_date, end_date, shared_cache,
                                             prices_by_ticker[tickers[0]])

    baselines_by_ticker = {}
    for ticker in tickers:
        baselines_by_ticker[ticker] = compute_ticker_baselines(
            ticker, start_date, end_date, train_end, val_end, prices=prices_by_ticker[ticker],
            garch_cache_dir=garch_cache_dir, garch_read_only=garch_read_only,
        )

    def hybrid_build_fn(ticker, s, e, precomputed_indicators=None, true_kospi=None):
        # precomputed_indicators/true_kospi(2026-09-20): pooled_dataset.build_pooled_sequences가
        # 이 시그니처를 보고(inspect) 거시지표 캐시를 종목 루프 시작 전에 한 번만 만들어 넘긴다
        # — 성능 최적화일 뿐 조인/계산 로직은 동일(값 변경 없음 검증 완료).
        b = baselines_by_ticker[ticker]
        return build_merged_dataset_v2_volatility_hybrid(
            ticker, s, e, precomputed_sigma=b["sigma_full"], garch_params=b["params"],
            precomputed_indicators=precomputed_indicators, true_kospi=true_kospi,
            precomputed_prices=prices_by_ticker[ticker],
        )

    splits, feature_cols, ticker_to_id, (train_end2, val_end2) = build_pooled_sequences(
        tickers, start_date, end_date, lookback=LOOKBACK, build_fn=hybrid_build_fn,
        shared_cache=shared_cache,
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
    result = train_model(
        X_train_s, y_train, tid_train, X_val_s, y_val, tid_val,
        len(tickers), LOOKBACK, train_config, device, label="pooled-volatility-hybrid",
    )

    # 평가 단계: 하이브리드 종목평균 RMSE vs baseline 3종(종목별 val RMSE 평균) — evaluate.evaluate_gate.
    gate = evaluate_gate(result["model"], result["val_loader"], device, baselines_by_ticker)

    return {
        "model": result["model"], "scaler": scaler, "feature_cols": feature_cols,
        "ticker_to_id": ticker_to_id, "device": device, "gate": gate,
        "baselines_by_ticker": baselines_by_ticker,
        "train_end": train_end, "val_end": val_end,
        "n_train": len(X_train), "n_val": len(X_val),
        "n_train_by_ticker": n_train_by_ticker,
        "history": result["history"], "best_epoch": result["best_epoch"],
        "best_val_loss": result["best_val_loss"],
        "overfit_ratio_at_end": result["overfit_ratio_at_end"],
        "train_config": train_config,
    }


# 주가지수(KOSPI/KOSDAQ) 신선도 점검(2026-09-27, 가시성 확보용 경고만 — 파이프라인은 멈추지 않음).
# 배경: FinanceDataReader의 KS11/KQ11은 KRX 실시간이 아니라 FDR 작성자가 운영하는 GitHub 캐시
# (FinanceData/fdr_krx_data_cache)에서 읽는데, 이 캐시의 지수 파일이 2026-09-17 이후 갱신되지 않아
# market_indicators의 KOSPI/KOSDAQ가 9/17에서 멈췄다(같은 기간 네이버 fchart에는 9/18~9/23 존재
# 확인). 결측이 있으면 (1) feature_loader의 asof 조인이 KOSPI/KOSDAQ 레벨 피처를 마지막 값으로 조용히
# forward-fill하고, (2) momentum_feature.load_true_kospi 경로는 reindex -> pct_change(fill_method=
# 'pad')로 결측일 KOSPI 수익률이 0이 되어 excess_return_z_lag1이 사실상 "종목 절대수익률 z"로
# 오염된다 — 둘 다 에러 없이 진행되므로 최소한 로그에 드러나게 한다.
INDEX_FRESHNESS_CODES = ("KOSPI", "KOSDAQ")


def check_market_index_freshness(end_date):
    """daily_stock_prices의 최신 거래일(<= end_date) 대비 KOSPI/KOSDAQ가 몇 거래일 뒤처졌는지
    경고로 출력한다. 반환: {indicator_code: 뒤처진 거래일 수}."""
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패 — check_market_index_freshness")
        with conn.cursor() as cur:
            cur.execute(
                "SELECT indicator_code, MAX(date) FROM market_indicators "
                "WHERE indicator_code = ANY(%s) AND date <= %s GROUP BY indicator_code",
                (list(INDEX_FRESHNESS_CODES), end_date),
            )
            latest = dict(cur.fetchall())
            lag = {}
            for code in INDEX_FRESHNESS_CODES:
                last = latest.get(code)
                cur.execute(
                    "SELECT COUNT(DISTINCT date), MAX(date) FROM daily_stock_prices "
                    "WHERE date <= %s AND (%s::date IS NULL OR date > %s::date)",
                    (end_date, last, last),
                )
                n_behind, stock_last = cur.fetchone()
                lag[code] = int(n_behind)
                if n_behind > 0:
                    print(f"⚠️ {code} 지수 결측: market_indicators 최신={last}, 주가 최신={stock_last} — "
                          f"{n_behind}거래일째 결측. 결측일 KOSPI/KOSDAQ 피처는 마지막 값으로 채워지고 "
                          f"excess_return_z_lag1은 KOSPI 수익률 0으로 계산되어 예측이 오염될 수 있음.")
    return lag


# 종목별 주가 최신일 정렬 점검(2026-10-17). 10/13 05:02 실행이 Actions 주가 수집(05:02:58~05:07:15)과
# 겹쳐 앞 12종목만 10/12 주가가 들어간 상태를 읽었고, 12종목은 prediction_date 10/12, 88종목은 10/08로
# 예측했다(88종목의 10/12 기준 예측은 영영 만들어지지 않음).
# 판정: 공통일 = 종목별 최신일(<= end_date)의 최빈값(동률이면 불일치). 이탈 종목 = 최신일이 공통일과
# 다르거나(앞서든 뒤처지든) 가격 행이 없는 종목. 이탈이 MAX_DEVIANT_TICKERS 이하면 대기 없이 진행하되 그
# 종목은 이번 실행에서 예측을 저장하지 않는다(학습 데이터는 그대로). 초과·동률이면 수집 진행 중일 수 있어
# PRICE_ALIGNMENT_RETRY_INTERVAL_SEC마다 PRICE_ALIGNMENT_RETRIES번 재조회하고, 끝까지 같으면 학습 전에 중단한다.
MAX_DEVIANT_TICKERS = 5
PRICE_ALIGNMENT_RETRIES = 3          # 첫 조회 뒤 재조회 횟수(최대 대기 = RETRIES x INTERVAL)
PRICE_ALIGNMENT_RETRY_INTERVAL_SEC = 300


def summarize_price_alignment(latest_by_ticker, tickers, max_deviant=MAX_DEVIANT_TICKERS):
    """종목별 최신일 딕셔너리를 판정한다(DB 없이 단위 확인 가능한 순수 함수).
    반환: {"ok": 진행 가능 여부, "common_date": 최빈 최신일(동률·전부 없음이면 None),
           "deviants": {이탈 종목: 최신일 또는 None}, "distribution": {날짜: 종목 수}, "missing": [행 없는 종목]}."""
    missing = [t for t in tickers if latest_by_ticker.get(t) is None]
    distribution = {}
    for t in tickers:
        d = latest_by_ticker.get(t)
        if d is not None:
            distribution[d] = distribution.get(d, 0) + 1
    top = max(distribution.values(), default=0)
    modes = [d for d, n in distribution.items() if n == top]
    common_date = modes[0] if len(modes) == 1 else None
    deviants = ({t: latest_by_ticker.get(t) for t in tickers if latest_by_ticker.get(t) != common_date}
                if common_date is not None else {})
    return {"ok": common_date is not None and len(deviants) <= max_deviant,
            "common_date": common_date, "deviants": deviants,
            "distribution": dict(sorted(distribution.items())), "missing": missing}


def _fetch_latest_price_dates(tickers, end_date):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ticker, MAX(date) FROM daily_stock_prices "
                "WHERE ticker = ANY(%s) AND date <= %s GROUP BY ticker",
                (list(tickers), end_date),
            )
            return dict(cur.fetchall())


def check_price_dates_aligned(tickers, end_date, retries=PRICE_ALIGNMENT_RETRIES,
                              interval_sec=PRICE_ALIGNMENT_RETRY_INTERVAL_SEC,
                              max_deviant=MAX_DEVIANT_TICKERS,
                              fetch=_fetch_latest_price_dates, sleep=time.sleep):
    """종목별 주가 최신일(<= end_date) 정렬을 점검한다(판정 규칙은 위 주석). 진행 가능하면
    {"common_date", "deviants"}를 돌려주고, 재조회 후에도 불가하면 RuntimeError를 던진다."""
    for attempt in range(retries + 1):
        summary = summarize_price_alignment(fetch(tickers, end_date), tickers, max_deviant)
        dist = ", ".join(f"{d}: {n}종목" for d, n in summary["distribution"].items())
        if summary["ok"]:
            deviants = summary["deviants"]
            print(f"[PRICE-ALIGN] 공통 최신일 {summary['common_date']} — "
                  f"{len(tickers) - len(deviants)}/{len(tickers)}종목 일치")
            if deviants:
                listing = ", ".join(f"{t}={d}" for t, d in deviants.items())
                print(f"[PRICE-ALIGN] ⚠️ 이탈 {len(deviants)}종목(허용 {max_deviant}) — 이번 실행에서 예측을 "
                      f"저장하지 않음(학습 데이터는 그대로): {listing}")
            return {"common_date": summary["common_date"], "deviants": deviants}
        reason = ("최빈 최신일 동률" if summary["common_date"] is None
                  else f"이탈 {len(summary['deviants'])}종목 > 허용 {max_deviant}")
        print(f"[PRICE-ALIGN] 주가 최신일 불일치 (조회 {attempt + 1}/{retries + 1}, {reason}) — "
              f"분포 {{{dist}}}, 가격 없음 {len(summary['missing'])}종목 {summary['missing'][:10]}")
        if attempt < retries:
            print(f"[PRICE-ALIGN] {interval_sec}초 뒤 재조회합니다(수집 진행 중일 수 있음).")
            sleep(interval_sec)
    raise RuntimeError(
        f"[PRICE-ALIGN] 재조회 {retries}회 후에도 종목별 주가 최신일 불일치({reason}) — 학습·예측·저장을 "
        f"하지 않고 중단합니다. 분포 {{{dist}}}, 가격 없음 {summary['missing']}"
    )


def run_pooled_volatility_pipeline(tickers, start_date, end_date, train_config=DEFAULT_TRAIN_CONFIG, sinks=None,
                                   checkpoint_dir=None, garch_cache_dir=None, garch_read_only=False, log_dir=None,
                                   version=None):
    """전체 파이프라인: 입력 점검 -> baseline 3종 계산(GARCH 캐시 포함) -> pooled 하이브리드 학습(전 종목
    통합) -> 배포 게이트 판정 -> 체크포인트 저장 -> 종목별 다음 거래일 예측(run_isolated로
    종목별 격리) -> 예측 저장(게이트 결과와 무관하게 항상 저장) -> 실행 기록.

    2026-10-17 단계 분리: 저장 대상은 sinks(store.Sink)로 바꿔 끼운다. 기본값 None = 운영
    [store.DbSink(), store.MlflowSink()](model_registry·model_predictions·MLflow "일일_자동화"). 저장 없이 돌리려면
    sinks=[store.CollectSink()], checkpoint_dir=<임시 폴더>, garch_read_only=True(+ garch_cache_dir)를 넘긴다.
    checkpoint_dir/garch_cache_dir/log_dir 기본값은 CHECKPOINT_DIR/GARCH_PARAM_CACHE_DIR/LOG_DIR,
    version 기본값은 체크포인트 저장 시각(UTC)이다.

    학습 자체는 격리하지 않는다(가격예측_통합모델.py와 동일 근거 — 모델이 하나라 학습 실패는
    전체 실패, 단일 종목 파이프라인처럼 "한 종목만 실패, 나머지는 계속"이 구조적으로 불가능).
    예측 단계만 종목별로 run_isolated로 감싼다.

    2026-09-14(D-1, A-1의 "저장 스킵" 결정 철회 — 같은 날 안에서 방침이 두 번 바뀜, 아래
    경과 참고): A-1(같은 날 오전)은 "gate['passed']가 False면 예측 루프 자체를 건너뛴다"는
    엄격한 방식을 채택했다 — 그전에는 gate_passed=False가 model_predictions에 기록만 될
    뿐 저장을 막지 않아 게이트가 사실상 장식이었기 때문. 이후 actual_volatility 백필로 A-1
    적용 *이전*에 저장됐던 100행을 실제 결과와 대조해보니 하이브리드가 세 baseline 전부에
    뒤졌음이 확인돼(pre-cleanup 태그 시점의 결과_TaskT_변동성예측_최종.md 및 아래 "actual_volatility 백필" 절
    참고) 게이트 강제의 방향성 자체는 지지됐다. 그런데 A-1 적용 이후 게이트가 연속
    미통과하면서 예측 계산 자체가 아예 안 일어나 "미래 예측이 실제로 얼마나 정확한지"를
    계속 검증할 데이터가 더 이상 쌓이지 않는 문제가 드러났다 — 게이트를 검증하려면 게이트가
    걸러낸 예측이라도 계속 쌓여야 하는데, 엄격 스킵이 그 검증 자체를 불가능하게 만드는
    모순이었다. **결정: model_predictions에는 게이트 통과 여부와 무관하게 항상 저장하고,
    이미 있던 gate_passed 컬럼으로 신뢰 가능 여부를 구분한다.** 새 테이블을 만들지 않고
    기존 model_predictions에 통합 — `gate_passed=true`만 보고 싶으면 아래 D-3에서 준비한
    `model_predictions_trusted` 뷰를 사용할 것(뷰 DDL만 준비됨, 실제 생성은 사람이 psql로
    직접 실행 — `Database/가격예측/뷰_생성.sql`). A-1 이전(2026-09-12)에 저장된 기존 100행의
    gate_passed 값은 소급 수정하지 않고 그대로 히스토리로 보존한다.

    A-2(예측값 NaN/inf/음수 sanity check)는 게이트와 무관한 별개 로직이라 이번 결정과
    무관하게 그대로 유지된다. 체크포인트 저장(학습 자체가 성공했다는 기록)도 게이트와
    무관하게 항상 수행하는 기존 동작 그대로."""
    if sinks is None:
        sinks = [store.DbSink(), store.MlflowSink()]
    checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
    log_dir = log_dir or LOG_DIR
    # 종목별 주가 최신일 정렬 점검(2026-10-17): 소수 이탈은 그 종목 저장만 빼고 진행, 다수 이탈은 학습 전에 중단.
    price_deviants = check_price_dates_aligned(tickers, end_date)["deviants"]
    # 지수 결측이 있으면 이번 실행의 예측 전부를 input_data_suspect=True로 기록한다(2026-09-27).
    input_data_suspect = any(n > 0 for n in check_market_index_freshness(end_date).values())
    # 거시지표·KOSPI·종목 가격은 실행 시작 시 한 번만 읽어 학습과 예측이 같은 스냅숏을 쓴다(2026-10-05).
    shared_cache = load_shared_cache(start_date, end_date)
    prices_by_ticker = load_price_cache(tickers, start_date, end_date)
    train_out = train_daily_pooled_model(tickers, start_date, end_date, train_config=train_config, shared_cache=shared_cache,
                                         prices_by_ticker=prices_by_ticker, garch_cache_dir=garch_cache_dir,
                                         garch_read_only=garch_read_only)
    model, scaler, feature_cols = train_out["model"], train_out["scaler"], train_out["feature_cols"]
    ticker_to_id, device, gate = train_out["ticker_to_id"], train_out["device"], train_out["gate"]
    baselines_by_ticker = train_out["baselines_by_ticker"]

    model_kwargs = _model_kwargs(len(tickers), train_out["train_config"])
    extra_meta = {
        "tickers": tickers, "ticker_to_id": ticker_to_id,
        "gate": {k: v for k, v in gate.items()},
        "n_train": train_out["n_train"], "n_val": train_out["n_val"],
        "n_train_by_ticker": train_out["n_train_by_ticker"],
        "train_start": start_date, "train_end": end_date,
        "split_train_end": str(train_out["train_end"].date()),
        "split_val_end": str(train_out["val_end"].date()),
    }
    version = store.save_checkpoint(
        model, scaler, feature_cols, model_kwargs, checkpoint_dir, version=version,
        model_class_name=MODEL_CLASS_NAME, extra_meta=extra_meta,
    )
    ckpt = {"version": version, "checkpoint_dir": checkpoint_dir, "model": model, "scaler": scaler,
            "feature_cols": feature_cols, "model_kwargs": model_kwargs, "model_class": MODEL_CLASS_NAME,
            "extra_meta": extra_meta}
    for sink in sinks:
        sink.on_checkpoint(ckpt)

    gate_str = "통과" if gate["passed"] else "미통과"
    print(f"\n📊 pooled 하이브리드 변동성 모델 학습 완료(버전 {version}, "
          f"train={train_out['n_train']}/val={train_out['n_val']}) — 배포 게이트 {gate_str}")
    print(f"   val RMSE: hybrid={gate['hybrid_rmse']:.4f}  GARCH={gate['garch_rmse']:.4f}  "
          f"SMA20={gate['sma_rmse']:.4f}  Parkinson-SMA20={gate['parkinson_rmse']:.4f}")
    print(f"   개별 판정: vs GARCH={gate['beats_garch']}  vs SMA20={gate['beats_sma']}  "
          f"vs Parkinson-SMA20={gate['beats_parkinson']}")

    if not gate["passed"]:
        failed_against = [
            name for name, beat in (
                ("GARCH", gate["beats_garch"]),
                ("SMA20", gate["beats_sma"]),
                ("Parkinson-SMA20", gate["beats_parkinson"]),
            ) if not beat
        ]
        print(f"\n⚠️⚠️⚠️ 배포 게이트 미통과 — 다음 baseline을 이기지 못함: {', '.join(failed_against)} ⚠️⚠️⚠️")
        print("⚠️ 이번 예측은 저장되지만 gate_passed=False로 표시됩니다 — 참고용으로만 사용하세요.")

    results = {}
    for ticker in tickers:
        if ticker in price_deviants:
            results[ticker] = f"제외 (가격 정렬 이탈, 최신일 {price_deviants[ticker]})"
            continue
        r = store.run_isolated(
            store.predict_and_save_for_ticker, ticker, start_date, end_date, LOOKBACK,
            model, scaler, feature_cols, ticker_to_id, device, version, gate, baselines_by_ticker, sinks,
            input_data_suspect=input_data_suspect, shared_cache=shared_cache,
            prices=prices_by_ticker[ticker], label=ticker, log_dir=log_dir,
        )
        results[ticker] = r["result"] if r["status"] == "success" else f"실패 ({r['error']})"

    n_success, n_failed, n_excluded = store._count_prediction_results(results)
    print(f"\n=== 종목별 예측 결과 ({n_success}/{len(tickers)} 성공, 실패 {n_failed}, 제외 {n_excluded}) ===")
    for ticker, r in results.items():
        print(f"  {ticker}: {r}")
    if n_failed:
        print(f"⚠️ {n_failed}개 종목 예측 실패 — 가격예측/logs/ 에서 상세 로그를 확인하세요.")

    payload = store.build_run_payload(tickers, version, train_out, gate, results, model_kwargs,
                                      input_data_suspect=input_data_suspect)
    for sink in sinks:
        sink.log_run(payload)
    return {"version": version, "gate": gate, "predictions": results}
