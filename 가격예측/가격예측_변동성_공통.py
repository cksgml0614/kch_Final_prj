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
# 남기고 재귀적 σ 계산(forecast_with_fixed_params, 비용이 훨씬 작음)은 매일 최신 가격 이력
# 전체에 대해 새로 돌린다.

import json
import os
import time
from datetime import date, timedelta

import mlflow
import numpy as np
import pandas as pd
import requests
import torch
from arch import arch_model

from config import redact
from constants import MIN_TRAIN_SEQUENCES
from db_manager import get_db_connection
from 가격예측.garch_baseline import (
    SMA_WINDOW, compute_log_returns_pct, compute_sma_baseline, forecast_next_day_sigma,
    forecast_with_fixed_params, load_close_prices, rmse_mae,
)
from 가격예측.momentum_feature import load_true_kospi
from 가격예측.parkinson_baseline import compute_parkinson_vol_pct, load_high_low, PARK_WINDOW
from 가격예측.pooled_dataset import build_pooled_sequences, compute_global_split_dates
from 가격예측.sequence_dataset import FeatureScaler
from 가격예측.split_dataset import (
    build_merged_dataset_v2_volatility_hybrid,
    build_next_day_merged_window_volatility_hybrid,
    valid_feature_dates,
)
from 가격예측.train_common import (
    evaluate_pooled_predictions,
    passes_deployment_gate_volatility,
    run_isolated,
    save_checkpoint,
    train_pooled_lstm,
)
from 시장지표.feature_loader import load_indicator_cache, load_price_cache

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

# ── 모델 계열 (2026-10-17: LSTM 단일화) ──
# 2026-09-27에 transformer -> lstm으로 전환했고(사람 결정, 지도교수님 피드백 "수치형 시계열은 수치형답게"),
# 2026-10-17 정리에서 Transformer 학습 코드와 MODEL_ARCHITECTURE 환경변수 스위치를 지웠다(git 태그 pre-cleanup).
# 아래 두 문자열은 체크포인트 meta.json의 model_class, model_registry.model_family/model_class, MLflow 태그
# model_family·파라미터 MODEL_ARCHITECTURE/model_class에 그대로 기록된다 — 바꾸지 말 것.
MODEL_FAMILY = "lstm"
MODEL_CLASS_NAME = "PooledLSTMRegressor"


# MIN_TRAIN_SEQUENCES(최소 train 시퀀스 수)는 2026-10-17부터 constants.py에 있다 — 종목_월간갱신.py가 이 상수
# 하나 때문에 이 모듈(torch·mlflow)을 import하지 않게 하려고 옮겼다. 값과 근거는 constants.py 참고.

_HERE = os.path.dirname(__file__)
CHECKPOINT_DIR = os.path.join(_HERE, "checkpoints", "pooled_volatility_hybrid")
# ⚠️ 8종목 시절 방향예측 pooled 모델의 checkpoints/pooled/ 와는 다른 별도 디렉터리다 — 같은
# 곳을 쓰면 model_class는 같지만(PooledTransformerRegressor) feature_cols(13 vs 15)와
# embedding_dim이 달라 load_checkpoint()가 latest.txt로 엉뚱한 모델을 집어올 위험이 있다.
GARCH_PARAM_CACHE_DIR = os.path.join(_HERE, "checkpoints", "garch_params")
LOG_DIR = os.path.join(_HERE, "logs")

# MLflow 일일 자동화 로깅 (2026-09-14 신규). 자동화 스크립트는 서버와 같은 컴퓨터에서
# 돌기 때문에 기본값은 127.0.0.1이다. 2026-10-11부터 서버를 127.0.0.1에만 바인딩하므로
# (mlflow_server_start.bat) 다른 기기에서는 UI에 접속할 수 없다 — 원격 조회가 필요하면 인증을
# 갖춘 터널을 따로 검토한다. 환경변수로 덮어쓸 수 있게 해 서버 위치가 바뀌어도 코드 수정 없이
# 대응 가능하게 함.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = "일일_자동화"
# 2026-10-11: 로깅 직전 서버 확인 타임아웃(초). 서버가 없거나 멈췄으면 기록을 건너뛴다(_log_mlflow_run).
MLFLOW_HEALTH_TIMEOUT_SEC = 3
# MLflow HTTP 재시도 상한(setdefault — 사람이 환경변수로 덮어쓸 수 있다). MLflow 기본값(재시도 7, 대기 계수 2,
# 타임아웃 120초)이면 서버가 없을 때 약 4분 24초(2026-10-11 실측), 멈췄을 때 약 20분(추정)을 기다린다.
MLFLOW_HTTP_LIMITS = {
    "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "3",
    "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR": "1",
    "MLFLOW_HTTP_REQUEST_TIMEOUT": "15",
}

# 2026-10-11: 체크포인트를 저장한 직후 그 version을 model_registry에 등록한다(_register_model_version).
# 일일 경로 전용 값이다. 백테스트를 다시 만들 때는 source를 인자로 받게 바꾼다(옛 백테스트_월간.py는 2026-10-17 삭제).
# 끄려면 환경변수 MODEL_REGISTRY_AUTO=0.
REGISTRY_SOURCE = "live"
REGISTRY_GATE_DEFINITION = "tickermean"
REGISTRY_MISSING_SHOW = 10  # [REGISTRY-MISSING] 줄에 보여줄 최대 version 수


def _register_model_version(version):
    """체크포인트 meta.json으로 model_registry에 1행을 INSERT ... ON CONFLICT DO NOTHING 한다(2026-10-11).
    이어서 로컬 체크포인트 version과 model_predictions의 version 중 registry에 없는 것을 찾아 경고한다
    (자동 보충 등록은 하지 않는다 — 시험 실행 체크포인트를 'live'로 잘못 넣을 수 있어 사람이 보고 넣는다).
    부가 기록이라 실패해도 예외를 던지지 않는다. 줄 맨 앞 [REGISTRY-SKIP]을 붙인 경고만 남기고 돌아간다.
    경고에는 예외 클래스명만 넣고(원문에 접속 정보가 섞일 수 있음), 출력 전에 redact를 거친다."""
    try:
        if os.environ.get("MODEL_REGISTRY_AUTO", "1") == "0":
            print(f"[REGISTRY-OFF] model_registry 자동 등록 꺼짐(MODEL_REGISTRY_AUTO=0) — version {version} 등록 안 함")
            return
        with open(os.path.join(CHECKPOINT_DIR, version, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        if meta["model_class"] != MODEL_CLASS_NAME:
            raise ValueError(f"예상하지 못한 model_class {meta['model_class']!r}")
        family = MODEL_FAMILY
        row = (version, family, meta["model_class"], "checkpoint_meta", REGISTRY_SOURCE, REGISTRY_GATE_DEFINITION,
               meta["train_end"], meta["split_train_end"], meta["split_val_end"], meta["n_train"])
        local_versions = {d for d in os.listdir(CHECKPOINT_DIR)
                          if os.path.isfile(os.path.join(CHECKPOINT_DIR, d, "meta.json"))}
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO model_registry (model_version, model_family, model_class, family_basis, source, "
                    "gate_definition, train_end, split_train_end, split_val_end, n_train) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (model_version) DO NOTHING", row)
                inserted = cur.rowcount
                cur.execute("SELECT model_version FROM model_registry")
                registered = {r[0] for r in cur.fetchall()}
                cur.execute("SELECT DISTINCT model_version FROM model_predictions")
                predicted = {r[0] for r in cur.fetchall()}
            conn.commit()
        print(f"[registry] {version} {'신규 등록' if inserted == 1 else '이미 등록돼 있음'}({family})")
        missing = sorted((local_versions | predicted) - registered)
        if missing:
            shown = ", ".join(missing[:REGISTRY_MISSING_SHOW]) + (" …" if len(missing) > REGISTRY_MISSING_SHOW else "")
            print(f"[REGISTRY-MISSING] ⚠️ model_registry에 없는 version {len(missing)}개: {shown}")
    except Exception as e:
        print(redact(f"[REGISTRY-SKIP] ⚠️ model_registry 등록 실패({type(e).__name__}) — 예측·저장은 계속된다. "
                     f"version {version}은 registry에 없으므로 수동 등록이 필요하다."))


def _mlflow_health_problem(uri):
    """MLflow 서버 /health를 MLFLOW_HEALTH_TIMEOUT_SEC 안에 확인한다(2026-10-11). 정상(HTTP 200)이면 None,
    아니면 원인 요약(예외 클래스명 또는 HTTP 상태)을 돌려준다. URL·예외 원문은 돌려주지 않는다."""
    try:
        r = requests.get(uri.rstrip("/") + "/health", timeout=MLFLOW_HEALTH_TIMEOUT_SEC)
    except requests.RequestException as e:
        return type(e).__name__
    return None if r.status_code == 200 else f"HTTP {r.status_code}"


def _model_kwargs(num_stocks):
    # 체크포인트 meta의 model_kwargs — PooledLSTMRegressor 생성자가 받는 인자만 담는다(2026-09-27).
    return dict(
        lookback=LOOKBACK, num_stocks=num_stocks, embedding_dim=EMBEDDING_DIM,
        hidden_size=D_MODEL, num_layers=NUM_LAYERS, dropout=DROPOUT,
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
    # cov_type="classic" — 표준오차(샌드위치 공분산)는 안 쓰고 params 점추정치만 쓰므로
    # 기본값 "robust"의 추가 계산 비용을 뺀다(2026-09-20).
    res_train = am_train.fit(disp="off", cov_type="classic")
    params = res_train.params
    _save_garch_cache(ticker, params, train_end)
    return params, True


def compute_full_period_sigma_cached(returns, params):
    """이미 가진(캐시된) params로 재적합 없이 전체 구간 GARCH σ를 재귀 계산한다
    (forecast_with_fixed_params(start_pos=0)). 이름의 "cached"는 params가 캐시에서 온다는 뜻이다."""
    return forecast_with_fixed_params(returns, params, 0)


# ── baseline 3종: val 구간 RMSE(게이트용) + 다음 거래일 점예측(감사 컬럼용) ──────────────

def compute_ticker_baselines(ticker, start_date, end_date, train_end, val_end, prices=None):
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
    params, refit = get_or_fit_garch_params(ticker, returns, train_end)
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


def train_daily_pooled_model(tickers, start_date, end_date, seed=SEED, shared_cache=None, prices_by_ticker=None):
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
    result = train_pooled_lstm(
        X_train_s, y_train, tid_train, X_val_s, y_val, tid_val,
        len(feature_cols), LOOKBACK, len(tickers), EMBEDDING_DIM,
        D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT,
        BATCH_SIZE, LR, WEIGHT_DECAY, SMOKE_EPOCHS, MAX_EPOCHS, PATIENCE,
        device, seed=seed, label="pooled-volatility-hybrid",
    )

    val_preds, val_actuals, val_tids = evaluate_pooled_predictions(result["model"], result["val_loader"], device)
    # 2026-09-27 게이트 집계 버그 수정: 아래 baseline 3종은 "종목별 val RMSE의 단순 평균"인데,
    # 하이브리드만 전체 val 행을 한데 모은(pooled) RMSE로 계산해 비교하고 있었다 — 같은 오차라도
    # 종목별 RMSE 평균 <= pooled RMSE(젠센 부등식)라 baseline에 구조적으로 유리한 비교였다(2026-09-12
    # 첫 운영 실행부터 게이트가 계속 미통과한 원인). 100종목 검증 정본(pre-cleanup 태그 시점의 검증용
    # test_evaluation_pooled100_hybrid.py:253)과 같은 방식으로 하이브리드도 종목별 RMSE 평균으로 맞춘다. 경위는 CLAUDE.md 참고.
    hybrid_val_rmse = float(np.mean([
        rmse_mae(val_preds[val_tids == tid], val_actuals[val_tids == tid])[0]
        for tid in np.unique(val_tids)
    ]))

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
        "history": result["history"], "best_epoch": result["best_epoch"],
        "best_val_loss": result["best_val_loss"],
        "overfit_ratio_at_end": result["overfit_ratio_at_end"],
    }


# ── 다음 거래일 예측 + 저장 ───────────────────────────────────────────────────────

def next_weekday(d):
    """옛 방향예측 파이프라인(가격예측_공통.py·가격예측_통합모델.py)에서 쓰던 근사를 값 변경 없이 그대로
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
                     is_early_warning, model_version, gate, input_data_suspect=False):
    """model_predictions에 UPSERT((ticker, target_date) 기준) — 재실행 시 최신 값으로 덮어씀.
    input_data_suspect(2026-09-27): 입력 데이터 오염 의심 플래그 — 재실행으로 예측값이 바뀌면
    플래그도 그 실행 기준으로 함께 덮어쓴다(깨끗한 재실행 후에도 true가 남지 않도록)."""
    query = """
        INSERT INTO model_predictions
            (ticker, target_date, prediction_date, predicted_volatility,
             garch_baseline, sma20_baseline, parkinson_sma20_baseline,
             is_early_warning, model_version,
             gate_passed, gate_vs_garch, gate_vs_sma20, gate_vs_parkinson, input_data_suspect)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                gate_vs_parkinson = EXCLUDED.gate_vs_parkinson,
                input_data_suspect = EXCLUDED.input_data_suspect
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
                bool(input_data_suspect),
            ))
        conn.commit()


def predict_and_save_for_ticker(ticker, start_date, end_date, model, scaler, feature_cols,
                                 ticker_to_id, device, version, gate, baselines_by_ticker,
                                 input_data_suspect=False, shared_cache=None,
                                 prices=None):
    """pooled 하이브리드 모델로 한 종목의 다음 거래일 변동성을 예측하고 model_predictions에
    저장한다. run_isolated로 감싸 호출하는 것을 전제로 예외를 그대로 던진다(격리 책임은
    호출부에 있음).

    shared_cache(선택, 2026-10-05): load_shared_cache() 결과. None이면 종목마다 거시지표·KOSPI를
    다시 조회한다(예전 동작 — 일일 실행 전송량의 대부분을 차지했던 경로).
    prices(선택, 2026-10-05): load_price_cache(...)[ticker]. None이면 가격을 직접 조회한다."""
    params = baselines_by_ticker[ticker]["params"]
    window_df, last_confirmed_date = build_next_day_merged_window_volatility_hybrid(
        ticker, start_date, end_date, LOOKBACK, params, **(shared_cache or {}),
        precomputed_prices=prices,
    )
    X_next = window_df[feature_cols].values[np.newaxis, :, :].astype(np.float32)
    X_next_s = scaler.transform(X_next)
    tid = np.array([ticker_to_id[ticker]], dtype=np.int64)

    model.eval()
    with torch.no_grad():
        pred = model(
            torch.from_numpy(X_next_s).to(device), torch.from_numpy(tid).to(device)
        ).cpu().numpy()[0]

    predicted_volatility = float(pred)
    # sanity check(2026-09-14, 사람 결정): NaN/inf/음수만 차단 — 변동성은 음수가 될 수 없는
    # 값이라 확실한 결함 신호지만, 상한선은 일부러 두지 않는다. 실제 급변장에서는 크게 튀는
    # 예측이 정상일 수 있어 임의 상한이 오히려 정상 예측을 막을 위험이 있다는 판단.
    if not np.isfinite(predicted_volatility) or predicted_volatility < 0:
        raise ValueError(
            f"[{ticker}] 예측값 비정상(NaN/inf/음수) — predicted_volatility={predicted_volatility!r}. "
            f"저장을 건너뜁니다."
        )

    prediction_date = pd.Timestamp(last_confirmed_date).date()
    target_date = next_weekday(prediction_date)

    trailing_mean = _trailing_predicted_volatility_mean(ticker, target_date)
    is_early_warning = bool(trailing_mean is not None and predicted_volatility > 1.5 * trailing_mean)

    next_day = baselines_by_ticker[ticker]["next_day"]
    save_prediction(
        ticker, target_date, prediction_date, predicted_volatility,
        next_day["garch"], next_day["sma20"], next_day["parkinson"],
        is_early_warning, version, gate, input_data_suspect=input_data_suspect,
    )

    return {
        "predicted_volatility": predicted_volatility,
        "garch_baseline": next_day["garch"], "sma20_baseline": next_day["sma20"],
        "parkinson_sma20_baseline": next_day["parkinson"],
        "is_early_warning": is_early_warning,
        "prediction_date": str(prediction_date), "target_date": str(target_date),
    }


def _log_mlflow_run(tickers, version, train_out, gate, prediction_results, input_data_suspect=False):
    """일일 실행 결과(학습 곡선, 게이트 판정, 종목별 예측 성공/실패 수)를 MLflow
    "일일_자동화" experiment에 기록한다. 게이트 통과/미통과 여부와 무관하게 항상 호출된다
    — MLflow는 "무슨 일이 있었는지 전부 보는" 관측 도구이기 때문(2026-09-14, MLflow 통합
    설계 결정).

    2026-09-14(D-1/D-2, A-1의 "저장 스킵" 결정 철회): 예전에는 게이트 미통과 시
    run_pooled_volatility_pipeline()이 예측 루프 자체를 건너뛰어 이 함수도 "스킵됨" 분기
    (skip_reason 태그만 남기고 종목별 성공/실패 카운트는 생략)를 탔었다. 이제 예측 루프가
    게이트 결과와 무관하게 항상 실행되므로 그 분기 자체가 코드상 도달 불가능해져
    prediction_results를 필수 인자로 바꾸고 skip 관련 분기를 제거했다 — model_predictions에
    저장되는지 여부와 무관하게(지금은 항상 저장됨) MLflow에는 항상 같은 형태로 기록된다.

    ⚠️ 이 함수 전체를 하나의 try/except로 감싼다 — MLflow 서버가 꺼져있거나 응답이 없어도
    파이프라인 본체(학습/예측/DB저장)는 절대 중단되면 안 된다는 요구사항 때문. A-2의
    예측값 sanity check(잘못된 예측값 자체를 막는 것)와는 성격이 다른 방어 코드다: 이쪽은
    "로깅이라는 부가 기능의 실패를 흡수"하는 것이 목적이라 실패해도 예외를 다시 던지지
    않고 경고만 출력한다.

    2026-10-11 기록 정정(과거 run은 고치지 않는다, 읽는 법은 CLAUDE.md "MLflow 읽는 법"):
    - run 이름: 실행일(date.today()) 대신 "pd<prediction_date>_v<체크포인트 버전>". 실행일은 run_date 태그.
    - params: 모델 생성자 인자(_model_kwargs — 체크포인트 meta와 같은 출처)와 MODEL_ARCHITECTURE(= MODEL_FAMILY).
      예전에는 LSTM 운영 중에도 Transformer 전용 값(NHEAD, DIM_FEEDFORWARD)이 찍혔다.
    - tag model_family = MODEL_FAMILY('lstm'), model_registry.model_family와 같은 어휘.
    - 하이브리드 게이트 RMSE는 hybrid_rmse_tickermean으로만 기록한다. 옛 이름 hybrid_rmse는 2026-09-28
      실행부터 pooled에서 종목평균으로 정의가 바뀌어 한 줄로 읽으면 안 된다.
    - input_data_suspect는 실행 단위 플래그라 n_input_suspect는 0 또는 저장 성공 종목 수다.

    2026-10-11 지연 상한: 시작할 때 /health를 3초 안에 확인하고, 응답이 없으면 줄 맨 앞에 [MLFLOW-SKIP]을 붙인
    경고를 출력한 뒤 기록 없이 돌아간다(except를 거치지 않으므로 여기서 직접 출력한다). 확인을 통과한 뒤 실패해도
    MLFLOW_HTTP_LIMITS로 재시도를 줄여 상한을 둔다. 둘 다 예측 저장 이후라 결과·종료 코드에는 영향이 없다."""
    try:
        for k, v in MLFLOW_HTTP_LIMITS.items():
            os.environ.setdefault(k, v)
        if MLFLOW_TRACKING_URI.startswith("http"):
            problem = _mlflow_health_problem(MLFLOW_TRACKING_URI)
            if problem:
                print(f"[MLFLOW-SKIP] ⚠️ MLflow 서버 응답 없음 — 기록 건너뜀({problem}). "
                      f"예측 저장은 끝났고, 이번 실행의 MLflow run은 남지 않는다.")
                return
        pred_dates = sorted({v["prediction_date"] for v in prediction_results.values() if isinstance(v, dict)})
        pred_date_tag = ",".join(pred_dates) if pred_dates else "none"
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"pd{pred_date_tag}_v{version}"):
            mlflow.set_tags({
                "n_tickers": len(tickers), "version": version,
                "model_family": MODEL_FAMILY,
                "prediction_date": pred_date_tag,
                "run_date": date.today().isoformat(),
                "input_data_suspect": str(bool(input_data_suspect)).lower(),
            })
            mlflow.log_params({
                "MODEL_ARCHITECTURE": MODEL_FAMILY,
                "model_class": MODEL_CLASS_NAME,
                **{f"model_{k}": v for k, v in _model_kwargs(len(tickers)).items()},
                "BATCH_SIZE": BATCH_SIZE, "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY,
                "SMOKE_EPOCHS": SMOKE_EPOCHS, "MAX_EPOCHS": MAX_EPOCHS, "PATIENCE": PATIENCE, "SEED": SEED,
            })
            for epoch, train_loss, val_loss in train_out["history"]:
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metrics({
                "best_val_loss": train_out["best_val_loss"],
                "overfit_ratio_at_end": train_out["overfit_ratio_at_end"],
                "hybrid_rmse_tickermean": gate["hybrid_rmse"],
                "garch_rmse": gate["garch_rmse"],
                "sma_rmse": gate["sma_rmse"],
                "parkinson_rmse": gate["parkinson_rmse"],
                "gate_passed": int(gate["passed"]),
            })
            n_success, n_failed, n_excluded = _count_prediction_results(prediction_results)
            mlflow.log_metrics({
                "n_predictions_success": n_success, "n_predictions_failed": n_failed,
                "n_predictions_excluded": n_excluded,
                "n_input_suspect": n_success if input_data_suspect else 0,
            })
    except Exception as e:
        print(f"⚠️ MLflow 로깅 실패(파이프라인은 계속 진행됩니다) — {e}")


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


def _count_prediction_results(results):
    """종목별 결과를 (성공, 실패, 제외) 수로 센다. 성공 = dict, 실패 = "실패 ..." 문자열,
    제외 = "제외 ..." 문자열(가격 정렬 이탈로 예측을 저장하지 않은 종목, 2026-10-17)."""
    n_success = sum(1 for v in results.values() if isinstance(v, dict))
    n_failed = sum(1 for v in results.values() if isinstance(v, str) and v.startswith("실패"))
    n_excluded = sum(1 for v in results.values() if isinstance(v, str) and v.startswith("제외"))
    return n_success, n_failed, n_excluded


def run_pooled_volatility_pipeline(tickers, start_date, end_date):
    """전체 파이프라인: baseline 3종 계산(GARCH 캐시 포함) -> pooled 하이브리드 학습(전 종목
    통합) -> 체크포인트 저장 -> 배포 게이트 판정 -> 종목별 다음 거래일 예측(run_isolated로
    종목별 격리) -> model_predictions 저장(게이트 결과와 무관하게 항상 저장).

    학습 자체는 격리하지 않는다(가격예측_통합모델.py와 동일 근거 — 모델이 하나라 학습 실패는
    전체 실패, 단일 종목 파이프라인처럼 "한 종목만 실패, 나머지는 계속"이 구조적으로 불가능).
    예측 단계만 종목별로 run_isolated로 감싼다.

    2026-09-14(D-1, A-1의 "저장 스킵" 결정 철회 — 같은 날 안에서 방침이 두 번 바뀜, 아래
    경과 참고): A-1(같은 날 오전)은 "gate['passed']가 False면 예측 루프 자체를 건너뛴다"는
    엄격한 방식을 채택했다 — 그전에는 gate_passed=False가 model_predictions에 기록만 될
    뿐 저장을 막지 않아 게이트가 사실상 장식이었기 때문. 이후 actual_volatility 백필로 A-1
    적용 *이전*에 저장됐던 100행을 실제 결과와 대조해보니 하이브리드가 세 baseline 전부에
    뒤졌음이 확인돼(결과_TaskT_변동성예측_최종.md 및 아래 "actual_volatility 백필" 절
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
    # 종목별 주가 최신일 정렬 점검(2026-10-17): 소수 이탈은 그 종목 저장만 빼고 진행, 다수 이탈은 학습 전에 중단.
    price_deviants = check_price_dates_aligned(tickers, end_date)["deviants"]
    # 지수 결측이 있으면 이번 실행의 예측 전부를 input_data_suspect=True로 기록한다(2026-09-27).
    input_data_suspect = any(n > 0 for n in check_market_index_freshness(end_date).values())
    # 거시지표·KOSPI·종목 가격은 실행 시작 시 한 번만 읽어 학습과 예측이 같은 스냅숏을 쓴다(2026-10-05).
    shared_cache = load_shared_cache(start_date, end_date)
    prices_by_ticker = load_price_cache(tickers, start_date, end_date)
    train_out = train_daily_pooled_model(tickers, start_date, end_date, shared_cache=shared_cache,
                                         prices_by_ticker=prices_by_ticker)
    model, scaler, feature_cols = train_out["model"], train_out["scaler"], train_out["feature_cols"]
    ticker_to_id, device, gate = train_out["ticker_to_id"], train_out["device"], train_out["gate"]
    baselines_by_ticker = train_out["baselines_by_ticker"]

    version = save_checkpoint(
        model, scaler, feature_cols, _model_kwargs(len(tickers)), CHECKPOINT_DIR,
        model_class_name=MODEL_CLASS_NAME,
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
    _register_model_version(version)

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
        r = run_isolated(
            predict_and_save_for_ticker, ticker, start_date, end_date,
            model, scaler, feature_cols, ticker_to_id, device, version, gate, baselines_by_ticker,
            input_data_suspect=input_data_suspect, shared_cache=shared_cache,
            prices=prices_by_ticker[ticker], label=ticker, log_dir=LOG_DIR,
        )
        results[ticker] = r["result"] if r["status"] == "success" else f"실패 ({r['error']})"

    n_success, n_failed, n_excluded = _count_prediction_results(results)
    print(f"\n=== 종목별 예측 결과 ({n_success}/{len(tickers)} 성공, 실패 {n_failed}, 제외 {n_excluded}) ===")
    for ticker, r in results.items():
        print(f"  {ticker}: {r}")
    if n_failed:
        print(f"⚠️ {n_failed}개 종목 예측 실패 — 가격예측/logs/ 에서 상세 로그를 확인하세요.")

    _log_mlflow_run(tickers, version, train_out, gate, results, input_data_suspect=input_data_suspect)
    return {"version": version, "gate": gate, "predictions": results}
