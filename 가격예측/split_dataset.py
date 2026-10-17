# split_dataset.py — 기본 피처·excess_return_z·GARCH σ를 합친 학습 frame과 서빙용 다음 거래일 창을 만든다.
# 파이프라인 단계: 피처(학습·서빙 공용 조립). 파일 이름은 옛 이름이고, 날짜 분할은 pooled_dataset.py가 한다.
# 주요 진입 함수: build_merged_dataset_v2_volatility_hybrid(학습), build_next_day_merged_window_volatility_hybrid(서빙).
# 배경: 결정기록.md의 "GARCH σ를 15번째 피처로 넣는 하이브리드 구조"

import sys

import pandas as pd

from 가격예측.dataset_builder import add_recent_vol, build_stationary_features, volatility_target
from 가격예측.momentum_feature import build_momentum_feature
from 시장지표.feature_loader import get_features

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO는 나머지 전부 (반올림 오차를 test 쪽에 흡수)


def _load_raw(ticker, start_date, end_date, precomputed_indicators, precomputed_prices):
    """
    설명: 종목 가격과 지표를 합친 원자료(get_features)를 읽는다.
    주의 사항: 캐시가 없으면 DB 읽기. 행이 없으면 RuntimeError.
    입력 타입->반환 타입: (str, str, str, dict|None, pd.DataFrame|None) -> pd.DataFrame
    """
    raw = get_features(ticker, start_date, end_date, precomputed_indicators=precomputed_indicators,
                       precomputed_prices=precomputed_prices)
    if raw.empty:
        raise RuntimeError(f"{ticker}: {start_date}~{end_date} 구간에 가격 데이터 없음")
    return raw


def _lagged_features(ticker, start_date, end_date, raw, true_kospi, index):
    """
    설명: index 각 행에 직전 행의 기본 피처 13개와 excess_return_z_lag1을 놓은 피처 frame과 momentum diag를 돌려준다.
    주의 사항: index=raw.index면 학습 행, raw.index+다음 평일이면 마지막 행이 다음 거래일 피처다(학습·서빙 공용).
    입력 타입->반환 타입: (str, str, str, pd.DataFrame, pd.Series|None, pd.DatetimeIndex) -> (pd.DataFrame, dict)
    """
    feats = add_recent_vol(build_stationary_features(raw)).reindex(index).shift(1)
    _, diag = build_momentum_feature(ticker, start_date, end_date, raw=raw, true_kospi=true_kospi)
    feats["excess_return_z_lag1"] = diag["z"].reindex(index).shift(1)
    return feats, diag


def build_merged_dataset_v2_volatility(ticker, start_date, end_date, precomputed_indicators=None, true_kospi=None,
                                       precomputed_prices=None):
    """
    설명: 기본 피처 13개·target·excess_return_z_lag1로 된 학습 frame(GARCH σ 전)과 meta를 만든다.
    주의 사항: 결측 행은 모두 버린다. 캐시 인자가 없으면 DB 읽기.
    입력 타입->반환 타입: (str, str, str, dict|None, pd.Series|None, pd.DataFrame|None) -> (pd.DataFrame, dict)
    """
    raw = _load_raw(ticker, start_date, end_date, precomputed_indicators, precomputed_prices)
    feats, diag = _lagged_features(ticker, start_date, end_date, raw, true_kospi, raw.index)
    merged = feats.drop(columns=["excess_return_z_lag1"])
    merged["target"] = volatility_target(raw)
    merged["excess_return_z_lag1"] = feats["excess_return_z_lag1"]
    before = len(merged)
    merged = merged.dropna(how="any")
    after = len(merged)
    return merged, {"raw": raw, "diag": diag,
                    "rows_before_dropna": before, "rows_after_dropna": after}


def valid_feature_dates(ticker, start_date, end_date, precomputed_indicators=None, true_kospi=None,
                        precomputed_prices=None):
    """
    설명: GARCH σ를 붙이기 전 변동성 frame의 유효 날짜(분할 경계·월간 게이팅용)를 돌려준다.
    주의 사항: GARCH 적합이 필요 없어 train_end를 정하기 전에 부를 수 있다. 캐시가 없으면 DB 읽기.
    입력 타입->반환 타입: (str, str, str, dict|None, pd.Series|None, pd.DataFrame|None) -> pd.DatetimeIndex
    """
    merged, _ = build_merged_dataset_v2_volatility(
        ticker, start_date, end_date, precomputed_indicators=precomputed_indicators, true_kospi=true_kospi,
        precomputed_prices=precomputed_prices,
    )
    return merged.index


def build_merged_dataset_v2_volatility_hybrid(ticker, start_date, end_date, precomputed_sigma, garch_params,
                                               precomputed_indicators=None, true_kospi=None,
                                               precomputed_prices=None):
    """
    설명: 14피처 학습 frame에 GARCH(1,1) 조건부 σ를 15번째 피처(garch_sigma)로 붙인 frame과 meta를 만든다.
    주의 사항: precomputed_sigma·garch_params 필수(train 구간으로만 적합한 값). 여기서 GARCH를 다시 적합하지 않는다.
    입력 타입->반환 타입: (str, str, str, pd.Series, pd.Series, dict|None, pd.Series|None, pd.DataFrame|None) -> (pd.DataFrame, dict)
    """
    if precomputed_sigma is None or garch_params is None:
        raise ValueError(f"{ticker}: precomputed_sigma와 garch_params는 필수다(GARCH 폴백 적합 경로는 없음)")
    merged, meta = build_merged_dataset_v2_volatility(
        ticker, start_date, end_date,
        precomputed_indicators=precomputed_indicators, true_kospi=true_kospi,
        precomputed_prices=precomputed_prices,
    )
    sigma_full, params = precomputed_sigma, garch_params

    merged = merged.copy()
    merged["garch_sigma"] = sigma_full.reindex(merged.index)
    before = len(merged)
    merged = merged.dropna(how="any")
    after = len(merged)

    meta = dict(meta)
    meta["garch_params"] = params
    meta["rows_before_garch_dropna"] = before
    meta["rows_after_garch_dropna"] = after
    return merged, meta


def build_next_day_merged_window_volatility_hybrid(ticker, start_date, end_date, lookback, garch_params,
                                                   precomputed_indicators=None, true_kospi=None,
                                                   precomputed_prices=None):
    """
    설명: 서빙용으로 다음 거래일을 예측할 15피처 창(마지막 lookback행)과 마지막 확정 거래일을 돌려준다.
    주의 사항: 마지막 행 라벨은 다음 평일(공휴일 미반영). garch_params는 고정값으로만 쓴다. 이력·σ가 부족하면 RuntimeError.
    입력 타입->반환 타입: (str, str, str, int, pd.Series, dict|None, pd.Series|None, pd.DataFrame|None) -> (pd.DataFrame, pd.Timestamp)
    """
    from 가격예측.baselines import (  # 순환 임포트 방지(지연 임포트)
        compute_log_returns_pct,
        forecast_next_day_sigma,
        garch_sigma_series,
    )

    raw = _load_raw(ticker, start_date, end_date, precomputed_indicators, precomputed_prices)
    if len(raw) < lookback:
        raise RuntimeError(f"{ticker}: 이력이 lookback({lookback})보다 짧음({len(raw)}행) — 예측 불가")
    last_confirmed_date = raw.index[-1]
    next_label = last_confirmed_date + pd.offsets.BDay(1)
    index = raw.index.append(pd.DatetimeIndex([next_label]))

    feats, diag = _lagged_features(ticker, start_date, end_date, raw, true_kospi, index)
    if pd.isna(diag["z"].iloc[-1]):
        raise RuntimeError(f"{ticker}: 최신 momentum 값이 NaN — 가격 이력이 window보다 짧을 수 있음")

    returns = compute_log_returns_pct(raw["close"])
    sigma = garch_sigma_series(returns, garch_params)
    # 마지막 칸은 오늘(T)까지의 수익률로 낸 T+1 예측
    sigma = pd.concat([sigma, pd.Series([forecast_next_day_sigma(returns, garch_params)], index=[next_label])])
    feats["garch_sigma"] = sigma.reindex(index)

    window = feats.tail(lookback)
    if len(window) != lookback or window["garch_sigma"].isna().any():
        raise RuntimeError(f"{ticker}: 서빙 창 길이·GARCH σ 정렬 불일치(길이 {len(window)}, 기대 {lookback})")
    return window, last_confirmed_date
