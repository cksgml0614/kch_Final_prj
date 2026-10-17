# split_dataset.py — 기본 피처 + excess_return_z + GARCH σ를 합친 학습용 frame과 서빙용 다음 거래일 창.
# (파일 이름은 1세대에서 정상/스트레스 분할을 하던 시절 이름이다 — 분할은 pooled_dataset.py가 한다.)
# 2026-10-17 정리: 1세대 build_merged_dataset·build_next_day_merged_window·split_full_period·
# split_normal_regime·__main__ 데모를 지웠다(git 태그 pre-cleanup 시점에 있음).

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
    raw = get_features(ticker, start_date, end_date, precomputed_indicators=precomputed_indicators,
                       precomputed_prices=precomputed_prices)
    if raw.empty:
        raise RuntimeError(f"{ticker}: {start_date}~{end_date} 구간에 가격 데이터 없음")
    return raw


def _lagged_features(ticker, start_date, end_date, raw, true_kospi, index):
    """학습·서빙 공용 피처 조립(2026-10-17). index의 각 행 t에 "raw 기준 직전 행"의 값을 놓는다(shift(1)):
    기본 피처 13개(build_stationary_features + recent_vol_ma20)와 excess_return_z_lag1(= z_(t-1)).
    index = raw.index면 학습 행(행 t = target 날짜 t), index = raw.index + 다음 평일이면 마지막 행이 "오늘(T)의
    원값 = T+1행 피처"가 된다. 반환: (features DataFrame, momentum diag)"""
    feats = add_recent_vol(build_stationary_features(raw)).reindex(index).shift(1)
    _, diag = build_momentum_feature(ticker, start_date, end_date, raw=raw, true_kospi=true_kospi)
    feats["excess_return_z_lag1"] = diag["z"].reindex(index).shift(1)
    return feats, diag


def build_merged_dataset_v2_volatility(ticker, start_date, end_date, precomputed_indicators=None, true_kospi=None,
                                       precomputed_prices=None):
    """변동성 예측용 학습 frame(GARCH σ 전, 2026-09-06): 기본 피처 13개 + target(|로그수익률x100|) +
    excess_return_z_lag1. 행 t의 피처는 t 직전 거래일까지의 정보다. GARCH σ는 _hybrid가 붙인다.
    (2026-10-17: 피처 조립을 서빙과 공용인 _lagged_features로 바꿨다 — 열 순서·값은 그대로.)

    precomputed_indicators/true_kospi/precomputed_prices(선택): 공유 캐시(가격예측_변동성_공통.load_shared_cache,
    feature_loader.load_price_cache). None이면 직접 조회한다."""
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
    """분할 경계 기준 달력·월간 게이팅용 유효 날짜(2026-10-17) — GARCH σ를 붙이기 전 변동성 frame의 인덱스.
    예전에는 방향 예측 frame(build_merged_dataset_v2)의 인덱스를 썼다. 두 인덱스는 momentum의 60일 창이 선행
    결측을 가장 길게 만들어 같았고(recent_vol_ma20 20일, GARCH σ 1행), 정리 때 골든 비교로 분할 경계·게이팅
    시퀀스 수가 같음을 확인했다. GARCH 적합이 필요 없어 train_end를 정하기 전에 부를 수 있다."""
    merged, _ = build_merged_dataset_v2_volatility(
        ticker, start_date, end_date, precomputed_indicators=precomputed_indicators, true_kospi=true_kospi,
        precomputed_prices=precomputed_prices,
    )
    return merged.index


def build_merged_dataset_v2_volatility_hybrid(ticker, start_date, end_date, precomputed_sigma, garch_params,
                                               precomputed_indicators=None, true_kospi=None,
                                               precomputed_prices=None):
    """build_merged_dataset_v2_volatility(14피처) + GARCH(1,1) 조건부 σ를 15번째 피처로
    추가한 하이브리드 버전(2026-09-06) — GARCH가 이미 잡아낸 "어제 변동성→오늘 변동성"
    persistence 신호를 신경망이 처음부터 재학습하지 않고 그대로 입력받게 하기 위함.

    precomputed_sigma/garch_params(필수): 호출부(가격예측_변동성_공통.compute_ticker_baselines)가
    train 구간만으로 추정(또는 캐시에서 로드)한 파라미터와 그 파라미터로 계산한 전체 구간 σ 시리즈.
    2026-10-17: 둘 다 없을 때 검증용 함수로 GARCH를 다시 적합하던 폴백 분기와 그 분기에서만 쓰던
    train_end 인자를 지웠다(운영은 항상 넘기고 있었다).

    precomputed_indicators/true_kospi(선택, 2026-09-20): build_merged_dataset_v2_volatility에
    그대로 전달(성능 최적화, 결과는 동일). None이면 기존과 동일."""
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
    """서빙(추론) 전용 — 변동성 하이브리드 모델(15피처)의 다음 거래일 예측 시퀀스를 만든다(2026-09-06).
    2026-10-17: 학습 frame과 같은 _lagged_features·garch_sigma_series로 조립한다. index를 raw 날짜 + "다음
    평일" 한 행으로 늘려 shift(1)하므로 마지막 행이 오늘(T)의 원값(= T+1행 피처), garch_sigma 마지막 칸은
    forecast_next_day_sigma(T까지 수익률로 낸 T+1 예측)다. 마지막 행 라벨은 다음 평일(save_prediction의
    target_date와 같은 규칙, 공휴일 미반영)이다 — 예전에는 원자료 날짜 T가 붙어 바로 앞 행 라벨과 중복됐다.

    garch_params: 월 1회 캐싱된(또는 방금 재추정한) 고정 GARCH 파라미터 — 호출부가 넘겨준다.
    이 함수 내부에서 재추정하지 않는다.

    반환: (window, last_confirmed_date) — window 컬럼 15개(호출부는 컬럼명으로 인덱싱할 것).

    precomputed_indicators/true_kospi(선택, 2026-10-05, Neon 전송량 절감): build_merged_dataset_v2와
    같은 공유 캐시. 예전에는 이 함수가 종목마다 get_features()를 두 번(base window용, momentum용)
    캐시 없이 불러 거시지표 9종 전체 이력을 종목당 2회 읽었고, 그게 일일 실행 전송량의 대부분이었다.
    이제 get_features()를 한 번만 부르고 그 raw를 base window·momentum·GARCH 수익률이 함께 쓴다.
    raw["close"]는 예전 load_close_prices()와 같은 쿼리 조건(daily_stock_prices, ticker,
    date BETWEEN start_date AND end_date, ORDER BY date)·같은 float 변환이라 값이 같다.
    둘 다 None이면 거시지표·KOSPI를 예전처럼 직접 조회한다."""
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
    sigma = pd.concat([sigma, pd.Series([forecast_next_day_sigma(returns, garch_params)], index=[next_label])])
    feats["garch_sigma"] = sigma.reindex(index)

    window = feats.tail(lookback)
    if len(window) != lookback or window["garch_sigma"].isna().any():
        raise RuntimeError(f"{ticker}: 서빙 창 길이·GARCH σ 정렬 불일치(길이 {len(window)}, 기대 {lookback})")
    return window, last_confirmed_date
