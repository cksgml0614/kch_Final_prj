# split_dataset.py — 기본 피처 + excess_return_z + GARCH σ를 합친 학습용 frame과 서빙용 다음 거래일 창.
# (파일 이름은 1세대에서 정상/스트레스 분할을 하던 시절 이름이다 — 분할은 pooled_dataset.py가 한다.)
# 2026-10-17 정리: 1세대 build_merged_dataset·build_next_day_merged_window·split_full_period·
# split_normal_regime·__main__ 데모를 지웠다(git 태그 pre-cleanup 시점에 있음).

import sys

import pandas as pd

from 가격예측.dataset_builder import (
    build_base_dataset_v2_volatility,
    build_next_day_feature_window_volatility,
)
from 가격예측.momentum_feature import build_momentum_feature
from 시장지표.feature_loader import get_features

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO는 나머지 전부 (반올림 오차를 test 쪽에 흡수)


def build_merged_dataset_v2_volatility(ticker, start_date, end_date, precomputed_indicators=None, true_kospi=None,
                                       precomputed_prices=None):
    """변동성 예측용 frame(2026-09-06): build_base_dataset_v2_volatility(target |로그수익률x100|, 피처에
    recent_vol_ma20 포함) + excess_return_z_lag1. GARCH σ는 build_merged_dataset_v2_volatility_hybrid가 붙인다.
    (2026-10-17: 같은 구조의 방향 예측판 build_merged_dataset_v2는 지웠다.)

    precomputed_indicators/true_kospi(선택, 2026-09-20): build_merged_dataset_v2와 동일한
    성능 최적화 — get_features() 종목당 중복 호출 제거. 둘 다 None이면 기존과 동일."""
    base, raw = build_base_dataset_v2_volatility(ticker, start_date, end_date,
                                                  precomputed_indicators=precomputed_indicators,
                                                  precomputed_prices=precomputed_prices)
    momentum, diag = build_momentum_feature(ticker, start_date, end_date, raw=raw, true_kospi=true_kospi)

    merged = base.copy()
    merged["excess_return_z_lag1"] = momentum
    before = len(merged)
    merged = merged.dropna(how="any")
    after = len(merged)

    return merged, {"base": base, "raw": raw, "momentum": momentum, "diag": diag,
                     "rows_before_momentum_dropna": before, "rows_after_momentum_dropna": after}


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
    """서빙(추론) 전용 — 변동성 하이브리드 모델(15피처)의 다음 거래일 예측 시퀀스를 만든다
    (2026-09-06, 가격예측_변동성_일일수집.py용). build_next_day_feature_window_volatility()
    (13개: base 12 + recent_vol_ma20) + momentum(1개, 마지막 칸 = 오늘 z) + garch_sigma(1개, GARCH 다음
    거래일 예측)를 합친다.

    garch_params: 월 1회 캐싱된(또는 방금 재추정한) 고정 GARCH 파라미터 — 호출부가 넘겨준다.
    이 함수 내부에서 재추정하지 않는다(캐싱 취지 유지 — 매일 예측 때마다 재추정하면 캐싱이
    무의미해짐).

    ⚠️ 정렬 주의: returns(로그수익률)는 raw(get_features 원본)보다 첫 행이 하나 적다(pct_change
    특성상 첫 행이 NaN이라 dropna로 빠짐) — 그래서 위치 기반이 아니라 base_window의 실제
    날짜 라벨(base_window.index)로 GARCH 조회 시작 위치를 찾는다(위치 산술로 하면 하루
    어긋난다).

    반환: (window, last_confirmed_date) — window 컬럼 15개(순서는 feature_cols와 다를 수
    있으니 호출부는 반드시 컬럼명으로 인덱싱할 것).

    precomputed_indicators/true_kospi(선택, 2026-10-05, Neon 전송량 절감): build_merged_dataset_v2와
    같은 공유 캐시. 예전에는 이 함수가 종목마다 get_features()를 두 번(base window용, momentum용)
    캐시 없이 불러 거시지표 9종 전체 이력을 종목당 2회 읽었고, 그게 일일 실행 전송량의 대부분이었다.
    이제 get_features()를 한 번만 부르고 그 raw를 base window·momentum·GARCH 수익률이 함께 쓴다.
    raw["close"]는 예전 load_close_prices()와 같은 쿼리 조건(daily_stock_prices, ticker,
    date BETWEEN start_date AND end_date, ORDER BY date)·같은 float 변환이라 값이 같다.
    둘 다 None이면 거시지표·KOSPI를 예전처럼 직접 조회한다."""
    from 가격예측.garch_baseline import (  # 순환 임포트 방지(지연 임포트)
        compute_log_returns_pct,
        forecast_next_day_sigma,
        forecast_with_fixed_params,
    )

    raw = get_features(ticker, start_date, end_date, precomputed_indicators=precomputed_indicators,
                       precomputed_prices=precomputed_prices)
    base_window, last_confirmed_date = build_next_day_feature_window_volatility(
        ticker, start_date, end_date, lookback, raw=raw,
    )

    momentum, diag = build_momentum_feature(ticker, start_date, end_date, raw=raw, true_kospi=true_kospi)
    momentum_next = diag["z"].iloc[-1]
    if pd.isna(momentum_next):
        raise RuntimeError(f"{ticker}: 최신 momentum 값이 NaN — 가격 이력이 window보다 짧을 수 있음")
    past_momentum = momentum.tail(lookback - 1).values
    momentum_col = list(past_momentum) + [float(momentum_next)]

    close = raw["close"]
    returns = compute_log_returns_pct(close)

    past_dates = base_window.index[:-1]  # window 마지막 행(오늘=T+1용) 제외 (lookback-1)개 날짜
    # ⚠️ 2026-09-06: forecast_with_fixed_params()가 "start_pos+1부터 시작"하도록 수정됨(라벨링
    # 버그 수정, garch_baseline.py 참고) — 출력 첫 날짜가 past_dates[0]이 되려면 그 하루 전
    # 위치를 start_pos로 넘겨야 한다.
    target_pos = returns.index.get_loc(past_dates[0])
    past_sigma_series = forecast_with_fixed_params(returns, garch_params, target_pos - 1)
    if not past_sigma_series.index.equals(pd.DatetimeIndex(past_dates)):
        raise RuntimeError(f"{ticker}: GARCH 과거 시퀀스 날짜가 base_window와 어긋남")
    garch_next = forecast_next_day_sigma(returns, garch_params)
    garch_col = list(past_sigma_series.values) + [garch_next]

    if len(momentum_col) != lookback or len(garch_col) != lookback:
        raise RuntimeError(
            f"{ticker}: 시퀀스 길이 불일치 — momentum={len(momentum_col)}, garch={len(garch_col)}, 기대={lookback}"
        )

    window = base_window.copy()
    window["excess_return_z_lag1"] = momentum_col
    window["garch_sigma"] = garch_col
    return window, last_confirmed_date
