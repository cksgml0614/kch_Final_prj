# split_dataset.py — 기본 피처 + excess_return_z + GARCH σ를 합친 학습용 frame과 서빙용 다음 거래일 창.
# (파일 이름은 1세대에서 정상/스트레스 분할을 하던 시절 이름이다 — 분할은 pooled_dataset.py가 한다.)
# 2026-10-17 정리: 1세대 build_merged_dataset·build_next_day_merged_window·split_full_period·
# split_normal_regime·__main__ 데모를 지웠다(git 태그 pre-cleanup 시점에 있음).

import sys

import pandas as pd

from 가격예측.dataset_builder import (
    build_base_dataset_v2,
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


def build_merged_dataset_v2(ticker, start_date, end_date, precomputed_indicators=None, true_kospi=None,
                            precomputed_prices=None):
    """build_merged_dataset과 동일하나 build_base_dataset_v2(정상성 재설계 피처)를 쓴다.
    momentum(excess_return_z_lag1)은 이미 정상성 있는 z-score라 변경 없이 그대로 재사용.

    precomputed_indicators/true_kospi(선택, 2026-09-20 추가, 성능 최적화 — 100종목 pooled
    파이프라인에서 종목과 무관한 데이터를 종목마다 재조회하던 것을 없앤다): 각각 시장지표.
    feature_loader.load_indicator_cache(end_date) / momentum_feature.load_true_kospi(...)의
    결과를 그대로 전달. 이전에는 이 함수가 get_features()를 build_base_dataset_v2 안에서
    한 번, build_momentum_feature 안에서 또 한 번 — 종목당 총 두 번 호출했다. 여기서는
    build_base_dataset_v2가 반환하는 raw를 그대로 build_momentum_feature에 넘겨 get_features
    호출을 종목당 1회로 줄인다. 조인/계산 로직 자체는 전혀 바뀌지 않는다 — raw가 언제나
    build_base_dataset_v2(ticker, start_date, end_date, precomputed_indicators)와 정확히
    같은 값이기 때문(캐시가 있든 없든 get_features()의 반환값은 동일함이 캐시 추가 시점에
    검증됨). 둘 다 None이면 기존과 완전히 동일하게 동작한다."""
    base, raw = build_base_dataset_v2(ticker, start_date, end_date, precomputed_indicators=precomputed_indicators,
                                      precomputed_prices=precomputed_prices)
    momentum, diag = build_momentum_feature(ticker, start_date, end_date, raw=raw, true_kospi=true_kospi)

    merged = base.copy()
    merged["excess_return_z_lag1"] = momentum
    before = len(merged)
    merged = merged.dropna(how="any")
    after = len(merged)

    return merged, {"base": base, "raw": raw, "momentum": momentum, "diag": diag,
                     "rows_before_momentum_dropna": before, "rows_after_momentum_dropna": after}


def build_merged_dataset_v2_volatility(ticker, start_date, end_date, precomputed_indicators=None, true_kospi=None,
                                       precomputed_prices=None):
    """build_merged_dataset_v2의 변동성 버전(2026-09-06, Task T 변동성 예측). target이
    |로그수익률x100|이고 피처에 recent_vol_ma20이 추가된 build_base_dataset_v2_volatility를
    쓴다는 점만 다르다 — momentum(excess_return_z_lag1)은 방향/변동성 어느 쪽이든 "종목의
    시장 대비 초과수익률 흐름"이라는 같은 의미의 입력 피처라 그대로 재사용한다(target 정의와
    무관, 재계산 불필요).

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


def build_merged_dataset_v2_volatility_hybrid(ticker, start_date, end_date, train_end,
                                               precomputed_sigma=None, garch_params=None,
                                               precomputed_indicators=None, true_kospi=None,
                                               precomputed_prices=None):
    """build_merged_dataset_v2_volatility(14피처) + GARCH(1,1) 조건부 σ를 15번째 피처로
    추가한 하이브리드 버전(2026-09-06) — GARCH가 이미 잡아낸 "어제 변동성→오늘 변동성"
    persistence 신호를 Transformer가 처음부터 재학습하지 않고 그대로 입력받게 하기 위함.

    train_end가 필요한 이유: GARCH 파라미터는 train 구간만으로 추정해야 하는데, 이 함수는
    종목별로 독립 호출되므로 전역 분할 경계(pooled_dataset.compute_global_split_dates로 미리
    계산된 값)를 호출부가 넘겨줘야 한다 — 이 함수 내부에서 새로 계산하면 대표 종목 기준과
    어긋날 위험이 있다(2026-09-06 GARCH 실험 때 raw 인덱스 기준으로 잘못 계산해 경계가
    한 달 가까이 어긋났던 사고 재발 방지).

    precomputed_sigma/garch_params(둘 다 2026-09-06 추가, 선택): 이미 계산해둔 전체 구간
    σ 시리즈·파라미터가 있으면 그대로 쓰고 내부에서 다시 계산하지 않는다 — 가격예측_변동성_
    일일수집.py가 GARCH 파라미터를 월 1회만 재추정(캐싱)하도록 만들면서, 매일 실행 시
    이 함수가 매번 재추정하면 캐싱이 무의미해지는 문제를 막기 위해 추가했다. 생략하면
    기존과 동일하게(garch_baseline.compute_full_period_sigma()로 매번 계산) 동작한다 —
    기존 호출부(test_evaluation_pooled_volatility_hybrid.py 등)는 값 변경 없음.

    garch_baseline 함수들을 그대로 재사용한다 — 여기서 지연 임포트하는 이유는 garch_baseline.py가
    이미 split_dataset.build_merged_dataset_v2를 임포트하고 있어(분할 경계 계산용), 모듈
    최상단에서 서로를 임포트하면 순환 임포트가 되기 때문이다.

    precomputed_indicators/true_kospi(선택, 2026-09-20): build_merged_dataset_v2_volatility에
    그대로 전달(성능 최적화, 결과는 동일). None이면 기존과 동일."""
    merged, meta = build_merged_dataset_v2_volatility(
        ticker, start_date, end_date,
        precomputed_indicators=precomputed_indicators, true_kospi=true_kospi,
        precomputed_prices=precomputed_prices,
    )

    if precomputed_sigma is not None:
        sigma_full, params = precomputed_sigma, garch_params
    else:
        from 가격예측.검증용.garch_baseline_check import compute_full_period_sigma  # 2026-09-12: 함수가 검증용/으로 이동(경로 갱신, 순환 임포트 방지 목적은 동일)
        sigma_full, params = compute_full_period_sigma(ticker, start_date, end_date, train_end)

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
    (13개: base 12 + recent_vol_ma20) + momentum(1개, build_next_day_merged_window()와 동일
    원리) + garch_sigma(1개, GARCH 다음 거래일 예측)를 합친다.

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
