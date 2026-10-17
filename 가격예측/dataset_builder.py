# dataset_builder.py — 행 단위 기본 피처(OHLCV 비율 + 시장지표, t-1 정렬)와 target 계산.
# excess_return_z 피처는 momentum_feature.py, GARCH σ는 split_dataset.py에서 합친다.
# 2026-10-17 정리: 1세대(레벨 피처·단일 종목) build_base_dataset·build_next_day_feature_window·
# verify_index_alignment·__main__ 데모를 지웠다(git 태그 pre-cleanup 시점에 있음).

import sys

import numpy as np
import pandas as pd

from 시장지표.feature_loader import get_features

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def build_stationary_features(raw):
    """레벨 피처를 정상성 있는 형태로 재설계한다 (사용자 확정, 2026-08-17, 진단 결과 반영).

    - OHLC(레벨, 서로 r>0.99로 사실상 중복) -> 비율/수익률 3종으로 대체:
        close_return    = pct_change(close)                          # 종가 수익률
        hl_range_ratio  = (high - low) / close                       # 당일 변동폭 비율
        open_gap_ratio  = (open - close_(d-1)) / close_(d-1)         # 전일 종가 대비 시가 갭
      raw의 open/high/low/close(레벨) 자체는 버린다 — train(45,800~91,000원)과 test
      (53,000~162,400원)의 가격 레벨 자체가 달라 표준화 시 test가 학습 범위 밖으로 외삽되는
      문제(체크포인트4 진단)를 구조적으로 없앤다.
    - volume -> log1p(volume)
    - 국고채 3년/10년(r=0.989, 사실상 중복) -> 스프레드(10년-3년) 1개로 축소
    - 나머지 지표(KOSPI/KOSDAQ/USD_KRW/기준금리/CPI/M2/선행지수)는 레벨 그대로 유지
      (이번 재설계 범위 밖 — 필요 시 다음 반복에서 재검토)

    반환 DataFrame은 raw와 동일한 인덱스(거래일, 아직 t-1 shift 전)를 가진다.
    """
    df = pd.DataFrame(index=raw.index)

    df["close_return"] = raw["close"].pct_change()
    df["hl_range_ratio"] = (raw["high"] - raw["low"]) / raw["close"]
    df["open_gap_ratio"] = (raw["open"] - raw["close"].shift(1)) / raw["close"].shift(1)
    df["log_volume"] = np.log1p(raw["volume"])

    df["KOSPI"] = raw["KOSPI"]
    df["KOSDAQ"] = raw["KOSDAQ"]
    df["USD_KRW"] = raw["USD_KRW"]
    df["ECOS_722Y001_0101000"] = raw["ECOS_722Y001_0101000"]  # 기준금리
    df["bond_spread_10y_3y"] = raw["ECOS_817Y002_010210000"] - raw["ECOS_817Y002_010200000"]
    df["ECOS_901Y009_0"] = raw["ECOS_901Y009_0"]  # CPI
    df["ECOS_161Y005_BBHS00"] = raw["ECOS_161Y005_BBHS00"]  # M2
    df["ECOS_901Y067_I16E"] = raw["ECOS_901Y067_I16E"]  # 선행지수 순환변동치

    return df


def build_base_dataset_v2(ticker, start_date, end_date, precomputed_indicators=None, precomputed_prices=None):
    """build_base_dataset과 동일한 t-1 정렬 방식이되, build_stationary_features로 재설계한
    피처 세트를 쓴다. target 정의는 동일 — close 레벨은 target 계산에만 쓰고 피처에서는 뺀다.

    precomputed_indicators(선택, 2026-09-20): get_features()에 그대로 전달 — 시장지표.
    feature_loader.load_indicator_cache(end_date) 참고. None이면 기존과 동일.
    precomputed_prices(선택, 2026-10-05): get_features()에 그대로 전달 — 시장지표.feature_loader.
    load_price_cache() 참고. None이면 기존과 동일."""
    raw = get_features(ticker, start_date, end_date, precomputed_indicators=precomputed_indicators,
                       precomputed_prices=precomputed_prices)
    if raw.empty:
        raise RuntimeError(f"{ticker}: {start_date}~{end_date} 구간에 가격 데이터 없음")

    target = raw["close"].pct_change()
    target.name = "target"

    stationary = build_stationary_features(raw)
    features = stationary.shift(1)

    combined = features.copy()
    combined["target"] = target
    combined = combined.dropna(how="any")

    return combined, raw


def build_base_dataset_v2_volatility(ticker, start_date, end_date, vol_window=20, precomputed_indicators=None,
                                     precomputed_prices=None):
    """V2 변동성 예측용 데이터셋(2026-09-06, Task T 변동성 예측 — GARCH baseline과 비교).
    build_base_dataset_v2와 피처 계산 기반은 동일(build_stationary_features 재사용 — 방향
    모델과 완전히 공유)하되 두 가지가 다르다:

    1) target 정의 — 방향 모델의 pct_change()(부호 있는 단순수익률) 대신 garch_baseline.py와
       완전히 동일한 척도를 쓴다: |log(close_t/close_(t-1)) x 100|(로그수익률x100의 절댓값).
       척도가 다르면 GARCH/SMA와의 비교 자체가 무의미해지므로(사람 지시) 반드시 일치시켰다.
    2) 피처에 recent_vol_ma20(직전 vol_window일 |close_return| 이동평균)을 추가한다 — GARCH가
       "직전 변동성"을 핵심 입력(persistence)으로 쓰는 것과 대응시키기 위함. Transformer는
       원칙적으로 lookback 시퀀스 안의 raw close_return들로부터 이 통계를 스스로 학습할 수도
       있지만, 그러려면 attention/선형층이 절댓값+평균이라는 비선형 집계를 알아서 근사해야
       한다 — 얕은(1~2층, d_model 32) 인코더에 이걸 맡기기보다 GARCH와 대등한 조건을 만들기
       위해 명시적 피처로 제공하는 쪽을 택했다.

    build_stationary_features() 자체는 건드리지 않는다 — 방향 모델(build_base_dataset_v2)의
    피처셋은 이 함수와 무관하게 그대로 유지된다.

    반환: combined(피처+target), raw — build_base_dataset_v2와 동일한 형태(단 target 의미가 다름).

    precomputed_indicators(선택, 2026-09-20): get_features()에 그대로 전달. None이면 기존과 동일.
    precomputed_prices(선택, 2026-10-05): get_features()에 그대로 전달 — 시장지표.feature_loader.
    load_price_cache() 참고. None이면 기존과 동일.
    """
    raw = get_features(ticker, start_date, end_date, precomputed_indicators=precomputed_indicators,
                       precomputed_prices=precomputed_prices)
    if raw.empty:
        raise RuntimeError(f"{ticker}: {start_date}~{end_date} 구간에 가격 데이터 없음")

    log_return_pct = np.log(raw["close"] / raw["close"].shift(1)) * 100
    target = log_return_pct.abs()
    target.name = "target"

    stationary = build_stationary_features(raw).copy()
    stationary["recent_vol_ma20"] = stationary["close_return"].abs().rolling(vol_window, min_periods=vol_window).mean()

    features = stationary.shift(1)

    combined = features.copy()
    combined["target"] = target
    combined = combined.dropna(how="any")

    return combined, raw


def build_next_day_feature_window_volatility(ticker, start_date, end_date, lookback, vol_window=20, raw=None):
    """build_next_day_feature_window()의 변동성 버전(2026-09-06, 가격예측_변동성_일일수집.py용)
    — base V2 12개 + recent_vol_ma20(13개)로 다음 거래일 예측 시퀀스를 만든다. momentum/
    garch_sigma는 이 함수 밖(split_dataset.build_next_day_merged_window_volatility_hybrid)
    에서 합친다. 원리는 build_next_day_feature_window()와 완전히 동일 — 오늘(T)의 원값을
    'T+1행'의 피처로 취급한다.

    raw(선택, 2026-10-05): 호출부가 이미 계산한 get_features(ticker, start_date, end_date)
    결과를 넘기면 다시 조회하지 않는다(momentum_feature.build_momentum_feature의 raw 인자와 같은
    방식, 같은 인자로 계산된 것이어야 한다 — 호출부 책임). None이면 기존과 동일."""
    if raw is None:
        raw = get_features(ticker, start_date, end_date)
    if raw.empty:
        raise RuntimeError(f"{ticker}: {start_date}~{end_date} 구간에 가격 데이터 없음")
    if len(raw) < lookback:
        raise RuntimeError(f"{ticker}: 이력이 lookback({lookback})보다 짧음({len(raw)}행) — 예측 불가")

    stationary = build_stationary_features(raw).copy()
    stationary["recent_vol_ma20"] = stationary["close_return"].abs().rolling(vol_window, min_periods=vol_window).mean()
    shifted = stationary.shift(1)

    past_part = shifted.tail(lookback - 1)
    today_part = stationary.tail(1)

    window = pd.concat([past_part, today_part])
    if len(window) != lookback:
        raise RuntimeError(
            f"{ticker}: 시퀀스 길이 불일치 — 기대 {lookback}, 실제 {len(window)} (이력 부족 가능성)"
        )

    last_confirmed_date = raw.index[-1]
    return window, last_confirmed_date
