# dataset_builder.py — 행 단위 기본 피처(OHLCV 비율 + 시장지표 + recent_vol_ma20)와 변동성 target.
# t-1 정렬(lag)과 excess_return_z·GARCH σ 결합은 split_dataset.py가 학습·서빙 공용으로 한다.
# 2026-10-17 정리: 1세대(레벨 피처·단일 종목) build_base_dataset·build_next_day_feature_window·
# verify_index_alignment·__main__ 데모, 방향 target build_base_dataset_v2, 학습·서빙에 따로 있던
# build_base_dataset_v2_volatility·build_next_day_feature_window_volatility(recent_vol_ma20 복제)를 지웠다
# (git 태그 pre-cleanup 시점에 있음).

import sys

import numpy as np
import pandas as pd

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


def add_recent_vol(stationary, vol_window=20):
    """recent_vol_ma20 = 직전 vol_window일 |close_return| 이동평균(같은 날 포함, lag 전). GARCH가 "직전 변동성"을
    핵심 입력(persistence)으로 쓰는 것과 대응시키려고 명시적 피처로 준다(2026-09-06). 학습·서빙 공용(2026-10-17,
    이전에는 두 경로에 같은 줄이 복제돼 있었다)."""
    out = stationary.copy()
    out["recent_vol_ma20"] = out["close_return"].abs().rolling(vol_window, min_periods=vol_window).mean()
    return out


def volatility_target(raw):
    """target_t = |log(close_t/close_(t-1)) x 100| — garch_baseline.compute_log_returns_pct와 같은 척도(GARCH·SMA
    baseline과 비교하려면 반드시 같아야 한다, 2026-09-06)."""
    target = (np.log(raw["close"] / raw["close"].shift(1)) * 100).abs()
    target.name = "target"
    return target
