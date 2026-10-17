# dataset_builder.py — 행 단위 기본 피처(OHLCV 비율·시장지표·recent_vol_ma20)와 변동성 target을 만든다.
# 파이프라인 단계: 피처. t-1 정렬과 excess_return_z·GARCH σ 결합은 split_dataset.py가 한다.
# 주요 진입 함수: build_stationary_features, add_recent_vol, volatility_target.

import sys

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def build_stationary_features(raw):
    """
    설명: 가격 레벨 대신 비율·수익률·로그 거래량과 시장지표 레벨, 국고채 스프레드로 기본 피처를 만든다.
    주의 사항: 반환값은 raw와 같은 거래일 인덱스이며 아직 t-1 shift 전이다(그대로 예측 피처로 쓰면 누수).
    입력 타입->반환 타입: (pd.DataFrame) -> pd.DataFrame
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
    """
    설명: 직전 vol_window일 |close_return| 이동평균을 recent_vol_ma20 열로 더한 복사본을 돌려준다.
    주의 사항: 같은 날 값을 포함한다(lag 전). 학습과 서빙이 함께 쓰는 단일 경로다.
    입력 타입->반환 타입: (pd.DataFrame, int) -> pd.DataFrame
    """
    out = stationary.copy()
    out["recent_vol_ma20"] = out["close_return"].abs().rolling(vol_window, min_periods=vol_window).mean()
    return out


def volatility_target(raw):
    """
    설명: 변동성 target |log(close_t/close_(t-1)) x 100|을 만든다.
    주의 사항: 단위는 %. baselines.compute_log_returns_pct와 같은 척도여야 baseline과 비교가 맞는다.
    입력 타입->반환 타입: (pd.DataFrame) -> pd.Series
    """
    target = (np.log(raw["close"] / raw["close"].shift(1)) * 100).abs()
    target.name = "target"
    return target
