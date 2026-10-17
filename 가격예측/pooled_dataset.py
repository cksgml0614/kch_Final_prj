# pooled_dataset.py — 종목별 피처 frame으로 종목별 lookback 시퀀스를 만들고, 전역 날짜로 나눠 하나의 pooled 모집단으로 합친다.
# 파이프라인 단계: 시퀀스. 주요 진입 함수: build_pooled_sequences, compute_global_split_dates.
# 한 시퀀스 안에는 한 종목의 연속 구간만 들어간다(종목 경계를 넘지 않음).
# 배경: 결정기록.md의 "전역 분할 경계는 대표 종목 하나의 거래일로 정한다"

import inspect

import numpy as np
import pandas as pd

from 가격예측.momentum_feature import load_true_kospi
from 가격예측.sequence_dataset import build_sequences, split_sequences_by_date
from 가격예측.split_dataset import TRAIN_RATIO, VAL_RATIO
from 시장지표.feature_loader import load_indicator_cache


def compute_global_split_dates(reference_dates, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """
    설명: 날짜 목록을 정렬해 train·val 비율로 자른 마지막 날짜(train_end, val_end)를 돌려준다.
    주의 사항: 전 종목이 같은 거래일 달력을 쓴다고 가정한다(상장일이 다른 종목이 들어오면 다시 설계해야 함).
    입력 타입->반환 타입: (DatetimeIndex-like, float, float) -> (pd.Timestamp, pd.Timestamp)
    """
    dates = pd.Series(sorted(pd.DatetimeIndex(reference_dates).unique()))
    n = len(dates)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)
    train_end = dates.iloc[n_train - 1]
    val_end = dates.iloc[n_train + n_val - 1]
    return train_end, val_end


def _build_fn_accepts_shared_cache(build_fn):
    """
    설명: build_fn이 precomputed_indicators·true_kospi 키워드 인자를 받는지 시그니처로 확인한다.
    주의 사항: 인자 이름으로 판정하므로 build_fn의 인자 이름을 바꾸면 캐시가 조용히 안 넘어간다.
    입력 타입->반환 타입: (Callable) -> bool
    """
    try:
        params = inspect.signature(build_fn).parameters
    except (TypeError, ValueError):
        return False
    return "precomputed_indicators" in params and "true_kospi" in params


def build_pooled_sequences(tickers, start_date, end_date, lookback=20, build_fn=None,
                           shared_cache=None):
    """
    설명: 종목별 frame·시퀀스를 만들고 전역 경계로 나눠 {"train"|"val"|"test": (X, y, ticker_ids, dates, tickers)}로 합친다.
    주의 사항: build_fn 필수. shared_cache는 같은 start/end로 만든 것이어야 하고, 없으면 DB를 읽는다. 첫 종목이 경계 기준.
    입력 타입->반환 타입: (list[str], str, str, int, Callable, dict|None) -> (dict, list[str], dict, (Timestamp, Timestamp))
    """
    if not tickers:
        raise ValueError("tickers가 비어있음")
    if build_fn is None:
        raise ValueError("build_fn은 필수다")

    ticker_to_id = {t: i for i, t in enumerate(tickers)}
    merged_by_ticker = {}
    feature_cols = None

    shared_cache_kwargs = {}
    if _build_fn_accepts_shared_cache(build_fn):
        if shared_cache is not None:
            shared_cache_kwargs = {
                "precomputed_indicators": shared_cache["precomputed_indicators"],
                "true_kospi": shared_cache["true_kospi"],
            }
        else:
            shared_cache_kwargs = {
                "precomputed_indicators": load_indicator_cache(end_date, start_date),
                "true_kospi": load_true_kospi(start_date, end_date),
            }

    for ticker in tickers:
        merged, _ = build_fn(ticker, start_date, end_date, **shared_cache_kwargs)
        if merged.empty:
            raise RuntimeError(f"{ticker}: {start_date}~{end_date} 구간에 유효 데이터 없음")
        if feature_cols is None:
            feature_cols = [c for c in merged.columns if c != "target"]
        elif [c for c in merged.columns if c != "target"] != feature_cols:
            raise RuntimeError(f"{ticker}: feature_cols가 다른 종목과 다름 — pooling 불가")
        merged_by_ticker[ticker] = merged

    reference_dates = merged_by_ticker[tickers[0]].index
    train_end, val_end = compute_global_split_dates(reference_dates)

    buckets = {"train": [], "val": [], "test": []}
    for ticker in tickers:
        merged = merged_by_ticker[ticker]
        X, y, dates = build_sequences(merged, feature_cols, lookback)
        if len(X) == 0:
            raise RuntimeError(f"{ticker}: lookback({lookback})을 만족하는 시퀀스가 없음")

        per_ticker_splits = split_sequences_by_date(X, y, dates, train_end, val_end)
        for split_name, (X_s, y_s, d_s) in per_ticker_splits.items():
            if len(X_s) == 0:
                continue
            tid = np.full(len(X_s), ticker_to_id[ticker], dtype=np.int64)
            tname = np.full(len(X_s), ticker, dtype=object)
            buckets[split_name].append((X_s, y_s, tid, d_s, tname))

    splits = {}
    for split_name, parts in buckets.items():
        if not parts:
            raise RuntimeError(f"{split_name}: 어떤 종목도 이 split에 시퀀스를 내지 못함")
        splits[split_name] = (
            np.concatenate([p[0] for p in parts], axis=0).astype(np.float32),
            np.concatenate([p[1] for p in parts], axis=0).astype(np.float32),
            np.concatenate([p[2] for p in parts], axis=0),
            np.concatenate([p[3] for p in parts], axis=0),
            np.concatenate([p[4] for p in parts], axis=0),
        )

    return splits, feature_cols, ticker_to_id, (train_end, val_end)
