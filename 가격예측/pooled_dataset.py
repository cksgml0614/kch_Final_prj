# pooled_dataset.py — 다종목 pooled 시퀀스 구성과 전역 날짜 분할(2026-09-06, 운영 경로).
# 2026-10-17 정리: compute_rolling_split_dates·build_pooled_sequences_with_split(진단·검증용)을 지웠다.
#
# 종목별 피처 frame(호출부가 넘기는 build_fn, 운영은 GARCH σ 포함 15피처)을 각자 독립적으로 계산한 뒤, 종목별 lookback
# 시퀀스를 만들고(시퀀스 내부는 항상 한 종목의 연속 구간만 사용 — 종목 경계를 넘지 않음),
# 전 종목 공유 캘린더 기준 날짜 컷으로 train/val/test를 나눠 하나의 pooled 모집단으로 합친다.
#
# ⚠️ 전제: 8종목(2026-09-06 활성화)이 완전히 동일한 거래일 캘린더를 공유함을 실측 확인했다
# (전 종목 1,639행, 2020-01-02~2026-09-04 — 상장일 차이로 인한 시작일 불일치 없음). 이 전제
# 덕분에 대표 종목 하나의 날짜만으로 전역 분할 경계를 구해도 안전하다. 향후 종목이 추가되어
# 캘린더가 어긋나는 경우(예: 최근 상장 종목)가 생기면 compute_global_split_dates()를 "전
# 종목 날짜의 교집합" 기준으로 다시 설계해야 한다 — 지금은 그 필요가 없어 구현하지 않았다.
#
# "풀링"은 서로 다른 종목의 (독립적인) 시퀀스들을 하나의 학습 배치 모집단으로 합치는 것뿐이다
# — 한 시퀀스의 lookback 구간 안에 서로 다른 종목의 값이 섞이는 일은 없다.

import inspect

import numpy as np
import pandas as pd

from 가격예측.momentum_feature import load_true_kospi
from 가격예측.sequence_dataset import build_sequences, split_sequences_by_date
from 가격예측.split_dataset import TRAIN_RATIO, VAL_RATIO
from 시장지표.feature_loader import load_indicator_cache


def compute_global_split_dates(reference_dates, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """공유 거래일 캘린더 기준으로 train/val 경계 날짜를 구한다(파일 상단 전제 참고 — 대표
    종목 하나의 날짜만으로 충분한 것은 전 종목 캘린더가 동일하기 때문)."""
    dates = pd.Series(sorted(pd.DatetimeIndex(reference_dates).unique()))
    n = len(dates)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)
    train_end = dates.iloc[n_train - 1]
    val_end = dates.iloc[n_train + n_val - 1]
    return train_end, val_end


def _build_fn_accepts_shared_cache(build_fn):
    """build_fn이 precomputed_indicators/true_kospi 키워드 인자를 받는지 확인한다(2026-09-20,
    100종목 피처 생성 성능 최적화 — 종목과 무관한 거시지표 9종·KOSPI 원자료를 매 종목 재조회
    하던 것을 없애기 위함). inspect로 한 번만 확인하고 종목 루프 전체에서 재사용한다.

    이 검사가 필요한 이유: build_fn은 호출부(가격예측_변동성_공통.py, ablation_*.py 등)가
    자유롭게 만드는 클로저라 시그니처를 강제할 수 없다. 캐시를 지원하도록 아직 갱신되지 않은
    build_fn(옛 검증용/ 스크립트 등)은 이 검사에서 False가 나와 캐시 없이 기존과 완전히
    동일하게(매 종목 재조회) 동작한다 — 하위 호환이 깨지지 않는다."""
    try:
        params = inspect.signature(build_fn).parameters
    except (TypeError, ValueError):
        return False
    return "precomputed_indicators" in params and "true_kospi" in params


def build_pooled_sequences(tickers, start_date, end_date, lookback=20, build_fn=None,
                           shared_cache=None):
    """종목별 피처 계산 + 종목별 시퀀스 구성 + 전역 날짜 컷 분할 + pooling.

    build_fn(필수): 종목 하나의 (merged, meta)를 반환하는 함수. 운영은 가격예측_변동성_공통의 hybrid_build_fn
    (build_merged_dataset_v2_volatility_hybrid). 2026-10-17: 기본값(방향 예측 build_merged_dataset_v2)을 지웠다.

    shared_cache(선택, 2026-10-05): {"precomputed_indicators": load_indicator_cache(...),
    "true_kospi": load_true_kospi(start_date, end_date)}를 호출부가 이미 로드했으면 그대로 넘긴다
    — 같은 실행의 예측 단계와 캐시를 함께 써서 거시지표를 한 번만 읽기 위함. 같은 start_date/
    end_date로 만든 것이어야 한다(호출부 책임). None이면 여기서 새로 로드한다(기존 동작).

    반환
    -------
    splits : dict, {"train": (X, y, ticker_ids, dates, tickers), "val": (...), "test": (...)}
             X: (N, lookback, n_features) float32, y: (N,) float32,
             ticker_ids: (N,) int64(0..len(tickers)-1), dates: (N,) datetime64,
             tickers: (N,) object(종목코드 문자열, 로깅/디버깅용)
    feature_cols : list[str]
    ticker_to_id : dict{ticker: id}
    (train_end, val_end) : 전역 분할 경계 날짜(포함 기준 — train_end까지 train, 그 다음날부터
             val_end까지 val, 그 이후 전부 test)
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
