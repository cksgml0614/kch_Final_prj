# pooled_dataset.py — Task T 다종목 pooled 모델(2026-09-06, 별도 실험 트랙).
#
# 종목별 V2 피처(build_merged_dataset_v2)를 각자 독립적으로 계산한 뒤, 종목별 lookback
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

import numpy as np
import pandas as pd

from 가격예측.sequence_dataset import build_sequences, split_sequences_by_date
from 가격예측.split_dataset import TRAIN_RATIO, VAL_RATIO, build_merged_dataset_v2


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


def build_pooled_sequences(tickers, start_date, end_date, lookback=20, build_fn=build_merged_dataset_v2):
    """종목별 V2 피처 계산 + 종목별 시퀀스 구성 + 전역 날짜 컷 분할 + pooling.

    build_fn: 종목 하나의 (merged, meta)를 반환하는 함수(기본 build_merged_dataset_v2, 방향
    예측용). 2026-09-06 변동성 예측 실험에서 build_merged_dataset_v2_volatility를 넘겨
    재사용할 수 있도록 매개변수화했다 — 기본값을 그대로 두면 기존 호출부(가격예측_통합모델.py,
    test_evaluation_pooled.py)는 동작이 전혀 바뀌지 않는다.

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

    ticker_to_id = {t: i for i, t in enumerate(tickers)}
    merged_by_ticker = {}
    feature_cols = None

    for ticker in tickers:
        merged, _ = build_fn(ticker, start_date, end_date)
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


if __name__ == "__main__":
    import sys

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
    from datetime import date

    splits, feature_cols, ticker_to_id, (train_end, val_end) = build_pooled_sequences(
        ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START, date.today().isoformat(), lookback=20
    )
    print(f"종목 {len(ACTIVE_TICKERS)}개: {ticker_to_id}")
    print(f"전역 분할 경계: train_end={train_end.date()}  val_end={val_end.date()}")
    print(f"feature_cols({len(feature_cols)}개): {feature_cols}")
    for name, (X, y, tid, d, tname) in splits.items():
        print(f"\n[{name}] X{X.shape} y{y.shape}  종목별 시퀀스 수:")
        for ticker, idx in ticker_to_id.items():
            n = int((tid == idx).sum())
            print(f"  {ticker}: {n}개")
