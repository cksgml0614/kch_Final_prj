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

import inspect

import numpy as np
import pandas as pd

from 가격예측.momentum_feature import load_true_kospi
from 가격예측.sequence_dataset import build_sequences, split_sequences_by_date
from 가격예측.split_dataset import TRAIN_RATIO, VAL_RATIO, build_merged_dataset_v2
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


def compute_rolling_split_dates(reference_dates, val_days=60, test_days=0):
    """2026-09-15 단발성 진단용(D-4 후속) — compute_global_split_dates()의 비율(70/15/15)
    기반 분할은 전체 ~6.6년 데이터 중 val 구간이 거래일 캘린더 앞쪽 절반 근처(실측: 2024년
    하반기)에 거의 고정되는 부작용이 있다. `--end`를 하루씩 움직여도 val 윈도우 자체는
    몇 주 단위로만 미세 이동해 "여러 날짜로 반복 검증"의 실효성이 떨어진다(19영업일
    워크포워드 백테스트에서 실측 확인 — CLAUDE.md "19영업일 워크포워드 백테스트" 절 참고).

    이 함수는 그 대안 가설(val 윈도우를 데이터 끝쪽 최근 구간에 고정하는 롤링 분할이 val
    게이트 판정을 바꾸는지)을 진단하기 위해 추가했다 — `compute_global_split_dates()`는
    100종목 검증을 통과한 정본이므로 절대 수정하지 않고 이 함수를 병렬로 추가만 한다.

    train_end를 "마지막 거래일 - val_days(-test_days)" 직전 거래일로 고정한다(비율이
    아니라 절대 일수 기준). val_days=60 기본값 근거: 거래일 기준 약 3개월(분기) 윈도우 —
    compute_global_split_dates()의 15% val 비율이 ~6.6년 데이터에서는 약 1년에 해당했던
    것과 대비해 "최근 국면"만 보도록 훨씬 좁히려는 의도이면서, 동시에 100종목 x 60일 =
    6,000 val 시퀀스로 RMSE 추정 자체의 통계적 안정성도 확보하려는 절충값이다(값은
    잠정적 — 재검증 여부가 결정되면 사람이 재검토할 것).

    반환: (train_end, val_end) — compute_global_split_dates()와 동일한 시그니처/의미."""
    dates = pd.Series(sorted(pd.DatetimeIndex(reference_dates).unique()))
    n = len(dates)
    if val_days + test_days >= n:
        raise ValueError(f"val_days+test_days({val_days + test_days})가 전체 거래일 수({n})보다 많거나 같음")
    train_end = dates.iloc[n - val_days - test_days - 1]
    val_end = dates.iloc[n - test_days - 1]
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

    shared_cache_kwargs = {}
    if _build_fn_accepts_shared_cache(build_fn):
        shared_cache_kwargs = {
            "precomputed_indicators": load_indicator_cache(end_date),
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


def build_pooled_sequences_with_split(tickers, start_date, end_date, train_end, val_end,
                                       lookback=20, build_fn=build_merged_dataset_v2):
    """build_pooled_sequences()와 로직은 완전히 동일하되, train_end/val_end를
    compute_global_split_dates()로 계산하지 않고 호출부가 직접 지정한다(2026-09-20, ablation
    walk-forward fold 검증에서 70/15/15가 아닌 임의 fold 경계가 필요해 추가). 기존
    build_pooled_sequences()는 계약을 바꾸지 않기 위해 손대지 않고 이 함수를 병렬로 둔다.

    반환: splits, feature_cols, ticker_to_id — build_pooled_sequences()와 동일 형태(단
    (train_end, val_end)는 호출부가 이미 알고 있으므로 반환하지 않는다)."""
    if not tickers:
        raise ValueError("tickers가 비어있음")
    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)

    ticker_to_id = {t: i for i, t in enumerate(tickers)}
    feature_cols = None
    buckets = {"train": [], "val": [], "test": []}

    shared_cache_kwargs = {}
    if _build_fn_accepts_shared_cache(build_fn):
        shared_cache_kwargs = {
            "precomputed_indicators": load_indicator_cache(end_date),
            "true_kospi": load_true_kospi(start_date, end_date),
        }

    for ticker in tickers:
        merged, _ = build_fn(ticker, start_date, end_date, **shared_cache_kwargs)
        if merged.empty:
            raise RuntimeError(f"{ticker}: {start_date}~{end_date} 구간에 유효 데이터 없음")
        cols = [c for c in merged.columns if c != "target"]
        if feature_cols is None:
            feature_cols = cols
        elif cols != feature_cols:
            raise RuntimeError(f"{ticker}: feature_cols가 다른 종목과 다름 — pooling 불가")

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

    return splits, feature_cols, ticker_to_id


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
