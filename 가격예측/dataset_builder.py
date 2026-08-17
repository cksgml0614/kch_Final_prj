# dataset_builder.py — Task T-1 체크포인트 1
# 레이블(target_t)과 기본 피처(OHLCV + 시장지표 9종, t-1 정렬)를 계산한다.
# excess_return_z 모멘텀 피처는 체크포인트 2(momentum_feature.py)에서 별도로 추가한다.

import sys

import numpy as np
import pandas as pd

from 시장지표.feature_loader import get_features

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def build_base_dataset(ticker, start_date, end_date):
    """레이블·기본 피처 결합 데이터셋을 반환한다.

    target_t = (close_t - close_{t-1}) / close_{t-1}   (TASK_T 설계결정 1)
    feature_t = raw_features_(t-1)                      (TASK_T 설계결정 2: 당일 값 사용 불가)

    raw는 feature_loader.get_features()가 반환하는, 거래일 순서로 정렬된 DataFrame이므로
    .shift(1)은 달력일이 아니라 "직전 거래일" 기준으로 정확히 동작한다 (주말/휴장일 자동 처리).

    Returns
    -------
    combined : DataFrame, index=target date t, columns=[raw의 14개 피처..., 'target']
               첫 행(직전 거래일이 없는 행)은 NaN이라 제거됨.
    raw : DataFrame, feature_loader 원본 (검증용으로 함께 반환)
    """
    raw = get_features(ticker, start_date, end_date)
    if raw.empty:
        raise RuntimeError(f"{ticker}: {start_date}~{end_date} 구간에 가격 데이터 없음")

    target = raw["close"].pct_change()
    target.name = "target"

    features = raw.shift(1)

    combined = features.copy()
    combined["target"] = target
    combined = combined.dropna(how="any")

    return combined, raw


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


def build_base_dataset_v2(ticker, start_date, end_date):
    """build_base_dataset과 동일한 t-1 정렬 방식이되, build_stationary_features로 재설계한
    피처 세트를 쓴다. target 정의는 동일 — close 레벨은 target 계산에만 쓰고 피처에서는 뺀다."""
    raw = get_features(ticker, start_date, end_date)
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


def verify_index_alignment(raw, combined, n=10, seed=42):
    """combined의 각 target 날짜 t에 대해, 실제로 사용된 피처 행이 raw 기준
    '직전 거래일(t-1)'과 정확히 일치하는지 명시적으로 검증한다 (shift 연산을 맹신하지 않음).
    """
    dates = list(raw.index)
    date_pos = {d: i for i, d in enumerate(dates)}

    sample_dates = pd.Series(combined.index).sample(min(n, len(combined)), random_state=seed).sort_values()

    print(f"\n=== 인덱스 정렬 검증: target t 대비 피처가 t-1인지, 샘플 {len(sample_dates)}건 ===")
    all_ok = True
    for t in sample_dates:
        i = date_pos[t]
        if i == 0:
            print(f"  [SKIP] {t.date()}: raw의 첫 행(직전 거래일 없음)")
            continue
        prev_date = dates[i - 1]

        # (a) 날짜 자체가 target보다 이전인가
        date_ok = prev_date < t

        # (b) combined에 저장된 피처값이 실제로 raw.loc[prev_date]와 동일한가 (shift 결과 직접 대조)
        value_ok = bool((combined.loc[t, raw.columns] == raw.loc[prev_date]).all())

        # (c) target 값 자체가 정의(close_t/close_{t-1} - 1)와 일치하는가
        expected_target = raw.loc[t, "close"] / raw.loc[prev_date, "close"] - 1
        target_ok = abs(combined.loc[t, "target"] - expected_target) < 1e-9

        ok = date_ok and value_ok and target_ok
        all_ok &= ok
        mark = "OK" if ok else "FAIL"
        print(
            f"  [{mark}] target일={t.date()}  사용된 피처일={prev_date.date()}  "
            f"date_ok={date_ok} value_ok={value_ok} target_ok={target_ok}  "
            f"target={combined.loc[t, 'target']:+.4%}"
        )

    print(f"전체 통과 (피처일 < target일 AND 값 일치 AND target 정의 일치): {all_ok}")
    return all_ok


if __name__ == "__main__":
    TICKER = "005930"
    START = "2020-01-02"
    END = "2026-02-01"  # 정상 레짐 종료일 (TASK_T 확정: 스트레스 구간 2026-02-02~ 제외)

    combined, raw = build_base_dataset(TICKER, START, END)

    print(f"=== build_base_dataset({TICKER}, {START}, {END}) ===")
    print(f"raw shape (feature_loader 원본): {raw.shape}")
    print(f"combined shape (레이블 결합, 첫 행 제거 후): {combined.shape}")
    print(f"columns: {list(combined.columns)}")

    dropped = raw.shape[0] - combined.shape[0]
    print(f"제거된 행(첫 거래일, 직전 거래일 없음): {dropped}")

    verify_index_alignment(raw, combined, n=10)

    print("\n=== combined 앞 3행 (target, close만) ===")
    print(combined[["close", "target"]].head(3))
    print("\n=== combined 뒤 3행 (target, close만) ===")
    print(combined[["close", "target"]].tail(3))
