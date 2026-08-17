# split_dataset.py — Task T-1 체크포인트 3
# base 피처(체크포인트 1) + momentum 피처(체크포인트 2)를 merge한 실제 유효 행 기준으로
# 정상 레짐/스트레스 구간을 분리하고, 정상 레짐 내에서 70/15/15 시계열 분할한다.

import sys

import pandas as pd

from constants import STRESS_PERIOD_END, STRESS_PERIOD_START
from 가격예측.dataset_builder import build_base_dataset, build_base_dataset_v2
from 가격예측.momentum_feature import build_momentum_feature

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO는 나머지 전부 (반올림 오차를 test 쪽에 흡수)


def build_merged_dataset(ticker, start_date, end_date):
    """base(14피처+target) + momentum(1피처)을 merge하고, 둘 중 하나라도 NaN인 행은 제거한다."""
    base, raw = build_base_dataset(ticker, start_date, end_date)
    momentum, diag = build_momentum_feature(ticker, start_date, end_date)

    merged = base.copy()
    merged["excess_return_z_lag1"] = momentum
    before = len(merged)
    merged = merged.dropna(how="any")
    after = len(merged)

    return merged, {"base": base, "raw": raw, "momentum": momentum, "diag": diag,
                     "rows_before_momentum_dropna": before, "rows_after_momentum_dropna": after}


def build_merged_dataset_v2(ticker, start_date, end_date):
    """build_merged_dataset과 동일하나 build_base_dataset_v2(정상성 재설계 피처)를 쓴다.
    momentum(excess_return_z_lag1)은 이미 정상성 있는 z-score라 변경 없이 그대로 재사용."""
    base, raw = build_base_dataset_v2(ticker, start_date, end_date)
    momentum, diag = build_momentum_feature(ticker, start_date, end_date)

    merged = base.copy()
    merged["excess_return_z_lag1"] = momentum
    before = len(merged)
    merged = merged.dropna(how="any")
    after = len(merged)

    return merged, {"base": base, "raw": raw, "momentum": momentum, "diag": diag,
                     "rows_before_momentum_dropna": before, "rows_after_momentum_dropna": after}


def split_normal_regime(merged, stress_start):
    """merged를 stress_start 기준으로 정상/스트레스로 나누고, 정상 구간을 70/15/15로 분할한다."""
    stress_start_ts = pd.Timestamp(stress_start)

    normal = merged[merged.index < stress_start_ts]
    stress = merged[merged.index >= stress_start_ts]

    n = len(normal)
    n_train = round(n * TRAIN_RATIO)
    n_val = round(n * VAL_RATIO)

    train = normal.iloc[:n_train]
    val = normal.iloc[n_train:n_train + n_val]
    test = normal.iloc[n_train + n_val:]

    return train, val, test, normal, stress


if __name__ == "__main__":
    TICKER = "005930"
    START = "2020-01-02"
    END = "2026-07-31"  # 전체 구간(정상+스트레스) 통째로 merge 후 상수 기준 분리

    merged, meta = build_merged_dataset(TICKER, START, END)

    print(f"=== build_merged_dataset({TICKER}, {START}, {END}) ===")
    print(f"base(체크포인트1) 행 수: {len(meta['base'])}")
    print(f"momentum(체크포인트2) NaN 제외 유효 행 수: {meta['momentum'].notna().sum()}")
    print(f"merge 직후(dropna 전) 행 수: {meta['rows_before_momentum_dropna']}")
    print(f"merge 후 실제 유효 행 수(dropna 후, 최종 학습 가능 행 수): {meta['rows_after_momentum_dropna']}")
    print(f"merge로 추가 소실된 행 수: {meta['rows_before_momentum_dropna'] - meta['rows_after_momentum_dropna']}")
    print(f"컬럼: {list(merged.columns)}")

    train, val, test, normal, stress = split_normal_regime(merged, STRESS_PERIOD_START)

    print(f"\n=== 정상 레짐 / 스트레스 분리 (기준: STRESS_PERIOD_START={STRESS_PERIOD_START}) ===")
    print(f"정상 레짐: {len(normal)}행 ({normal.index.min().date()} ~ {normal.index.max().date()})")
    print(f"스트레스: {len(stress)}행 ({stress.index.min().date()} ~ {stress.index.max().date()})")
    print(f"합계 = merge 후 유효 행 수와 일치: {len(normal) + len(stress) == len(merged)}")

    print(f"\n=== 정상 레짐 내 70/15/15 분할 (실제 merge 유효 행 {len(normal)}행 기준) ===")
    print(f"train: {len(train)}행 ({train.index.min().date()} ~ {train.index.max().date()})")
    print(f"val:   {len(val)}행 ({val.index.min().date()} ~ {val.index.max().date()})")
    print(f"test:  {len(test)}행 ({test.index.min().date()} ~ {test.index.max().date()})")
    print(f"비율: train {len(train)/len(normal)*100:.1f}% / val {len(val)/len(normal)*100:.1f}% / "
          f"test {len(test)/len(normal)*100:.1f}%")

    # --- 스트레스 구간 상수와 test셋 경계 겹침 최종 확인 ---
    print("\n=== 스트레스 구간(2026-02-02~2026-07-31) vs test셋 경계 최종 확인 ===")
    stress_start_ts = pd.Timestamp(STRESS_PERIOD_START)
    stress_end_ts = pd.Timestamp(STRESS_PERIOD_END)

    test_max = test.index.max()
    no_overlap_test = test_max < stress_start_ts
    print(f"test셋 최종일 = {test_max.date()}  <  스트레스 시작일 = {stress_start_ts.date()}  →  겹침 없음: {no_overlap_test}")

    all_normal_before_stress = bool((normal.index < stress_start_ts).all())
    print(f"정상 레짐(train/val/test 전체) 모든 행이 스트레스 시작일 이전인가: {all_normal_before_stress}")

    all_stress_in_range = bool(((stress.index >= stress_start_ts) & (stress.index <= stress_end_ts)).all())
    print(f"스트레스 구간 모든 행이 [{STRESS_PERIOD_START}, {STRESS_PERIOD_END}] 안에 있는가: {all_stress_in_range}")

    gap_days = (stress.index.min() - test_max).days
    print(f"test 마지막날 → 스트레스 첫날 간격: {gap_days}일 (음수/0이면 겹침, CLAUDE.md 정의상 거래일 갭이라 "
          f"실제 거래일 기준으로는 test 다음 거래일이 스트레스 첫 거래일이어야 정상)")

    assert no_overlap_test and all_normal_before_stress and all_stress_in_range, "스트레스 구간과 정상 레짐이 겹친다 — 확인 필요"
    print("\n✅ 최종 확인: merge 후에도 스트레스 구간과 test셋(정상 레짐 전체) 겹침 없음.")
