# 라벨_보고.py
# Task E "보고 항목"(완료 조건) 출력 전용 — 정지 지점. daily_labels(z_score만 저장)를 조회해
# 라벨_공통.label_*() 함수로 임계값을 적용, 조회 시점에 라벨을 파생해 분포를 출력한다.
#
# ⚠️ 임계값을 확정하지 않는다 — TASK_EF_라벨링_비교실험.md "판단 정지 지점": 후보 전부를 분포와
# 함께 출력하고 사람이 최종 임계값을 고른다. 분위수 기반 임계값은 쓰지 않는다(D-2, 미래 정보 누수).
#
# 조회 전용: daily_labels를 포함해 어떤 테이블에도 쓰지 않는다. 뉴스 테이블(daily_news,
# daily_news_bigkinds)은 아예 건드리지 않는다.

import sys

import numpy as np
import pandas as pd

from constants import ACTIVE_TICKERS, TRAIN_PERIOD_END, TRAIN_PERIOD_START
from db_manager import get_db_connection
from 라벨.라벨_공통 import (
    DIRECTION_3CLASS_THRESHOLD,
    DIRECTION_5CLASS_THRESHOLDS,
    VOLATILITY_2CLASS_THRESHOLDS,
    WINDOW_SIZES,
    compute_z_scores,
    label_direction_3class,
    label_direction_5class,
    label_volatility_2class,
    load_labels,
    split_dates_by_ratio,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HIST_BINS = [-np.inf, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, np.inf]
HIST_LABELS = ["<-2", "-2~-1.5", "-1.5~-1", "-1~-0.5", "-0.5~0", "0~0.5", "0.5~1", "1~1.5", "1.5~2", ">2"]


def print_z_distribution(df):
    print("\n" + "=" * 78)
    print("[1] z 분포 — 윈도우 N별 기술통계 + 히스토그램")
    print("=" * 78)
    for n in WINDOW_SIZES:
        z = df.loc[df["window_n"] == n, "z_score"].dropna()
        print(f"\n--- window_n={n} (유효 z {len(z)}건) ---")
        desc = z.describe(percentiles=[0.25, 0.5, 0.75])
        print(desc.to_string())
        hist = pd.cut(z, bins=HIST_BINS, labels=HIST_LABELS).value_counts().reindex(HIST_LABELS)
        print("히스토그램:")
        for label, cnt in hist.items():
            pct = cnt / len(z) * 100 if len(z) else 0.0
            print(f"  {label:>10}: {cnt:5d} ({pct:5.1f}%)")


def _class_counts(z_series, label_fn, classes, **kwargs):
    labels = z_series.apply(lambda v: label_fn(v, **kwargs))
    counts = labels.value_counts().reindex(classes, fill_value=0)
    total = int(counts.sum())
    return counts, total


def print_label_distributions(df, header):
    print(f"\n--- 라벨 분포 [{header}] ---")
    for n in WINDOW_SIZES:
        z = df.loc[df["window_n"] == n, "z_score"]
        print(f"\n  window_n={n}")

        counts, total = _class_counts(
            z, label_direction_5class, [-2, -1, 0, 1, 2],
            t1=DIRECTION_5CLASS_THRESHOLDS[0], t2=DIRECTION_5CLASS_THRESHOLDS[1],
        )
        print(f"    방향 5-class (z ±{DIRECTION_5CLASS_THRESHOLDS[0]}/±{DIRECTION_5CLASS_THRESHOLDS[1]}), 유효 {total}건:")
        for cls, cnt in counts.items():
            pct = cnt / total * 100 if total else 0.0
            print(f"      {cls:+d}: {cnt:5d} ({pct:5.1f}%)")

        counts, total = _class_counts(z, label_direction_3class, [-1, 0, 1], t=DIRECTION_3CLASS_THRESHOLD)
        print(f"    방향 3-class (z ±{DIRECTION_3CLASS_THRESHOLD}), 유효 {total}건:")
        for cls, cnt in counts.items():
            pct = cnt / total * 100 if total else 0.0
            print(f"      {cls:+d}: {cnt:5d} ({pct:5.1f}%)")

        for t in VOLATILITY_2CLASS_THRESHOLDS:
            counts, total = _class_counts(z, label_volatility_2class, [0, 1], t=t)
            print(f"    변동성 2-class (|z|>{t}), 유효 {total}건:")
            for cls, cnt in counts.items():
                tag = "고변동" if cls == 1 else "정상"
                pct = cnt / total * 100 if total else 0.0
                print(f"      {tag}({cls}): {cnt:5d} ({pct:5.1f}%)")


def print_regime_stats(raw_df, splits):
    """D-8 레짐 특성 3종: 전 거래일 평균|cr|, 평균|초과수익률|, KOSPI 평균|cr|.
    window_n과 무관(raw change_rate/excess_return은 윈도우에 의존하지 않음) — 라벨_공통.
    compute_z_scores()가 반환하는 원 change_rate를 그대로 쓴다."""
    print("\n" + "=" * 78)
    print("[4] D-8 레짐 특성 3종 (split별)")
    print("=" * 78)
    for split_name, dates in splits:
        sub = raw_df[raw_df["date"].isin(dates)]
        mean_abs_cr = sub["stock_change_rate"].abs().mean() * 100
        mean_abs_excess = sub["excess_return"].abs().mean() * 100
        mean_abs_kospi = sub["kospi_change_rate"].abs().mean() * 100
        print(
            f"  {split_name:>5} ({len(sub):4d}일, {sub['date'].min().date()}~{sub['date'].max().date()}): "
            f"전 거래일 평균|cr| {mean_abs_cr:6.3f}% | 평균|초과수익률| {mean_abs_excess:6.3f}% | "
            f"KOSPI 평균|cr| {mean_abs_kospi:6.3f}%"
        )


def main():
    for ticker in ACTIVE_TICKERS:
        print("\n" + "#" * 78)
        print(f"# Task E 보고 — {ticker} (TRAIN_PERIOD {TRAIN_PERIOD_START} ~ {TRAIN_PERIOD_END})")
        print("#" * 78)

        df = load_labels(ticker)
        if df.empty:
            print(f"⚠️ {ticker}: daily_labels에 데이터 없음 — 라벨_생성.py를 먼저 실행하세요.")
            continue

        train_period_mask = (df["date"] >= pd.Timestamp(TRAIN_PERIOD_START)) & (df["date"] <= pd.Timestamp(TRAIN_PERIOD_END))
        df_tp = df[train_period_mask]

        print_z_distribution(df_tp)

        print("\n" + "=" * 78)
        print(f"[2] 라벨 분포 — 임계값 후보 x 윈도우 N 교차 (TRAIN_PERIOD 전체)")
        print("=" * 78)
        print_label_distributions(df_tp, f"전체 {TRAIN_PERIOD_START}~{TRAIN_PERIOD_END}")

        unique_dates = df_tp.loc[df_tp["window_n"] == WINDOW_SIZES[0], "date"].unique()
        train_dates, val_dates, test_dates = split_dates_by_ratio(unique_dates)
        splits = [("train", train_dates), ("val", val_dates), ("test", test_dates)]

        print("\n" + "=" * 78)
        print("[3] split별 라벨 분포 (train/val/test, 시계열 순서, SPLIT_RATIOS 기준)")
        print("=" * 78)
        print(
            f"train {len(train_dates)}일 ({train_dates.min().date()}~{train_dates.max().date()}) / "
            f"val {len(val_dates)}일 ({val_dates.min().date()}~{val_dates.max().date()}) / "
            f"test {len(test_dates)}일 ({test_dates.min().date()}~{test_dates.max().date()})"
        )
        for split_name, dates in splits:
            print_label_distributions(df_tp[df_tp["date"].isin(dates)], split_name)

        # D-8 레짐 특성 3종은 원 change_rate가 필요 — daily_labels에는 excess_return만 있으므로
        # 라벨_공통.compute_z_scores()를 다시 불러 원 change_rate를 붙인다(가벼운 재계산, 쓰기 없음)
        with get_db_connection() as conn:
            if not conn:
                raise RuntimeError("DB 연결 실패 — 라벨_보고 (레짐 특성)")
            with conn.cursor() as cur:
                raw = compute_z_scores(ticker, cur)
        raw_tp = raw[(raw["date"] >= pd.Timestamp(TRAIN_PERIOD_START)) & (raw["date"] <= pd.Timestamp(TRAIN_PERIOD_END))]
        print_regime_stats(raw_tp, [("전체", pd.concat([train_dates, val_dates, test_dates]))] + splits)


if __name__ == "__main__":
    main()
