# lag1_baseline_비교.py
# Lag-1 "어제 값 그대로" persistence baseline 추가 비교(2026-09-15) — 순수 읽기 전용 진단.
# DB에 아무것도 쓰지 않는다(SELECT만). 학습 코드/GARCH 캐시/운영 파이프라인/기존
# actual_volatility_비교.py는 전혀 건드리지 않고 완전히 새 스크립트로 작성했다.
#
# 배경: 지금까지 baseline 3종(GARCH, SMA20, Parkinson-SMA20)은 전부 "여러 날 평균/감쇠"
# 방식이다. 가장 단순한 lag-1(어제 하루의 실제 변동성을 그대로 내일 예측값으로 쓰는 것)은
# 한 번도 비교 대상에 넣은 적이 없었다. 변동성 클러스터링 성질상, 변동성이 급증하는
# 구간에서는 이 단순한 baseline이 분산 붕괴 경향이 있는 하이브리드보다 나을 수 있다는
# 가설을 19일 백테스트(2026-08-14~09-10, D-4)의 기존 actual_volatility로 검증한다 —
# 새로 모델을 돌리거나 재학습하지 않는다.
#
# 공식(garch_baseline.compute_log_returns_pct와 완전히 동일한 형태, 기존 target/
# actual_volatility 정의와 척도 일관성 유지):
#   lag1_baseline(ticker, target_date) = |log(close_prediction_date / close_그_직전_거래일)| x 100
# prediction_date는 model_predictions에 이미 저장된 값(target_date의 직전 확정 거래일,
# 즉 t-1) 그대로 쓴다. "그 직전 거래일"(t-2)은 actual_volatility_백필.py와 동일하게 단순
# 날짜 뺄셈이 아니라 daily_stock_prices에서 prediction_date 이전 가장 최근 실제 거래일을
# 찾는다(주말/공휴일 gap-safe).
#
# 대상 범위: 19일 백테스트(2026-08-14~09-10 입력일 → target_date 2026-08-17~09-11)만.
# 08-17(공휴일이라 actual_volatility NULL)은 WHERE actual_volatility IS NOT NULL로 자동
# 제외. 09-14/09-15 프로덕션 102행은 target_date < 2026-09-14 조건으로 범위 밖 처리.

import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from db_manager import get_db_connection

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUERY = """
    SELECT ticker, target_date, prediction_date, predicted_volatility, actual_volatility,
           garch_baseline, sma20_baseline, parkinson_sma20_baseline
    FROM model_predictions
    WHERE actual_volatility IS NOT NULL AND target_date < '2026-09-14'
    ORDER BY target_date, ticker
"""

ERR_COLS = [
    ("하이브리드", "err_hybrid"),
    ("GARCH", "err_garch"),
    ("SMA20", "err_sma20"),
    ("Parkinson-SMA20", "err_parkinson"),
    ("Lag-1", "err_lag1"),
]


def load_rows():
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패 — load_rows")
        with conn.cursor() as cur:
            cur.execute(QUERY)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def fetch_close_series(ticker, end_date):
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패 — fetch_close_series")
        with conn.cursor() as cur:
            cur.execute(
                "SELECT date, close FROM daily_stock_prices WHERE ticker = %s AND date <= %s ORDER BY date",
                (ticker, end_date),
            )
            rows = cur.fetchall()
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows, columns=["date", "close"])
    df["close"] = df["close"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"]


def compute_lag1(close_series, prediction_ts):
    """garch_baseline.compute_log_returns_pct와 동일 정의: |log(close_t/close_(t-1))| x 100.
    여기서 t=prediction_date, t-1은 daily_stock_prices에서 prediction_date 이전 가장
    최근 실제 거래일(actual_volatility_백필.py와 동일 원칙 — 단순 날짜 뺄셈 아님)."""
    if prediction_ts not in close_series.index:
        return None
    prior = close_series[close_series.index < prediction_ts]
    if prior.empty:
        return None
    close_t = close_series.loc[prediction_ts]
    close_prev = prior.iloc[-1]
    return abs(float(np.log(close_t / close_prev) * 100))


def print_summary(label, sub_df):
    n = len(sub_df)
    print(f"\n=== {label} (N={n}) ===")
    print("MAE:")
    for name, col in ERR_COLS:
        print(f"  {name:16s}: {sub_df[col].mean():.4f}")
    print("하이브리드 승률(하이브리드 |오차| < baseline |오차|):")
    for name, col in ERR_COLS[1:]:
        wins = int((sub_df["err_hybrid"] < sub_df[col]).sum())
        print(f"  vs {name:16s}: {wins}/{n} ({wins / n * 100:.1f}%)")


def main():
    df = load_rows()
    print(f"대상 행 수(백필 완료분): {len(df)} (target_date {df['target_date'].min()} ~ {df['target_date'].max()})")

    df["prediction_date"] = pd.to_datetime(df["prediction_date"])

    lag1_by_index = {}
    skip_reasons = defaultdict(int)
    for ticker, group in df.groupby("ticker"):
        max_pred_date = group["prediction_date"].max().date()
        close_series = fetch_close_series(ticker, max_pred_date)
        for idx, row in group.iterrows():
            lag1 = compute_lag1(close_series, row["prediction_date"])
            if lag1 is None:
                skip_reasons["직전 거래일 데이터 부족"] += 1
            lag1_by_index[idx] = lag1

    df["lag1_baseline"] = pd.Series(lag1_by_index)
    n_missing = df["lag1_baseline"].isna().sum()
    if n_missing:
        print(f"⚠️ lag1_baseline 계산 불가로 제외된 행: {n_missing}개 — {dict(skip_reasons)}")
    df = df.dropna(subset=["lag1_baseline"]).copy()

    for col in ["predicted_volatility", "actual_volatility", "garch_baseline",
                "sma20_baseline", "parkinson_sma20_baseline", "lag1_baseline"]:
        df[col] = df[col].astype(float)

    df["err_hybrid"] = (df["predicted_volatility"] - df["actual_volatility"]).abs()
    df["err_garch"] = (df["garch_baseline"] - df["actual_volatility"]).abs()
    df["err_sma20"] = (df["sma20_baseline"] - df["actual_volatility"]).abs()
    df["err_parkinson"] = (df["parkinson_sma20_baseline"] - df["actual_volatility"]).abs()
    df["err_lag1"] = (df["lag1_baseline"] - df["actual_volatility"]).abs()

    print_summary("전체 19일 통합", df)

    threshold = df["actual_volatility"].quantile(0.8)
    top_df = df[df["actual_volatility"] >= threshold]
    print(f"\n(actual_volatility 상위 20% 임계값: {threshold:.4f})")
    print_summary("변동성 상위 20% 부분집합 — 핵심 확인 대상", top_df)

    print(f"\n=== 일자별 하이브리드 vs Lag-1 (전체 {df['target_date'].nunique()}일) ===")
    daily = df.groupby("target_date").agg(
        n=("err_hybrid", "size"),
        mae_hybrid=("err_hybrid", "mean"),
        mae_lag1=("err_lag1", "mean"),
    )
    win_rate = df.groupby("target_date").apply(
        lambda g: (g["err_hybrid"] < g["err_lag1"]).mean(), include_groups=False
    )
    daily["hybrid_win_vs_lag1"] = win_rate
    pd.set_option("display.width", 200)
    print(daily.round(4).to_string())

    print("\n⚠️ 단일 19일 백테스트 구간(2026-08-14~09-10) 기준 관찰이며, 다른 구간/기간에")
    print("대한 추가 검증 없이는 일반화할 수 없다. 결론이 아니라 신호 확인용 참고 자료.")


if __name__ == "__main__":
    main()
