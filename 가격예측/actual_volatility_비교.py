# actual_volatility_비교.py
# actual_volatility_백필.py로 채운 model_predictions를 대상으로 하이브리드 Transformer
# 예측과 GARCH/SMA20/Parkinson-SMA20 baseline 3종의 오차를 비교한다(2026-09-14). DB를
# 읽기만 하고 아무 것도 쓰지 않는다 — 백필 후 언제든 재실행 가능(라벨_생성.py/라벨_보고.py
# 의 "쓰기/읽기 분리" 선례와 동일 패턴).

import sys

import numpy as np
import pandas as pd

from db_manager import get_db_connection

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

COMPARISON_QUERY = """
    SELECT ticker, target_date, predicted_volatility, actual_volatility,
           garch_baseline, sma20_baseline, parkinson_sma20_baseline, gate_passed
    FROM model_predictions
    WHERE actual_volatility IS NOT NULL
    ORDER BY target_date, ABS(predicted_volatility - actual_volatility) DESC
"""

NUMERIC_COLS = [
    "predicted_volatility", "actual_volatility",
    "garch_baseline", "sma20_baseline", "parkinson_sma20_baseline",
]

BASELINE_COLS = [
    ("GARCH", "garch_baseline", "err_garch"),
    ("SMA20", "sma20_baseline", "err_sma20"),
    ("Parkinson-SMA20", "parkinson_sma20_baseline", "err_parkinson"),
]


def load_comparison_rows():
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패 — load_comparison_rows")
        with conn.cursor() as cur:
            cur.execute(COMPARISON_QUERY)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def summarize(df):
    for col in NUMERIC_COLS:
        df[col] = df[col].astype(float)

    df["err_hybrid"] = (df["predicted_volatility"] - df["actual_volatility"]).abs()
    for _, base_col, err_col in BASELINE_COLS:
        df[err_col] = (df[base_col] - df["actual_volatility"]).abs()

    n = len(df)
    print(f"비교 대상 행 수: {n} (target_date 범위: {df['target_date'].min()} ~ {df['target_date'].max()})")

    print("\n=== 평균 절대오차(MAE, 값이 작을수록 정확) ===")
    print(f"하이브리드      : {df['err_hybrid'].mean():.4f}")
    for name, _, err_col in BASELINE_COLS:
        print(f"{name:16s}: {df[err_col].mean():.4f}")

    print("\n=== 종목별 승률(하이브리드 |오차| < baseline |오차|인 건수) ===")
    for name, _, err_col in BASELINE_COLS:
        wins = int((df["err_hybrid"] < df[err_col]).sum())
        losses = int((df["err_hybrid"] > df[err_col]).sum())
        ties = n - wins - losses
        print(f"vs {name:16s}: {wins}/{n} 승  {losses}/{n} 패  {ties}/{n} 동률")

    return df


def main():
    df = load_comparison_rows()
    if df.empty:
        print("actual_volatility가 채워진 행이 없습니다 — 먼저 actual_volatility_백필.py를 실행하세요.")
        return
    summarize(df)


if __name__ == "__main__":
    main()
