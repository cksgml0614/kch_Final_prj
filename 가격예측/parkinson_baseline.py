# parkinson_baseline.py — Parkinson(고가-저가 범위) 변동성 추정량 계산(2026-09-06, 원래
# parkinson_volatility_check.py). recent_vol_ma20/GARCH σ보다 유용한 정보를 주는지 검증한
# 실험(build_test_frame/main)은 검증용/parkinson_baseline_check.py로 분리됐다(2026-09-12,
# 가격예측/ 활성·검증 재분리) — 그 검증 결과 Parkinson-SMA20이 GARCH·SMA20을 이기는 유일한
# baseline으로 확인돼(결과_TaskT_변동성예측_최종.md), 아래 두 함수 + PARK_WINDOW가
# 가격예측_변동성_공통.py의 활성 피처 계산에 쓰인다.
#
# ⚠️ 리크 방지 설계: Parkinson 추정량 자체(parkinson_vol(t), high(t)/low(t) 사용)는 t일
# 자신의 장중 정보라 target(t)(=|종가수익률(t)|)을 예측하는 피처로 쓰면 리크다. 호출부가
# 반드시 rolling().shift(1)로 시차를 준 값("Parkinson-SMA20")만 예측 피처로 사용해야 한다 —
# compute_parkinson_vol_pct() 자체는 그 시차 처리 이전의 원시값을 반환한다.

import numpy as np
import pandas as pd

from db_manager import get_db_connection

PARK_WINDOW = 20


def load_high_low(ticker, start_date, end_date):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT date, high, low FROM daily_stock_prices WHERE ticker=%s AND date BETWEEN %s AND %s ORDER BY date",
                (ticker, start_date, end_date),
            )
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["date", "high", "low"]).astype({"high": float, "low": float})
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def compute_parkinson_vol_pct(high, low):
    """parkinson_vol(t) = sqrt(1/(4 ln2) * ln(high/low)^2), x100(%) 스케일 통일.
    당일 H/L을 쓰므로 그 자체는 t일 정보 — 리크 방지 주석 참고."""
    raw = np.sqrt((np.log(high / low) ** 2) / (4 * np.log(2)))
    return raw * 100
