# 대시보드/데이터.py — 새 대시보드의 조회 함수(2026-10-18, 2단계: 실측 데이터 연결)
# 규칙: DB는 db_manager로만 접속하고 연결마다 read_only로 연 뒤 rollback한다(쓰기 없음).
# 캐시는 st.cache_data(ttl=300)와 수동 새로고침만 쓴다(자동 새로고침 금지 — Neon 컴퓨트가 계속 깨어난다).
# NUMERIC 컬럼은 SQL에서 float8로 바꿔 읽는다(Decimal 섞임 방지).

import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from constants import STOCKS  # noqa: E402
from db_manager import get_db_connection  # noqa: E402

KST = timezone(timedelta(hours=9))
TTL = 300
WEEKDAYS = "월화수목금토일"


def query(sql, params=()):
    """
    설명: 읽기 전용 연결로 SQL 하나를 실행해 DataFrame으로 돌려준다.
    주의 사항: 연결을 read_only로 바꾼 뒤 실행하고, 끝나면 rollback·close한다. 실패하면 예외를 그대로 던진다.
    입력 타입->반환 타입: (str, tuple) -> pd.DataFrame
    """
    conn = get_db_connection()
    try:
        conn.read_only = True
        cur = conn.execute(sql, params)
        df = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
        conn.rollback()
        return df
    finally:
        conn.close()


def fmt_day(d):
    """
    설명: 날짜를 '10/16(금)' 형식으로 만든다.
    주의 사항: None·NaT면 '-'.
    입력 타입->반환 타입: (date|datetime|pd.Timestamp|None) -> str
    """
    if d is None or pd.isna(d):
        return "-"
    d = pd.Timestamp(d)
    return f"{d.month}/{d.day}({WEEKDAYS[d.weekday()]})"


def version_kst(version):
    """
    설명: 체크포인트 버전(UTC 저장 시각 YYYYMMDD_HHMMSS)을 KST datetime으로 바꾼다.
    주의 사항: 형식이 다르면 None.
    입력 타입->반환 타입: (str) -> datetime|None
    """
    try:
        return datetime.strptime(version, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc).astimezone(KST)
    except (TypeError, ValueError):
        return None


STOCK_NAMES = {s.ticker: s.name for s in STOCKS if s.active}

# ── 패널 B: 오늘의 예측 ──────────────────────────────────────────────────────────

# 평소 수준 = 종목의 prediction_date까지 최근 60거래일 실현 변동성 평균.
# 실현 변동성은 model_predictions_realized 뷰와 같은 정의: |ln(close_t / 직전 행 close)| x 100, close <= 0이면 NULL.
USUAL_WINDOW = 60
USUAL_SQL = """
WITH r AS (
    SELECT ticker, date, close,
           LAG(close) OVER (PARTITION BY ticker ORDER BY date) AS prev_close,
           row_number() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
    FROM daily_stock_prices
    WHERE ticker = ANY(%s) AND date <= %s AND date > %s::date - 200)
SELECT ticker,
       avg(CASE WHEN prev_close > 0 AND close > 0 THEN abs(ln(close::float8 / prev_close::float8)) * 100 END) AS usual_vol,
       count(*) AS n_days
FROM r WHERE rn <= %s AND prev_close IS NOT NULL
GROUP BY ticker
"""


@st.cache_data(ttl=TTL, show_spinner=False)
def latest_predictions():
    """
    설명: 가장 최근 prediction_date의 예측 행에 종목명과 평소 대비 배율을 붙여 돌려준다.
    주의 사항: DB 읽기. 평소 대비 = 예측값 ÷ 최근 60거래일 실현 변동성 평균. 예측 행이 없으면 빈 DataFrame.
    입력 타입->반환 타입: () -> pd.DataFrame
    """
    df = query("""
        SELECT ticker, prediction_date, target_date, predicted_volatility::float8 AS predicted_volatility,
               garch_baseline::float8 AS garch_baseline, gate_passed, input_data_suspect, model_version
        FROM model_predictions
        WHERE prediction_date = (SELECT max(prediction_date) FROM model_predictions)""")
    if df.empty:
        return df
    pdate = df["prediction_date"].iloc[0]
    usual = query(USUAL_SQL, (df["ticker"].tolist(), pdate, pdate, USUAL_WINDOW))
    df = df.merge(usual, on="ticker", how="left")
    df["ratio"] = df["predicted_volatility"] / df["usual_vol"]
    df["name"] = df["ticker"].map(STOCK_NAMES)
    df = df.sort_values("predicted_volatility", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df
