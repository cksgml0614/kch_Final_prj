# 대시보드/데이터.py — 운영 대시보드의 DB·파일 조회 함수(패널 B 최신 예측, 패널 C 운영 기록·백테스트 재생)와 RMSE 집계.
# 규칙: DB는 db_manager로만 접속하고 연결마다 read_only로 연 뒤 rollback한다(쓰기 없음). NUMERIC은 SQL에서 float8로 읽는다.
# 캐시는 st.cache_data(ttl=300)와 수동 새로고침만 쓴다(자동 새로고침 금지 — Neon 컴퓨트가 계속 깨어난다).
# 배경: 결정기록.md의 "읽기 전용·수동 새로고침", "사전 예측 판정은 평가일 08:30"

import json
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

# ── 패널 C: 성능 ──────────────────────────────────────────────────────────────────

# 그룹 이름(화면 라벨). model_registry.source='live' 행만 운영으로 보고, 'test'와 backtest_d4_20260914는 이 화면에서 뺀다.
GROUP_PRE, GROUP_POST, GROUP_REPLAY = "운영(사전 예측)", "운영(사후 계산)", "백테스트 재생"
PREDICTION_COLS = {"predicted_volatility": "하이브리드", "garch_baseline": "GARCH",
                   "sma20_baseline": "SMA20", "parkinson_sma20_baseline": "Parkinson"}
MIN_ROWS_PER_DAY = 90  # 이보다 행이 적은 평가일은 일별 값에서 뺀다("불완전")
PRE_CUTOFF_HOUR, PRE_CUTOFF_MINUTE = 8, 30  # model_version(KST)이 평가일 08:30보다 이르면 사전 예측
REPLAY_DIR = os.environ.get("BACKTEST_REPLAY_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
REPLAY_PARQUET = os.path.join(REPLAY_DIR, "backtest_replay.parquet")
REPLAY_META = os.path.join(REPLAY_DIR, "backtest_replay.meta.json")


def is_pre_prediction(version, realized_date):
    """
    설명: model_version 시각(KST)이 평가일(realized_date) 08:30 KST보다 이르면 True(사전 예측).
    주의 사항: model_version은 체크포인트 저장 시각이라 실제 저장보다 최대 약 11분 이르다(그래서 장 시작 09:00 대신 08:30).
    입력 타입->반환 타입: (str, date) -> bool|None
    """
    v = version_kst(version)
    if v is None or realized_date is None or pd.isna(realized_date):
        return None
    cut = datetime(realized_date.year, realized_date.month, realized_date.day, PRE_CUTOFF_HOUR, PRE_CUTOFF_MINUTE, tzinfo=KST)
    return v < cut


@st.cache_data(ttl=TTL, show_spinner=False)
def operational_records():
    """
    설명: 운영(live) 예측 행과 실측(model_predictions_realized 뷰)을 읽어 사전/사후 그룹을 붙여 돌려준다.
    주의 사항: DB 읽기. 실측 대기 행은 group이 '실측 대기'이고 계산에서 뺀다.
    입력 타입->반환 타입: () -> pd.DataFrame
    """
    df = query("""
        SELECT p.ticker, p.prediction_date, p.target_date, p.realized_date, p.realized_status,
               p.realized_volatility, p.target_halted,
               p.predicted_volatility::float8 AS predicted_volatility, p.garch_baseline::float8 AS garch_baseline,
               p.sma20_baseline::float8 AS sma20_baseline, p.parkinson_sma20_baseline::float8 AS parkinson_sma20_baseline,
               p.gate_passed, p.input_data_suspect, p.model_version
        FROM model_predictions_realized p JOIN model_registry r USING (model_version)
        WHERE r.source = 'live'""")
    pre = [is_pre_prediction(v, d) if s == "confirmed" else None
           for v, d, s in zip(df["model_version"], df["realized_date"], df["realized_status"])]
    df["group"] = [GROUP_PRE if x else GROUP_POST if x is False else "실측 대기" for x in pre]
    return df


@st.cache_data(ttl=TTL, show_spinner=False)
def replay_records():
    """
    설명: 백테스트 재생 파일(대시보드/cache/)을 읽어 (DataFrame, meta dict)로 돌려준다. 파일이 없으면 (None, None).
    주의 사항: 파일 읽기만. 재생은 대시보드/백테스트_재생.py를 사람이 실행해 만든다.
    입력 타입->반환 타입: () -> (pd.DataFrame|None, dict|None)
    """
    if not (os.path.exists(REPLAY_PARQUET) and os.path.exists(REPLAY_META)):
        return None, None
    df = pd.read_parquet(REPLAY_PARQUET)
    with open(REPLAY_META, encoding="utf-8") as f:
        meta = json.load(f)
    df["group"] = [GROUP_REPLAY if s == "confirmed" else "실측 대기" for s in df["realized_status"]]
    return df, meta


def tickermean_rmse(df, col):
    """
    설명: 종목별 RMSE(예측 col vs realized_volatility)를 구해 종목 평균을 돌려준다(배포 게이트와 같은 집계).
    주의 사항: 실측이 없는 행은 뺀다. 하루치(종목당 1행)면 종목별 RMSE = |오차|라 값은 평균 절대 오차와 같다.
    입력 타입->반환 타입: (pd.DataFrame, str) -> float
    """
    d = df.dropna(subset=["realized_volatility", col])
    if d.empty:
        return float("nan")
    err2 = (d[col] - d["realized_volatility"]) ** 2
    return float(err2.groupby(d["ticker"]).mean().pow(0.5).mean())


def rmse_row(df):
    """
    설명: 예측 4계열의 종목평균 RMSE를 {라벨: 값}으로 돌려준다.
    주의 사항: 없음.
    입력 타입->반환 타입: (pd.DataFrame) -> dict
    """
    return {label: tickermean_rmse(df, col) for col, label in PREDICTION_COLS.items()}


BASELINE_LABELS = ("GARCH", "SMA20", "Parkinson")


def daily_compare(g):
    """
    설명: 평가일 하루치(g)의 비교 값을 dict로 돌려준다 — 계열별 평균 절대 오차(라벨 키), 실현 평균, 예측 평균, 평균 오차
      (예측 - 실현, +면 과대), 최선 baseline 라벨과 값, 하이브리드 ÷ 최선 baseline(1 미만 = 하이브리드가 셋 다 이김).
    주의 사항: 실측이 없는 행은 뺀다. 하루치는 종목당 1행이라 평균 절대 오차 = 종목평균 RMSE(tickermean_rmse)와 같다.
      절대 오차는 그날 시장이 크게 움직일수록 커지므로(실현 평균과 같이 움직인다) 날짜끼리는 비율로 비교한다.
    입력 타입->반환 타입: (pd.DataFrame) -> dict
    """
    d = g.dropna(subset=["realized_volatility"])
    out = {label: float((d[col] - d["realized_volatility"]).abs().mean()) if d[col].notna().any() else float("nan")
           for col, label in PREDICTION_COLS.items()}
    base = {k: out[k] for k in BASELINE_LABELS if pd.notna(out[k])}
    best = min(base, key=base.get) if base else None
    out.update({"실현 평균": float(d["realized_volatility"].mean()), "예측 평균": float(d["predicted_volatility"].mean()),
                "평균 오차": float((d["predicted_volatility"] - d["realized_volatility"]).mean()),
                "최선 baseline": best, "최선 값": base.get(best, float("nan")),
                "비율": out["하이브리드"] / base[best] if best and base[best] else float("nan")})
    return out
