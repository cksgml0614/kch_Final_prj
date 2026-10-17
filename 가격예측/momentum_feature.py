# momentum_feature.py — 초과수익률 z의 하루 지연값(excess_return_z_lag1) 피처를 만든다(파일 이름은 옛 이름).
# 파이프라인 단계: 피처. 주요 진입 함수: build_momentum_feature, load_true_kospi.
# 배경: 결정기록.md의 "초과수익률 z는 당일 KOSPI를 date 기준으로 직접 조회해 계산"

import sys

import pandas as pd

from db_manager import get_db_connection
from 시장지표.feature_loader import get_features

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WINDOW = 60


def load_true_kospi(start_date, end_date):
    """
    설명: market_indicators의 KOSPI를 date(기준일) 기준으로 직접 읽어 날짜 인덱스 Series로 돌려준다.
    주의 사항: DB 읽기. feature_loader의 published_date 지연 조인을 거치지 않은 당일 값이다.
    입력 타입->반환 타입: (str, str) -> pd.Series
    """
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패")
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT date, value FROM market_indicators
                WHERE indicator_code = 'KOSPI' AND date BETWEEN %s AND %s
                ORDER BY date
                """,
                (start_date, end_date),
            )
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["date", "value"])
    df["value"] = df["value"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["value"]


def _excess_return_z(stock_cr, kospi_cr, window=WINDOW):
    """
    설명: 초과수익률, 직전 window일 표준편차(당일 제외), z를 계산한다.
    주의 사항: 앞쪽 window일은 sigma가 NaN이라 z도 NaN이다.
    입력 타입->반환 타입: (pd.Series, pd.Series, int) -> (pd.Series, pd.Series, pd.Series)
    """
    excess_return = stock_cr - kospi_cr
    # shift(1).rolling(window)는 위치 i에서 [i-window, ..., i-1]을 보므로 당일 값이 sigma에서 빠진다.
    sigma = excess_return.shift(1).rolling(window, min_periods=window).std()
    z = excess_return / sigma
    return excess_return, sigma, z


def build_momentum_feature(ticker, start_date, end_date, window=WINDOW, raw=None, true_kospi=None):
    """
    설명: target 행 t에 넣을 excess_return_z_(t-1) 피처와 중간 산출물 dict(diag)를 돌려준다.
    주의 사항: raw·true_kospi를 넘기면 다시 읽지 않으므로 같은 (ticker, start_date, end_date)로 만든 값이어야 한다.
    입력 타입->반환 타입: (str, str, str, int, pd.DataFrame|None, pd.Series|None) -> (pd.Series, dict)
    """
    if raw is None:
        raw = get_features(ticker, start_date, end_date)
    stock_cr = raw["close"].pct_change()

    if true_kospi is None:
        true_kospi = load_true_kospi(start_date, end_date)
    kospi_aligned = true_kospi.reindex(raw.index)
    n_missing = int(kospi_aligned.isna().sum())
    kospi_cr = kospi_aligned.pct_change()

    excess_return, sigma, z = _excess_return_z(stock_cr, kospi_cr, window)
    momentum = z.shift(1)
    momentum.name = "excess_return_z_lag1"

    diag = {
        "raw": raw,
        "true_kospi": true_kospi,
        "n_missing_kospi": n_missing,
        "excess_return": excess_return,
        "sigma": sigma,
        "z": z,
    }
    return momentum, diag
