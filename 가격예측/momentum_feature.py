# momentum_feature.py — Task T-1 체크포인트 2
# lag 적용 초과수익률(excess_return_z) 모멘텀 피처.
#
# 세션에서 승인된 두 가지 해석:
#   1) sigma_(t-1)의 window는 [t-61, ..., t-2] — t-1 자기 자신도 제외 (CLAUDE.md D-2의
#      "sigma_t는 t 시점 미포함" 원칙을 t→t-1로 재적용).
#   2) KOSPI는 feature_loader의 발표일 지연 조인(published_date < 거래일) 결과를 쓰지 않는다.
#      market_index_load.py에서 KOSPI의 published_date = date(당일 발표)이므로,
#      feature_loader가 반환하는 raw['KOSPI']는 이미 "d-1일자 값"이 d행에 들어가 있다.
#      excess_return은 "같은 날 종목 대비 시장" 비교여야 하므로, market_indicators를
#      date 기준으로 직접 재조회해 당일 실제 값으로 계산한다.

import sys

import pandas as pd

from db_manager import get_db_connection
from 시장지표.feature_loader import get_features

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WINDOW = 60


def load_true_kospi(start_date, end_date):
    """market_indicators를 date 기준으로 직접 조회한다 (feature_loader의 published_date
    지연 조인을 우회 — 당일 실제 KOSPI 값이 필요하므로)."""
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
    """excess_return_d, sigma_d(d 미포함 trailing), z_d를 계산한다."""
    excess_return = stock_cr - kospi_cr
    # shift(1).rolling(window)는 위치 i에서 [i-window, ..., i-1] 구간을 보므로
    # sigma_d 자신(d)은 window에서 자동으로 빠진다.
    sigma = excess_return.shift(1).rolling(window, min_periods=window).std()
    z = excess_return / sigma
    return excess_return, sigma, z


def build_momentum_feature(ticker, start_date, end_date, window=WINDOW, raw=None, true_kospi=None):
    """target 행 t에 배치할 excess_return_z_(t-1) 피처를 계산한다.

    raw(선택, 2026-09-20 추가): 호출부가 이미 계산해둔 get_features(ticker, start_date,
    end_date) 결과를 넘기면 재계산하지 않는다 — 같은 (ticker, start_date, end_date)로 계산된
    것이어야 한다(호출부 책임). None이면 기존과 동일하게 내부에서 계산한다.
    true_kospi(선택, 2026-09-20 추가): load_true_kospi(start_date, end_date) 결과를 넘기면
    재계산하지 않는다 — 종목과 무관한 값이라 100종목이 그대로 공유할 수 있다(단, start_date/
    end_date는 이 호출과 같아야 한다). None이면 기존과 동일하게 내부에서 계산한다.
    이 두 인자를 넘기지 않으면(둘 다 None) 이 함수의 동작은 리팩터링 이전과 완전히 동일하다.

    Returns
    -------
    momentum : Series, index=target date t, value=z_(t-1)
    diag : dict — 검증/비교용 중간 산출물 (raw, true_kospi, excess_return, sigma, z,
           wrong_momentum(잘못된 방식 비교용))
    """
    if raw is None:
        raw = get_features(ticker, start_date, end_date)
    stock_cr = raw["close"].pct_change()

    # --- 올바른 방식: market_indicators를 date 기준 직접 조회 ---
    if true_kospi is None:
        true_kospi = load_true_kospi(start_date, end_date)
    kospi_aligned = true_kospi.reindex(raw.index)
    n_missing = int(kospi_aligned.isna().sum())
    kospi_cr = kospi_aligned.pct_change()

    excess_return, sigma, z = _excess_return_z(stock_cr, kospi_cr, window)
    momentum = z.shift(1)
    momentum.name = "excess_return_z_lag1"

    # --- 잘못된 방식(비교용): feature_loader가 반환하는, 이미 발표일 지연이 적용된 KOSPI를
    #     그대로 사용 — d행의 값이 실제로는 d-1일자 KOSPI라 하루 밀린 채로 초과수익률이 계산됨 ---
    wrong_kospi_cr = raw["KOSPI"].pct_change()
    wrong_excess_return, wrong_sigma, wrong_z = _excess_return_z(stock_cr, wrong_kospi_cr, window)
    wrong_momentum = wrong_z.shift(1)
    wrong_momentum.name = "excess_return_z_lag1_WRONG"

    diag = {
        "raw": raw,
        "true_kospi": true_kospi,
        "n_missing_kospi": n_missing,
        "excess_return": excess_return,
        "sigma": sigma,
        "z": z,
        "wrong_excess_return": wrong_excess_return,
        "wrong_momentum": wrong_momentum,
    }
    return momentum, diag
