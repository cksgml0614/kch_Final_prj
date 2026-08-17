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


def build_momentum_feature(ticker, start_date, end_date, window=WINDOW):
    """target 행 t에 배치할 excess_return_z_(t-1) 피처를 계산한다.

    Returns
    -------
    momentum : Series, index=target date t, value=z_(t-1)
    diag : dict — 검증/비교용 중간 산출물 (raw, true_kospi, excess_return, sigma, z,
           wrong_momentum(잘못된 방식 비교용))
    """
    raw = get_features(ticker, start_date, end_date)
    stock_cr = raw["close"].pct_change()

    # --- 올바른 방식: market_indicators를 date 기준 직접 조회 ---
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


def verify_momentum_alignment(raw, excess_return, sigma, momentum, window=WINDOW, n=10, seed=42):
    """momentum[t] = z_(t-1)이 실제로
      (a) t-1, sigma window 전부 t보다 이전인지 (date_ok)
      (b) sigma가 excess_return[t-1-window .. t-2] 슬라이스로 독립 재계산한 값과 일치하는지 (value_ok)
      (c) z 정의식(excess_return/sigma) 재계산값과 일치하는지 (target_ok)
    를 rolling/shift 체인을 신뢰하지 않고 직접 대조한다."""
    dates = list(raw.index)
    pos = {d: i for i, d in enumerate(dates)}

    valid = momentum.dropna()
    sample = pd.Series(valid.index).sample(min(n, len(valid)), random_state=seed).sort_values()

    print(f"\n=== momentum 피처 검증: target t 대비 sigma가 t-2까지만 보는지, 샘플 {len(sample)}건 ===")
    all_ok = True
    for t in sample:
        i = pos[t]
        d_prev = dates[i - 1]  # excess_return_(t-1)의 실제 날짜
        sigma_window_dates = dates[i - 1 - window : i - 1]  # [t-1-window, ..., t-2], 60개

        date_ok = (d_prev < t) and all(d < d_prev for d in sigma_window_dates)

        manual_sigma = excess_return.loc[sigma_window_dates].std()
        stored_sigma = sigma.loc[d_prev]
        value_ok = abs(manual_sigma - stored_sigma) < 1e-9

        manual_z = excess_return.loc[d_prev] / manual_sigma
        target_ok = abs(manual_z - momentum.loc[t]) < 1e-9

        ok = date_ok and value_ok and target_ok
        all_ok &= ok
        mark = "OK" if ok else "FAIL"
        print(
            f"  [{mark}] target일={t.date()}  excess_return일={d_prev.date()}  "
            f"sigma window=[{sigma_window_dates[0].date()}..{sigma_window_dates[-1].date()}]({len(sigma_window_dates)}일)  "
            f"date_ok={date_ok} value_ok={value_ok} target_ok={target_ok}  "
            f"z={momentum.loc[t]:+.4f}"
        )

    print(f"전체 통과 (t/t-1 미포함 AND sigma 재계산 일치 AND z 정의식 일치): {all_ok}")
    return all_ok


def compare_correct_vs_wrong(diag, n=10, seed=42):
    """올바른 방식(당일 KOSPI 직접 조회)과 잘못된 방식(feature_loader의 지연된 KOSPI)의
    excess_return/모멘텀 피처가 실제로 다른지 정량 비교한다."""
    excess_return = diag["excess_return"]
    wrong_excess_return = diag["wrong_excess_return"]

    diff = (excess_return - wrong_excess_return).dropna()
    n_diff = int((diff.abs() > 1e-9).sum())

    print("\n=== 올바른 방식 vs 잘못된 방식(raw['KOSPI'] 그대로 사용) 비교 ===")
    print(f"excess_return 비교 대상 {len(diff)}건 중 값이 다른 건수: {n_diff} ({n_diff/len(diff)*100:.1f}%)")
    print(f"차이(diff = 올바른값 - 잘못된값) 통계: mean={diff.mean():+.6f}  std={diff.std():.6f}  "
          f"max_abs={diff.abs().max():.6f}")

    sample = diff.sample(min(n, len(diff)), random_state=seed).sort_index()
    print(f"\n샘플 {len(sample)}건 (excess_return 기준):")
    for d in sample.index:
        print(
            f"  {d.date()}  올바른={excess_return.loc[d]:+.4%}  "
            f"잘못됨={wrong_excess_return.loc[d]:+.4%}  차이={diff.loc[d]:+.4%}"
        )

    momentum = diag.get("momentum")
    return {"n_compared": len(diff), "n_diff": n_diff, "diff_stats": diff.describe()}


if __name__ == "__main__":
    TICKER = "005930"
    START = "2020-01-02"
    END = "2026-02-01"

    momentum, diag = build_momentum_feature(TICKER, START, END)

    print(f"=== build_momentum_feature({TICKER}, {START}, {END}) ===")
    print(f"momentum shape: {momentum.shape}, NaN 제외 후 유효 행 수: {momentum.notna().sum()}")
    print(f"true_kospi 조회 건수: {len(diag['true_kospi'])}, raw 인덱스 대비 결측: {diag['n_missing_kospi']}")

    ok = verify_momentum_alignment(diag["raw"], diag["excess_return"], diag["sigma"], momentum)

    diag["momentum"] = momentum
    compare_correct_vs_wrong(diag)

    print("\n=== momentum 앞 5개 유효값 ===")
    print(momentum.dropna().head(5))
    print("\n=== momentum 뒤 5개 유효값 ===")
    print(momentum.dropna().tail(5))
