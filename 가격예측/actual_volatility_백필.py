# actual_volatility_백필.py
# model_predictions.actual_volatility(지금까지 한 번도 채워진 적 없음, NULL)를
# daily_stock_prices 실측 종가로 채우는 백필 스크립트(2026-09-14). 목적은 과거 재현이
# 아니라 "이미 지나간 예측이 실제로 얼마나 정확했는지" 검증해 앞으로의 예측 신뢰도를
# 판단하는 것 — 특히 A-1(배포 게이트 강제) 적용 이전에 저장된 기존 100행(게이트 미통과
# 상태였음에도 저장됐던 것)의 실제 정확도가 게이트 강제가 옳은 결정이었는지의 첫 실증
# 데이터가 된다.
#
# 공식(가격예측/garch_baseline.compute_log_returns_pct, 가격예측/dataset_builder.
# build_base_dataset_v2_volatility의 target 정의와 완전히 동일 — 대조 확인 완료):
#   actual_volatility = |log(close_t / close_(t-1))| * 100
# close_(t-1)은 daily_stock_prices에서 target_date 이전 "가장 최근에 실제로 존재하는
# 거래일"의 종가다(단순 target_date - 1일 아님 — 주말/공휴일에 깨짐).
#
# 3-파일 도메인 패턴(공통/초기적재/일일수집)과 구분되는 일회성/주기성 유틸리티라 별도
# 명명(actual_volatility_백필.py / actual_volatility_비교.py, 라벨_생성.py/라벨_보고.py
# 선례와 동일한 "쓰기/읽기 분리" 원칙).
#
# 이 스크립트는 daily_stock_prices를 읽어 model_predictions를 UPDATE만 한다(UPSERT 아님
# — 새 행 생성 없음). 학습/예측 코드나 GARCH 캐시는 전혀 건드리지 않는다. target_date
# 자체가 daily_stock_prices에 없으면(아직 그 날 데이터가 안 들어왔거나 실제 거래일이
# 아니었던 경우 — next_weekday() 근사의 알려진 한계) actual_volatility를 NULL로 남기고
# 스킵한다 — 조용히 틀린 값을 채우지 않는다는, 이 프로젝트 스키마 주석에 이미 명시된
# 안전장치 원칙과 동일하다.
#
# 재실행 안전(멱등): actual_volatility IS NULL인 행만 대상으로 삼으므로, 이미 채워진
# 행은 건드리지 않는다. 여러 번 실행해도 결과가 같다 — 향후 daily_pipeline.bat 등
# 일일 자동화에 추가할 후보(예: 매일 실행분 끝에 "어제 예측 확정" 단계로 편입).

import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from db_manager import get_db_connection

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SAMPLE_LOG_COUNT = 3  # 검산용 샘플 몇 건을 로그에 남길지


def fetch_backfill_targets():
    """actual_volatility가 아직 안 채워진 행 중, target_date가 미래(내일 이후)가 아닌
    행만 가져온다.

    ⚠️ 2026-09-14 구현 중 발견: 원래는 `target_date < CURRENT_DATE`(오늘 미포함)로 짜려
    했으나, 이 스크립트를 실행하는 "오늘"이 마침 기존 100행의 target_date(2026-09-14)와
    정확히 같은 날이라 그 필터로는 0건이 걸린다 — 장은 이미 마감했고 daily_stock_prices에
    그날 종가가 이미 들어와 있는데도(이번 세션에서 이미 확인) 단순 날짜 비교만으로
    "오늘=아직 미실현"이라고 단정하면 실제로 계산 가능한 값을 부당하게 걸러내는 것이다.
    실제 "실현 여부"를 판단하는 진짜 안전장치는 날짜 비교가 아니라 아래
    compute_actual_volatility()의 존재 여부 체크(target_date 종가가 daily_stock_prices에
    있는지)다 — 그 체크가 이미 "아직 데이터가 없으면 NULL로 스킵"을 보장하므로, 여기서는
    `<= CURRENT_DATE`로 완화해 명백히 미래인 날짜(내일 이후)만 걸러내는 용도로만 쓴다."""
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패 — fetch_backfill_targets")
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ticker, target_date
                FROM model_predictions
                WHERE actual_volatility IS NULL AND target_date <= CURRENT_DATE
                ORDER BY ticker, target_date
                """
            )
            return cur.fetchall()


def fetch_close_prices(ticker, end_date):
    """target_date 이하의 종가 전체 이력을 가져온다(하한 없음 — "가장 최근 이전 거래일"
    탐색이 얼마나 과거로 가야 할지 미리 알 수 없으므로 안전하게 전체를 가져온다)."""
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패 — fetch_close_prices")
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT date, close FROM daily_stock_prices
                WHERE ticker = %s AND date <= %s
                ORDER BY date
                """,
                (ticker, end_date),
            )
            rows = cur.fetchall()
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows, columns=["date", "close"])
    df["close"] = df["close"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"]


def compute_actual_volatility(close_series, target_ts):
    """garch_baseline.compute_log_returns_pct / dataset_builder.
    build_base_dataset_v2_volatility와 완전히 동일한 정의로 계산한다:
    |log(close_t / close_(t-1))| * 100. close_(t-1)은 close_series에서 target_ts
    이전 가장 최근 값(연속된 두 행 — 거래일 gap-safe, 단순 -1일 아님).

    반환: (value 또는 None, 스킵 사유 또는 None)"""
    if target_ts not in close_series.index:
        return None, "target_date 종가 없음(아직 미수집이거나 실제 거래일 아님)"
    prior = close_series[close_series.index < target_ts]
    if prior.empty:
        return None, "직전 거래일 데이터 없음(이력 시작 지점)"
    close_t = close_series.loc[target_ts]
    close_prev = prior.iloc[-1]
    log_ret_pct = np.log(close_t / close_prev) * 100
    return abs(float(log_ret_pct)), None


def update_actual_volatility(ticker, target_date, value):
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패 — update_actual_volatility")
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE model_predictions SET actual_volatility = %s WHERE ticker = %s AND target_date = %s",
                (value, ticker, target_date),
            )
        conn.commit()


def main():
    targets = fetch_backfill_targets()
    print(f"백필 대상: {len(targets)}행 (actual_volatility IS NULL AND target_date <= 오늘 — "
          f"실제 '실현 여부'는 종가 존재 체크로 판단, 이 필터는 명백한 미래 날짜만 예비 제외)")
    if not targets:
        print("백필할 행이 없습니다.")
        return

    by_ticker = defaultdict(list)
    for ticker, target_date in targets:
        by_ticker[ticker].append(target_date)

    filled = 0
    skip_reasons = defaultdict(int)
    sample_logged = 0

    for ticker, dates in by_ticker.items():
        max_date = max(dates)
        close_series = fetch_close_prices(ticker, max_date)
        for target_date in dates:
            target_ts = pd.Timestamp(target_date)
            value, reason = compute_actual_volatility(close_series, target_ts)
            if value is None:
                print(f"⏭️  {ticker} {target_date}: 스킵 — {reason}")
                skip_reasons[reason] += 1
                continue

            update_actual_volatility(ticker, target_date, value)
            filled += 1

            if sample_logged < SAMPLE_LOG_COUNT:
                prior = close_series[close_series.index < target_ts]
                close_prev_date = prior.index[-1].date()
                close_prev = prior.iloc[-1]
                close_t = close_series.loc[target_ts]
                print(
                    f"🔍 검산 샘플 [{ticker} {target_date}]: "
                    f"close_t({target_date})={close_t}, close_prev({close_prev_date})={close_prev}, "
                    f"|log({close_t}/{close_prev})| x 100 = {value:.6f}"
                )
                sample_logged += 1

    print("\n=== 백필 요약 ===")
    print(f"총 대상: {len(targets)}행")
    print(f"채움: {filled}행")
    print(f"스킵: {len(targets) - filled}행")
    if skip_reasons:
        print("스킵 사유별 집계:")
        for reason, count in skip_reasons.items():
            print(f"  - {reason}: {count}건")


if __name__ == "__main__":
    main()
