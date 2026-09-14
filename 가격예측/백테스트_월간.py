# 백테스트_월간.py
# 약 1개월치 워크포워드 백테스트(2026-09-14, D-4) — val 기준 게이트 판정(GARCH는 5연속
# 이김)과 실측 단일일 결과(GARCH한테도 짐) 사이의 불일치 원인을 진단하기 위한 표본 확보용
# 일회성 진단 도구. 기존 3파일 도메인 패턴(공통/초기적재/일일수집)과 구분되는 명명 —
# actual_volatility_백필.py/actual_volatility_비교.py와 같은 계열.
#
# 과거 여러 날짜(--end에 해당하는 D)를 순서대로 run_pooled_volatility_pipeline()에 직접
# 넘겨(subprocess 반복보다 효율적) 각 날짜의 예측을 model_predictions에 쌓고,
# actual_volatility_백필.py로 실측을 채운 뒤 일자별 breakdown까지 집계한다.
#
# ⚠️ GARCH 캐시 시점 무결성(CLAUDE.md "알려진 이슈" 절 참고, 2026-09-12에 이미 확인된
# 문제): get_or_fit_garch_params()의 캐시 재사용 판단은 fitted_at 기준 경과일수만 보고
# train_end 일치 여부는 보지 않는다. 이 루프 전체가 실행되는 하루(오늘) 안에서는 fitted_at이
# 계속 "오늘"로 갱신되므로, 캐시를 비우지 않으면 날짜마다 새로 적합해야 할 파라미터가
# 직전 백테스트 날짜의 파라미터로 조용히 재사용된다 — 워크포워드 무결성을 근본적으로
# 깨뜨리므로 매 날짜(D)마다 캐시 디렉터리를 통째로 비우고 시작한다(절대 생략 불가).
#
# ⚠️ target_date 안전장치: next_weekday(D) >= PRODUCTION_CUTOFF(2026-09-14)가 되는 D는
# 자동으로 제외한다 — 기존 프로덕션 행(09-14 100행, 09-15 2행)과 절대 겹치면 안 된다.
#
# MLflow 분리: run_pooled_volatility_pipeline()이 내부적으로 참조하는
# 가격예측_변동성_공통.MLFLOW_EXPERIMENT_NAME 모듈 전역 상수를 이 스크립트 실행 중에만
# "백테스트"로 몽키패치한다(일일 자동화 코드 경로 자체는 전혀 건드리지 않음 — try/finally로
# 루프가 끝나거나 예외가 나도 원래 값("일일_자동화")으로 반드시 복원).
#
# 학습 로직/게이트 기준/A-2 sanity check는 전혀 건드리지 않는다 — run_pooled_volatility_
# pipeline()을 그대로 호출만 한다.

import glob
import os
import sys
import time
from datetime import date

import pandas as pd

from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
from db_manager import get_db_connection
import 가격예측.가격예측_변동성_공통 as pipeline_module
from 가격예측.actual_volatility_비교 import load_comparison_rows

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BACKTEST_EXPERIMENT_NAME = "백테스트"
PRODUCTION_CUTOFF = date(2026, 9, 14)  # 이 날짜 이상 target_date는 절대 생성 금지
GARCH_CACHE_DIR = pipeline_module.GARCH_PARAM_CACHE_DIR


def next_weekday(d):
    """가격예측_변동성_공통.next_weekday를 그대로 재사용(로직 복제 금지, 단일 출처 유지)."""
    return pipeline_module.next_weekday(d)


def get_backtest_dates(start_date, end_date):
    """daily_stock_prices의 실제 거래일 중, next_weekday(D)가 PRODUCTION_CUTOFF보다
    이른 날짜만 반환한다 — 기존 프로덕션 행과 절대 안 겹치도록 하는 안전장치."""
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패 — get_backtest_dates")
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT date FROM daily_stock_prices WHERE date BETWEEN %s AND %s ORDER BY date",
                (start_date, end_date),
            )
            all_dates = [r[0] for r in cur.fetchall()]

    safe_dates = [d for d in all_dates if next_weekday(d) < PRODUCTION_CUTOFF]
    excluded = [d for d in all_dates if d not in safe_dates]
    if excluded:
        print(f"⚠️ target_date가 {PRODUCTION_CUTOFF} 이상이 되어 제외된 날짜: {excluded}")
    return safe_dates


def clear_garch_cache():
    removed = glob.glob(os.path.join(GARCH_CACHE_DIR, "*.json"))
    for f in removed:
        os.remove(f)
    return len(removed)


def run_backtest_loop(start_date_range, end_date_range):
    """각 거래일 D에 대해: GARCH 캐시 전체 삭제 -> run_pooled_volatility_pipeline(D) 직접
    호출 -> 진행 상황 출력. MLflow는 이 함수 실행 중에만 "백테스트" experiment로 몽키패치."""
    dates = get_backtest_dates(start_date_range, end_date_range)
    print(f"백테스트 대상 거래일: {len(dates)}개 ({dates[0]} ~ {dates[-1]})")

    original_experiment = pipeline_module.MLFLOW_EXPERIMENT_NAME
    pipeline_module.MLFLOW_EXPERIMENT_NAME = BACKTEST_EXPERIMENT_NAME
    try:
        for i, d in enumerate(dates, start=1):
            n_removed = clear_garch_cache()
            t0 = time.time()
            result = pipeline_module.run_pooled_volatility_pipeline(
                ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START, d.isoformat()
            )
            elapsed = time.time() - t0
            gate = result["gate"]
            preds = result["predictions"]
            n_failed = sum(1 for v in preds.values() if isinstance(v, str) and v.startswith("실패"))
            n_success = len(preds) - n_failed
            print(f"[{i}/{len(dates)}] {d} 완료({elapsed:.0f}초, GARCH 캐시 {n_removed}개 비움) — "
                  f"gate_passed={gate['passed']}, 성공={n_success}, 실패={n_failed}")
    finally:
        pipeline_module.MLFLOW_EXPERIMENT_NAME = original_experiment
        print(f"MLflow experiment를 '{BACKTEST_EXPERIMENT_NAME}'에서 '{original_experiment}'로 복원했습니다.")

    return dates


def backfill_new_rows():
    """actual_volatility_백필.py를 그대로 호출 — actual_volatility IS NULL인 행만
    대상이라 기존 102행(이미 채워짐/아직 미실현)은 자연히 재대상 안 됨."""
    import 가격예측.actual_volatility_백필 as backfill_module
    backfill_module.main()


def compare_backtest_range(dates):
    """이번 백테스트로 생성된 target_date 범위만 필터링해 집계 — actual_volatility_
    비교.py의 load_comparison_rows()를 재사용하되(로직 복제 금지), 기존 프로덕션 100행
    (09-14)이 집계에 섞이지 않도록 범위를 좁힌다."""
    backtest_target_dates = sorted({next_weekday(d) for d in dates})

    df = load_comparison_rows()
    df = df[df["target_date"].isin(backtest_target_dates)].copy()

    if df.empty:
        print("백테스트 범위에 actual_volatility가 채워진 행이 없습니다 — 백필이 정상 실행됐는지 확인하세요.")
        return None, None

    for col in ["predicted_volatility", "actual_volatility", "garch_baseline", "sma20_baseline", "parkinson_sma20_baseline"]:
        df[col] = df[col].astype(float)
    df["err_hybrid"] = (df["predicted_volatility"] - df["actual_volatility"]).abs()
    df["err_garch"] = (df["garch_baseline"] - df["actual_volatility"]).abs()
    df["err_sma20"] = (df["sma20_baseline"] - df["actual_volatility"]).abs()
    df["err_parkinson"] = (df["parkinson_sma20_baseline"] - df["actual_volatility"]).abs()

    n = len(df)
    print(f"\n=== 백테스트 전체 집계 (N={n}, target_date {df['target_date'].min()} ~ {df['target_date'].max()}) ===")
    print(f"하이브리드 MAE      : {df['err_hybrid'].mean():.4f}")
    print(f"GARCH MAE           : {df['err_garch'].mean():.4f}")
    print(f"SMA20 MAE           : {df['err_sma20'].mean():.4f}")
    print(f"Parkinson-SMA20 MAE : {df['err_parkinson'].mean():.4f}")
    for name, col in [("GARCH", "err_garch"), ("SMA20", "err_sma20"), ("Parkinson-SMA20", "err_parkinson")]:
        wins = int((df["err_hybrid"] < df[col]).sum())
        print(f"vs {name:16s}: {wins}/{n} 승 ({wins / n * 100:.1f}%)")

    daily = df.groupby("target_date").agg(
        n=("err_hybrid", "size"),
        mae_hybrid=("err_hybrid", "mean"),
        mae_garch=("err_garch", "mean"),
        mae_sma20=("err_sma20", "mean"),
        mae_parkinson=("err_parkinson", "mean"),
    )
    win_rates = df.groupby("target_date").apply(
        lambda g: pd.Series({
            "win_vs_garch": (g["err_hybrid"] < g["err_garch"]).mean(),
            "win_vs_sma20": (g["err_hybrid"] < g["err_sma20"]).mean(),
            "win_vs_parkinson": (g["err_hybrid"] < g["err_parkinson"]).mean(),
        }),
        include_groups=False,
    )
    daily = daily.join(win_rates)

    print(f"\n=== 일자별 breakdown ({len(daily)}일) ===")
    pd.set_option("display.width", 200)
    print(daily.round(4).to_string())

    return df, daily


def additional_diagnostics(df):
    """종목별 집계(상/하위)와 분산 붕괴(과소추정 경향) 정량 확인(2026-09-15 세션 추가 —
    사람이 스크린샷으로 관찰한 "예측값이 baseline들보다 체계적으로 낮다"는 소견을 19일치
    실측 데이터로 정량 검증). df는 compare_backtest_range()가 반환하는 것과 동일 스키마."""
    print(f"\n=== 종목별 집계 (하이브리드 MAE 기준, N={df['ticker'].nunique()}종목) ===")
    per_ticker = df.groupby("ticker").agg(
        n=("err_hybrid", "size"),
        mae_hybrid=("err_hybrid", "mean"),
        mae_garch=("err_garch", "mean"),
        mae_sma20=("err_sma20", "mean"),
        mae_parkinson=("err_parkinson", "mean"),
    ).sort_values("mae_hybrid")
    pd.set_option("display.width", 200)
    print("가장 잘 맞춘 10종목:")
    print(per_ticker.head(10).round(4).to_string())
    print("\n가장 못 맞춘 10종목:")
    print(per_ticker.tail(10).round(4).to_string())

    print("\n=== 분산 붕괴(과소추정 경향) 정량 확인 ===")
    std_pred = df["predicted_volatility"].std()
    std_actual = df["actual_volatility"].std()
    ratio = std_pred / std_actual
    print(f"predicted_volatility 표준편차: {std_pred:.4f}")
    print(f"actual_volatility 표준편차   : {std_actual:.4f}")
    print(f"비율(예측 std / 실측 std)    : {ratio * 100:.1f}% "
          f"(참고: 결과_TaskT_변동성예측_최종.md의 100종목 검증 33.4%)")

    under_rate = (df["predicted_volatility"] < df["actual_volatility"]).mean()
    print(f"predicted < actual인 비율     : {under_rate * 100:.1f}% "
          f"(50%보다 뚜렷이 크면 체계적 과소추정, 50% 근처면 단순 노이즈에 가까움)")

    return per_ticker


def main():
    start_range = "2026-08-14"
    end_range = "2026-09-11"  # get_backtest_dates가 09-14 이상 target_date 안전 필터링
    dates = run_backtest_loop(start_range, end_range)
    backfill_new_rows()
    df, daily = compare_backtest_range(dates)
    if df is not None:
        additional_diagnostics(df)


if __name__ == "__main__":
    main()
