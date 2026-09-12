# garch_baseline_check.py — garch_baseline.py에서 분리된 진단/검증 전용 코드
# (2026-09-12, 가격예측/ 활성·검증 재분리 세션).
#
# 원본 가격예측/garch_baseline.py의 fit_and_forecast_garch()/compute_full_period_sigma()/
# main()/BETA_EXPECTED_RANGE를 값 변경 없이 그대로 옮겼다. 활성 파이프라인
# (가격예측_변동성_공통.py)은 GARCH 파라미터를 자체 캐싱판(get_or_fit_garch_params,
# compute_full_period_sigma_cached)으로 계산하고 build_merged_dataset_v2_volatility_hybrid에
# precomputed_sigma를 항상 넘기므로, 여기 있는 fit_and_forecast_garch()/
# compute_full_period_sigma()는 운영 경로에서 호출되지 않는다.
#
# ⚠️ 2026-09-12 재검증 메모: compute_full_period_sigma()는 split_dataset.py(보호 대상,
# 활성 파일)의 build_merged_dataset_v2_volatility_hybrid()가 precomputed_sigma=None일 때
# 지연 임포트(가격예측.garch_baseline)로 호출하는 폴백 경로가 있다 — 이 파일로 옮긴 뒤에도
# split_dataset.py의 그 지연 임포트 대상은 원본 가격예측/garch_baseline.py 그대로다(그쪽은
# 수정하지 않음, "절대 건드리면 안 되는 것" 범위). 폴백을 실제로 트리거하는 호출부는 전부
# 이 파일(및 이 파일을 참조하는 검증용/의 다른 진단 스크립트)뿐이라 문제 없다.

import sys
from datetime import date

import numpy as np
import pandas as pd
from arch import arch_model

from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
from 가격예측.garch_baseline import (
    compute_log_returns_pct, compute_sma_baseline, forecast_with_fixed_params,
    load_close_prices, rmse_mae, SMA_WINDOW,
)
from 가격예측.pooled_dataset import compute_global_split_dates
from 가격예측.split_dataset import build_merged_dataset_v2

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BETA_EXPECTED_RANGE = (0.85, 0.99)


def fit_and_forecast_garch(returns, train_end, test_start_first_date):
    """train_end까지로 GARCH(1,1) 파라미터를 추정하고, 그 파라미터를 고정한 채 전체 이력
    위에서 재귀적으로 1-step-ahead 조건부 표준편차를 계산한다. test_start_first_date부터
    시작하는 leak-free 예측만 잘라 반환한다.

    ⚠️ 2026-09-06 라벨링 버그 수정: forecast_with_fixed_params()의 출력이 이제
    "start_pos+1부터 시작"하는 규칙으로 바뀌었으므로, 반환 결과의 첫 날짜가 정확히
    test_start_first_date가 되도록 get_loc(test_start_first_date) - 1을 start_pos로 넘긴다.
    test_start_first_date가 returns의 첫 날짜(위치 0)인 특수 호출(compute_full_period_sigma)
    에서는 -1이 나오는데, 그 경우 0으로 clamp한다 — 위치 0 자체는 원리적으로 leak-free 예측이
    존재할 수 없는 시점(그 이전 데이터가 아예 없음)이라 출력이 위치 1부터 시작하는 것이 맞다.

    반환: sigma_pred(pandas Series, index=날짜, 가능한 만큼 test_start_first_date부터),
    params(pandas Series, ω/α/β 등)
    """
    train_returns = returns[returns.index <= train_end]
    am_train = arch_model(train_returns.values, mean="Constant", vol="Garch", p=1, q=1, dist="normal")
    res_train = am_train.fit(disp="off")

    target_pos = returns.index.get_loc(test_start_first_date)
    start_pos = max(target_pos - 1, 0)
    sigma_pred = forecast_with_fixed_params(returns, res_train.params, start_pos)
    return sigma_pred, res_train.params


def compute_full_period_sigma(ticker, start_date, end_date, train_end):
    """전체 구간(train+val+test)에 대한 GARCH(1,1) 1-step-ahead 조건부 표준편차(σ)를 계산한다
    (2026-09-06, 하이브리드 Transformer 피처용 — GARCH가 이미 잡아낸 persistence 신호를
    Transformer 입력에 그대로 제공하기 위함).

    fit_and_forecast_garch()를 코드 변경 없이 그대로 재사용한다 — test_start_first_date
    자리에 returns의 첫 날짜를 넘기기만 하면 전체 구간이 나온다(그 함수는 "test 전용"으로
    작성된 게 아니라 "start 이후 전체"를 반환하는 범용 함수였다). train으로만 파라미터를
    추정해 고정한 뒤 전체 구간에 재귀 적용하므로 leak-free다 — train 구간 자체도 재귀적으로
    "그 시점까지의 과거 수익률"만 사용해 계산되고, val/test로는 파라미터 추정에 전혀
    쓰이지 않는다.

    반환: (sigma_full: pd.Series(index=date, 전체 구간), params: GARCH 파라미터)
    """
    close = load_close_prices(ticker, start_date, end_date)
    returns = compute_log_returns_pct(close)
    return fit_and_forecast_garch(returns, train_end, returns.index[0])


def main():
    end_date = date.today().isoformat()

    print(f"=== GARCH(1,1) 변동성 예측 baseline — {len(ACTIVE_TICKERS)}종목 독립 적합 ===")
    print(f"수익률 정의: 로그수익률x100(%). 실현 변동성 타겟: |수익률|. SMA baseline: 직전{SMA_WINDOW}일 평균|수익률|")

    reference_merged, _ = build_merged_dataset_v2(ACTIVE_TICKERS[0], STOCK_INITIAL_LOAD_START, end_date)
    train_end, val_end = compute_global_split_dates(reference_merged.index)
    print(f"전역 분할 경계: train_end={train_end.date()}  val_end={val_end.date()}")

    results = []
    for ticker in ACTIVE_TICKERS:
        print(f"\n--- {ticker} ---")
        close = load_close_prices(ticker, STOCK_INITIAL_LOAD_START, end_date)
        returns = compute_log_returns_pct(close)

        test_returns = returns[returns.index > val_end]
        if test_returns.empty:
            print(f"⚠️ {ticker}: test 구간 데이터 없음 — 스킵")
            continue
        test_start_first_date = test_returns.index[0]

        sigma_pred, params = fit_and_forecast_garch(returns, train_end, test_start_first_date)
        sma_pred = compute_sma_baseline(returns).loc[sigma_pred.index]

        valid = sigma_pred.index.intersection(sma_pred.dropna().index)
        realized = test_returns.abs().loc[valid]
        sigma_pred_v = sigma_pred.loc[valid]
        sma_pred_v = sma_pred.loc[valid]

        garch_rmse, garch_mae = rmse_mae(sigma_pred_v.values, realized.values)
        sma_rmse, sma_mae = rmse_mae(sma_pred_v.values, realized.values)

        omega, alpha, beta = params["omega"], params["alpha[1]"], params["beta[1]"]
        beta_in_range = BETA_EXPECTED_RANGE[0] <= beta <= BETA_EXPECTED_RANGE[1]
        stationary = (alpha + beta) < 1.0

        print(f"  GARCH 파라미터: omega={omega:.6f}  alpha={alpha:.6f}  beta={beta:.6f}  "
              f"(alpha+beta={alpha+beta:.6f}, 정상성(<1): {stationary})")
        print(f"  beta가 일반적 범위{BETA_EXPECTED_RANGE}에 있는가: {beta_in_range}"
              + ("" if beta_in_range else "  ⚠️ 범위 이탈 — 적합 실패 가능성 의심"))
        print(f"  test(n={len(valid)}): GARCH RMSE={garch_rmse:.6f} MAE={garch_mae:.6f}  |  "
              f"SMA{SMA_WINDOW} RMSE={sma_rmse:.6f} MAE={sma_mae:.6f}")
        beats_sma = garch_rmse < sma_rmse and garch_mae < sma_mae
        print(f"  GARCH가 SMA{SMA_WINDOW} baseline을 이기는가(RMSE·MAE 둘 다 개선): {beats_sma}")

        results.append({
            "ticker": ticker, "n_test": len(valid),
            "omega": omega, "alpha": alpha, "beta": beta,
            "alpha_plus_beta": alpha + beta, "beta_in_range": beta_in_range, "stationary": stationary,
            "garch_rmse": garch_rmse, "garch_mae": garch_mae,
            "sma_rmse": sma_rmse, "sma_mae": sma_mae, "beats_sma": beats_sma,
        })

    if not results:
        print("\n⚠️ 결과 없음 — 모든 종목이 스킵됨")
        return

    print("\n" + "=" * 100)
    print("=== 종목별 GARCH(1,1) 파라미터 ===")
    print("=" * 100)
    print(f"{'ticker':<10}{'omega':>12}{'alpha':>10}{'beta':>10}{'a+b':>10}{'beta 범위 정상':>16}{'정상성':>8}")
    for r in results:
        print(f"{r['ticker']:<10}{r['omega']:>12.6f}{r['alpha']:>10.6f}{r['beta']:>10.6f}"
              f"{r['alpha_plus_beta']:>10.6f}{str(r['beta_in_range']):>16}{str(r['stationary']):>8}")

    print("\n" + "=" * 100)
    print("=== 종목별 + 8종목 평균 RMSE/MAE (GARCH vs SMA baseline) ===")
    print("=" * 100)
    print(f"{'ticker':<10}{'n':>6}{'GARCH RMSE':>12}{'GARCH MAE':>11}{'SMA RMSE':>10}{'SMA MAE':>9}{'GARCH 승':>9}")
    for r in results:
        print(f"{r['ticker']:<10}{r['n_test']:>6}{r['garch_rmse']:>12.6f}{r['garch_mae']:>11.6f}"
              f"{r['sma_rmse']:>10.6f}{r['sma_mae']:>9.6f}{str(r['beats_sma']):>9}")

    avg_garch_rmse = float(np.mean([r["garch_rmse"] for r in results]))
    avg_garch_mae = float(np.mean([r["garch_mae"] for r in results]))
    avg_sma_rmse = float(np.mean([r["sma_rmse"] for r in results]))
    avg_sma_mae = float(np.mean([r["sma_mae"] for r in results]))
    n_beats = sum(r["beats_sma"] for r in results)

    print(f"{'평균(8종목)':<10}{'-':>6}{avg_garch_rmse:>12.6f}{avg_garch_mae:>11.6f}"
          f"{avg_sma_rmse:>10.6f}{avg_sma_mae:>9.6f}{f'{n_beats}/{len(results)}':>9}")

    print("\n" + "=" * 100)
    print("=== 핵심 확인 포인트 ===")
    print("=" * 100)
    overall_beats = avg_garch_rmse < avg_sma_rmse and avg_garch_mae < avg_sma_mae
    print(f"8종목 평균 기준 GARCH가 SMA{SMA_WINDOW} baseline을 이기는가: {overall_beats}")
    print(f"  평균 RMSE: GARCH {avg_garch_rmse:.6f} vs SMA {avg_sma_rmse:.6f} "
          f"({'개선' if avg_garch_rmse < avg_sma_rmse else '악화'} "
          f"{abs(avg_garch_rmse-avg_sma_rmse)/avg_sma_rmse*100:.2f}%)")
    print(f"  평균 MAE : GARCH {avg_garch_mae:.6f} vs SMA {avg_sma_mae:.6f} "
          f"({'개선' if avg_garch_mae < avg_sma_mae else '악화'} "
          f"{abs(avg_garch_mae-avg_sma_mae)/avg_sma_mae*100:.2f}%)")
    print(f"종목별 GARCH 승리: {n_beats}/{len(results)}개 종목")
    n_beta_ok = sum(r["beta_in_range"] for r in results)
    n_stationary = sum(r["stationary"] for r in results)
    print(f"beta가 일반 범위{BETA_EXPECTED_RANGE} 안: {n_beta_ok}/{len(results)}개 종목")
    print(f"정상성(alpha+beta<1) 만족: {n_stationary}/{len(results)}개 종목")


if __name__ == "__main__":
    main()
