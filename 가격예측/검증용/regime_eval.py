# 가격예측/검증용/regime_eval.py — 변동성 예측 평가 공용 함수(2026-09-27 세션에서 반복 사용한 로직 정리)
#
# 이 세션의 실험들(결과_TaskT_급변구간_손실함수_시도.md [9]~[12], CLAUDE.md "게이트 집계 버그" 절)이 스크래치
# 스크립트마다 복붙해 쓰던 평가 로직을 한 곳에 모았다. 다음에 무엇을 시도하든 같은 잣대로 비교하기 위함.
#   (a) 급변일/평상일/지속중 그룹 분리 — 결과 문서 [1]의 정의 그대로
#   (b) 운영 게이트와 같은 "종목별 RMSE 평균" 비교(2026-09-27 게이트 집계 버그 수정 후 정본)
#   (c) 오라클 방향 가격 환산(방향은 실제 값을 쓴다 — "모델이 방향을 맞힌다"는 주장이 아님, 발표 시 반드시 명시)
# test 구간 평가에 쓰지 말 것 — 지금까지의 모든 결과는 val(fold3) 기준이다.

import numpy as np
import pandas as pd

from 가격예측.garch_baseline import compute_log_returns_pct, load_close_prices
from 가격예측.검증용.highvol_regime_diagnosis import compute_big_move_labels, compute_first_day_labels

TOP_QUANTILE = 0.8          # 상위 20% = 실측 타겟이 해당 평가 구간 80분위 초과
FIRST_DAY_ISOLATION_N = 5   # "국면 첫날" = 직전 5거래일 안에 다른 급변일이 없는 급변일


# ── (b) 게이트 ─────────────────────────────────────────────────────────────────

def rmse(pred, actual):
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def per_ticker_rmse(pred, actual, ticker_ids):
    """운영 게이트와 같은 집계: 종목별 RMSE를 구한 뒤 단순 평균. 하이브리드와 baseline에 반드시 같은 방식을
    써야 한다(pooled vs 종목평균을 섞으면 젠센 부등식 때문에 종목평균 쪽이 구조적으로 유리 — 게이트 버그의 원인)."""
    pred, actual, ticker_ids = np.asarray(pred, float), np.asarray(actual, float), np.asarray(ticker_ids)
    return float(np.mean([rmse(pred[ticker_ids == t], actual[ticker_ids == t]) for t in np.unique(ticker_ids)]))


def gate_verdict(model_rmse, baseline_rmses):
    """baseline_rmses: {"garch": x, "sma20": y, "parkinson": z}(같은 집계 방식). 셋 다 이겨야 통과."""
    beats = {k: model_rmse < v for k, v in baseline_rmses.items()}
    return {"model_rmse": model_rmse, **{f"beats_{k}": b for k, b in beats.items()}, "passed": all(beats.values())}


def std_ratio(pred, actual):
    """예측 std / 실측 std — 분산 붕괴 지표(운영 LSTM val 약 0.27)."""
    return float(np.std(pred) / np.std(actual))


# ── (a) 그룹 분리 ──────────────────────────────────────────────────────────────

def bigmove_labels(tickers, start_date, end_date):
    """종목별 급변일(|r| >= 2 x 직전60일평균)·국면 첫날 라벨을 (ticker, date) 행 DataFrame으로 반환."""
    rows = []
    for t in tickers:
        ret = compute_log_returns_pct(load_close_prices(t, start_date, end_date))
        bm = compute_big_move_labels(ret)
        fd = compute_first_day_labels(bm, FIRST_DAY_ISOLATION_N)
        rows.append(pd.DataFrame({"ticker": t, "date": bm.index, "bigmove": bm.values, "firstday": fd.values}))
    return pd.concat(rows, ignore_index=True)


def regime_masks(y, bigmove, firstday, top_quantile=TOP_QUANTILE):
    """y: 실측 타겟, bigmove/firstday: 같은 행 순서의 bool 배열. 결과 문서 [1]/[9]의 그룹 정의.
    ① 급변일 = 상위20% ∩ 급변일, ③ 평상일 = 하위80%, ④ 지속중 = ① 중 국면 첫날이 아닌 날.
    (상위20%인데 2x 미달인 행은 ①·③ 어디에도 속하지 않는다.)
    ⚠️ 실측으로 나눈 사후 그룹이라 예측 시점엔 알 수 없다 — 그룹별 순위만으로 운영 규칙을 정하면 안 된다."""
    y = np.asarray(y, float); bigmove = np.asarray(bigmove, bool); firstday = np.asarray(firstday, bool)
    top = y > np.quantile(y, top_quantile)
    return {"전체": np.ones(len(y), bool), "①급변일": top & bigmove, "③평상일": ~top, "④지속중": top & bigmove & ~firstday}


# ── (c) 오라클 방향 가격 환산 ──────────────────────────────────────────────────

def oracle_price_estimate(close_prev, actual_close, vol_pred_pct):
    """다음날 가격 = 전일종가 x exp(sign(실제 로그수익률) x 예측 변동성/100). 방향은 실제 값(오라클)."""
    close_prev, actual_close = np.asarray(close_prev, float), np.asarray(actual_close, float)
    sign = np.sign(np.log(actual_close / close_prev))
    return close_prev * np.exp(sign * np.asarray(vol_pred_pct, float) / 100)


def price_errors(estimate, actual_close, mask=None):
    """(원 단위 RMSE, %오차 RMSE). mask가 있으면 그 행만, NaN 행은 제외."""
    est, act = np.asarray(estimate, float), np.asarray(actual_close, float)
    if mask is not None:
        est, act = est[mask], act[mask]
    ok = np.isfinite(est) & np.isfinite(act)     # baseline 미정의 행(최근 상장 종목 초기 등)은 제외
    est, act = est[ok], act[ok]
    err = est - act
    return float(np.sqrt(np.mean(err ** 2))), float(np.sqrt(np.mean((err / act * 100) ** 2)))
