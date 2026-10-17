# baselines.py — 베이스라인 3종(GARCH(1,1)·SMA20·Parkinson-SMA20)의 계산과 GARCH 파라미터 파일 캐시.
# 파이프라인 단계: 피처(GARCH σ)·평가(val RMSE)·저장(다음 거래일 baseline 점예측). 주요 진입 함수: compute_ticker_baselines.
# 수익률은 로그수익률x100(%), 실현 변동성은 |수익률|이다.
# 배경: 결정기록.md의 "GARCH는 종목별로 train 구간에서만 적합하고 고정 파라미터로 재귀 계산", "GARCH 파라미터 캐시 30일"

import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
from arch import arch_model

from db_manager import get_db_connection

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── 공통 지표 ──────────────────────────────────────────────────────────────────
SMA_WINDOW = 20


def load_close_prices(ticker, start_date, end_date):
    """
    설명: daily_stock_prices에서 종목의 종가를 날짜 인덱스 Series로 읽는다.
    주의 사항: DB 읽기. 운영 경로는 가격 캐시(prices)를 넘겨 이 함수를 거치지 않는다.
    입력 타입->반환 타입: (str, str, str) -> pd.Series
    """
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패 — load_close_prices")
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT date, close FROM daily_stock_prices
                WHERE ticker = %s AND date BETWEEN %s AND %s
                ORDER BY date
                """,
                (ticker, start_date, end_date),
            )
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["date", "close"])
    df["close"] = df["close"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"]


def compute_log_returns_pct(close):
    """
    설명: 로그수익률 x 100을 계산하고 첫 행(NaN)을 뺀다.
    주의 사항: 단위는 %(arch 패키지 권장 스케일). 단순 등락률 change_rate와 다른 값이다.
    입력 타입->반환 타입: (pd.Series) -> pd.Series
    """
    log_ret = np.log(close / close.shift(1)) * 100
    return log_ret.dropna()


def rmse_mae(pred, actual):
    """
    설명: 예측과 실제의 RMSE와 MAE를 계산한다.
    주의 사항: 두 배열의 길이·정렬이 같아야 한다(NaN은 호출부에서 미리 빼야 함).
    입력 타입->반환 타입: (np.ndarray, np.ndarray) -> (float, float)
    """
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    mae = float(np.mean(np.abs(pred - actual)))
    return rmse, mae


def compute_sma_baseline(returns, window=SMA_WINDOW):
    """
    설명: 직전 window일 평균 |수익률|(SMA baseline)을 날짜별로 계산한다.
    주의 사항: shift(1)을 먼저 해서 당일 값을 포함하지 않는다.
    입력 타입->반환 타입: (pd.Series, int) -> pd.Series
    """
    return returns.abs().shift(1).rolling(window, min_periods=window).mean()


# ── GARCH(파라미터 캐시 포함) ─────────────────────────────────────────────────────
def forecast_with_fixed_params(returns, params, start_pos):
    """
    설명: 고정 GARCH(1,1) 파라미터로 start_pos+1번째 날짜부터 1-step-ahead 조건부 σ를 재귀 계산한다.
    주의 사항: arch 원시 출력 행 i는 index[start_pos+i+1]의 예측이라 그 날짜로 다시 붙인다(그대로 쓰면 누수).
    입력 타입->반환 타입: (pd.Series, pd.Series, int) -> pd.Series
    """
    am_full = arch_model(returns.values, mean="Constant", vol="Garch", p=1, q=1, dist="normal")
    res_fixed = am_full.fix(params)
    fc = res_fixed.forecast(horizon=1, start=start_pos, reindex=False)
    sigma_raw = np.sqrt(fc.variance.values[:, 0])

    n = len(sigma_raw)
    target_positions = np.arange(start_pos + 1, start_pos + 1 + n)
    keep = target_positions < len(returns)
    dates = returns.index[target_positions[keep]]
    return pd.Series(sigma_raw[keep], index=dates)


def garch_sigma_series(returns, params):
    """
    설명: 전체 구간 GARCH σ를 계산한다(날짜 t 값 = t-1까지의 수익률로 낸 t 예측).
    주의 사항: 학습 frame과 서빙 창이 함께 쓰는 단일 경로다.
    입력 타입->반환 타입: (pd.Series, pd.Series) -> pd.Series
    """
    return forecast_with_fixed_params(returns, params, 0)


def forecast_next_day_sigma(returns, params):
    """
    설명: returns의 마지막 날짜 다음 거래일 GARCH σ를 예측한다(서빙용).
    주의 사항: returns 전체(마지막 날 포함)를 쓴 값이라 재라벨링이 필요 없다.
    입력 타입->반환 타입: (pd.Series, pd.Series) -> float
    """
    am_full = arch_model(returns.values, mean="Constant", vol="Garch", p=1, q=1, dist="normal")
    res_fixed = am_full.fix(params)
    fc = res_fixed.forecast(horizon=1)
    return float(np.sqrt(fc.variance.values[-1, 0]))


# 캐시된 GARCH 파라미터를 다시 적합하지 않고 재사용하는 최대 경과 일수(월 1회 재적합).
GARCH_REFIT_MAX_AGE_DAYS = 30

_HERE = os.path.dirname(__file__)
GARCH_PARAM_CACHE_DIR = os.path.join(_HERE, "checkpoints", "garch_params")


def _garch_cache_path(ticker, cache_dir=None):
    """
    설명: 종목의 GARCH 캐시 파일 경로(<cache_dir>/<ticker>.json)를 만든다.
    주의 사항: cache_dir가 없으면 운영 캐시 폴더(GARCH_PARAM_CACHE_DIR)다.
    입력 타입->반환 타입: (str, str|None) -> str
    """
    return os.path.join(cache_dir or GARCH_PARAM_CACHE_DIR, f"{ticker}.json")


def _load_garch_cache(ticker, cache_dir=None):
    """
    설명: 종목의 GARCH 캐시 JSON을 읽어 dict로 돌려준다. 파일이 없으면 None.
    주의 사항: 파일 읽기.
    입력 타입->반환 타입: (str, str|None) -> dict|None
    """
    path = _garch_cache_path(ticker, cache_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_garch_cache(ticker, params, train_end, cache_dir=None):
    """
    설명: GARCH 파라미터를 fitted_at(오늘)·train_end와 함께 캐시 JSON으로 저장한다.
    주의 사항: 파일 쓰기(폴더가 없으면 만든다). 기본 위치는 운영 캐시 폴더다.
    입력 타입->반환 타입: (str, pd.Series, pd.Timestamp, str|None) -> None
    """
    os.makedirs(cache_dir or GARCH_PARAM_CACHE_DIR, exist_ok=True)
    payload = {
        "params": {k: float(v) for k, v in params.items()},
        "fitted_at": date.today().isoformat(),
        "train_end": str(pd.Timestamp(train_end).date()),
    }
    with open(_garch_cache_path(ticker, cache_dir), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def get_or_fit_garch_params(ticker, returns, train_end, max_age_days=GARCH_REFIT_MAX_AGE_DAYS, cache_dir=None,
                            read_only=False):
    """
    설명: 캐시가 max_age_days 이내면 그 파라미터를, 아니면 train_end까지로 새로 적합한 파라미터를 (params, 재적합 여부)로 돌려준다.
    주의 사항: 재적합하면 캐시 파일을 쓴다. 캐시는 fitted_at만 보고 train_end는 보지 않는다(과거 날짜 백테스트에 누수). read_only면 적합 대신 예외.
    입력 타입->반환 타입: (str, pd.Series, pd.Timestamp, int, str|None, bool) -> (pd.Series, bool)
    """
    cached = _load_garch_cache(ticker, cache_dir)
    if cached is not None:
        fitted_at = date.fromisoformat(cached["fitted_at"])
        age_days = (date.today() - fitted_at).days
        if age_days <= max_age_days:
            return pd.Series(cached["params"]), False
    if read_only:
        raise RuntimeError(f"[GARCH-READ-ONLY] {ticker}: 캐시가 없거나 {max_age_days}일 지나 만료됨 — 읽기 전용이라 적합하지 않음")

    train_returns = returns[returns.index <= train_end]
    am_train = arch_model(train_returns.values, mean="Constant", vol="Garch", p=1, q=1, dist="normal")
    # 표준오차는 쓰지 않으므로 robust 공분산 계산을 생략한다.
    res_train = am_train.fit(disp="off", cov_type="classic")
    params = res_train.params
    _save_garch_cache(ticker, params, train_end, cache_dir)
    return params, True


def compute_full_period_sigma_cached(returns, params):
    """
    설명: 이미 가진(캐시된) 파라미터로 재적합 없이 전체 구간 GARCH σ를 계산한다(garch_sigma_series와 같음).
    주의 사항: 없음.
    입력 타입->반환 타입: (pd.Series, pd.Series) -> pd.Series
    """
    return garch_sigma_series(returns, params)


# ── Parkinson ─────────────────────────────────────────────────────────────────────
PARK_WINDOW = 20


def load_high_low(ticker, start_date, end_date):
    """
    설명: daily_stock_prices에서 종목의 고가·저가를 날짜 인덱스 DataFrame으로 읽는다.
    주의 사항: DB 읽기. 운영 경로는 가격 캐시(prices)를 넘겨 이 함수를 거치지 않는다.
    입력 타입->반환 타입: (str, str, str) -> pd.DataFrame
    """
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
    """
    설명: Parkinson 추정량 sqrt(ln(high/low)^2 / (4 ln2)) x 100을 날짜별로 계산한다.
    주의 사항: 당일 고가·저가를 쓰는 원시값이라 예측에는 rolling 평균 뒤 shift(1)한 값만 써야 한다(그대로 쓰면 누수).
    입력 타입->반환 타입: (pd.Series, pd.Series) -> pd.Series
    """
    raw = np.sqrt((np.log(high / low) ** 2) / (4 * np.log(2)))
    return raw * 100


# ── baseline 3종: val 구간 RMSE(게이트용) + 다음 거래일 점예측(감사 컬럼용) ──────────────

def compute_ticker_baselines(ticker, start_date, end_date, train_end, val_end, prices=None, garch_cache_dir=None,
                             garch_read_only=False):
    """
    설명: 종목 하나의 GARCH 파라미터·전체 구간 σ·val RMSE 3종·다음 거래일 baseline 3종을 한 번에 계산해 dict로 돌려준다.
    주의 사항: GARCH 재적합 시 캐시 파일을 쓴다(garch_read_only면 예외). prices가 없으면 DB 읽기.
    입력 타입->반환 타입: (str, str, str, pd.Timestamp, pd.Timestamp, pd.DataFrame|None, str|None, bool) -> dict
    """
    if prices is not None:
        prices_by_date = prices.set_index("date")
        close = prices_by_date["close"]
        hl = prices_by_date[["high", "low"]]
    else:
        close = load_close_prices(ticker, start_date, end_date)
        hl = load_high_low(ticker, start_date, end_date)
    returns = compute_log_returns_pct(close)
    params, refit = get_or_fit_garch_params(ticker, returns, train_end, cache_dir=garch_cache_dir,
                                            read_only=garch_read_only)
    sigma_full = compute_full_period_sigma_cached(returns, params)

    sma_full = compute_sma_baseline(returns)
    park_raw = compute_parkinson_vol_pct(hl["high"], hl["low"])
    park_ma20 = park_raw.rolling(PARK_WINDOW, min_periods=PARK_WINDOW).mean().shift(1)

    val_mask = (returns.index > train_end) & (returns.index <= val_end)
    realized_val = returns.abs()[val_mask]
    sigma_val = sigma_full.reindex(realized_val.index)
    sma_val = sma_full.reindex(realized_val.index)
    park_val = park_ma20.reindex(realized_val.index)
    valid = sigma_val.notna() & sma_val.notna() & park_val.notna()

    garch_rmse, _ = rmse_mae(sigma_val[valid].values, realized_val[valid].values)
    sma_rmse, _ = rmse_mae(sma_val[valid].values, realized_val[valid].values)
    park_rmse, _ = rmse_mae(park_val[valid].values, realized_val[valid].values)

    next_garch = forecast_next_day_sigma(returns, params)
    next_sma = float(returns.abs().tail(SMA_WINDOW).mean())
    next_park = float(park_raw.tail(PARK_WINDOW).mean())

    return {
        "params": params, "sigma_full": sigma_full, "refit": refit,
        "val_rmse": {"garch": garch_rmse, "sma20": sma_rmse, "parkinson": park_rmse},
        "n_val_valid": int(valid.sum()),
        "next_day": {"garch": next_garch, "sma20": next_sma, "parkinson": next_park},
    }
