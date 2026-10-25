# 가격예측_변동성_공통.py — 하이브리드(GARCH σ 피처 + pooled LSTM) 변동성 예측 일일 파이프라인의 오케스트레이터와 설정.
# 파이프라인 단계: 입력 점검(가격 정렬·지수 신선도) -> 피처·baseline -> 시퀀스 -> [튜닝 값 읽기] -> 학습 -> 평가
#   -> [전체 재학습] -> 저장+예측 -> 실행 기록.
# 주요 진입 함수: run_pooled_volatility_pipeline(가격예측_변동성_일일수집.py가 호출).
# 배경: 결정기록.md의 "매일 전체 재학습, 학습 실패는 전체 실패·예측만 종목별 격리", "예측은 게이트와 무관하게 저장한다",
#       "가격 정렬 점검", "시장지수 신선도 점검과 input_data_suspect", "게이트는 train→val 모델로 판정하고 서빙은 전체 재학습 모델로 한다"

import os
import time
import traceback

import numpy as np
import torch

from config import redact

from constants import MIN_TRAIN_SEQUENCES
from db_manager import get_db_connection
from 가격예측 import store
from 가격예측 import 하이퍼파라미터_튜닝 as tuning
from 가격예측.baselines import compute_ticker_baselines
from 가격예측.evaluate import evaluate_gate
from 가격예측.momentum_feature import load_true_kospi
from 가격예측.pooled_dataset import build_pooled_sequences, compute_global_split_dates
from 가격예측.sequence_dataset import FeatureScaler
from 가격예측.split_dataset import (
    build_merged_dataset_v2_volatility_hybrid,
    valid_feature_dates,
)
from 가격예측.model import MODEL_CLASS_NAME
from 가격예측.train_common import TrainConfig, train_fixed_epochs, train_model
from 시장지표.feature_loader import load_indicator_cache, load_price_cache

LOOKBACK = 20

# 학습 하이퍼파라미터 기본값(운영값). 종목 수가 크게 달라지면 embedding_dim은 사람이 다시 검토해야 한다(자동 조정 없음).
DEFAULT_TRAIN_CONFIG = TrainConfig()

_HERE = os.path.dirname(__file__)
CHECKPOINT_DIR = os.path.join(_HERE, "checkpoints", "pooled_volatility_hybrid")
# 운영 경로 기본값. run_pooled_volatility_pipeline의 checkpoint_dir·garch_cache_dir·log_dir 인자로 바꿀 수 있다.
LOG_DIR = os.path.join(_HERE, "logs")

# 전체 재학습 스위치. 기본 켜짐, 환경변수 FINAL_REFIT=0이면 게이트 모델로 서빙한다(이 단계 추가 전과 같은 동작·기록).
FINAL_REFIT_ENV = "FINAL_REFIT"


def _model_kwargs(num_stocks, config):
    """
    설명: 체크포인트 meta에 남길 PooledLSTMRegressor 생성자 인자(model_kwargs)를 만든다.
    주의 사항: 체크포인트로 모델을 다시 만들 때 이 키 그대로 쓰인다(키 이름 바꾸지 말 것).
    입력 타입->반환 타입: (int, TrainConfig) -> dict
    """
    return dict(
        lookback=LOOKBACK, num_stocks=num_stocks, embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout,
    )


# ── pooled 하이브리드 학습 ────────────────────────────────────────────────────────

def load_shared_cache(start_date, end_date):
    """
    설명: 종목과 무관한 거시지표 9종·KOSPI 원자료를 한 번 읽어 학습·예측이 함께 쓸 dict로 돌려준다.
    주의 사항: DB 읽기. 키 이름은 split_dataset 피처 함수의 키워드 인자 이름과 같아야 한다.
    입력 타입->반환 타입: (str, str) -> dict
    """
    return {
        "precomputed_indicators": load_indicator_cache(end_date, start_date),
        "true_kospi": load_true_kospi(start_date, end_date),
    }


def compute_split_dates(ref_ticker, start_date, end_date, shared_cache, prices):
    """
    설명: 기준 종목의 유효 피처 날짜로 전역 train/val 분할 경계를 구한다.
    주의 사항: 운영 기준 종목은 tickers[0](ACTIVE_TICKERS 첫 종목)이다.
    입력 타입->반환 타입: (str, str, str, dict, pd.DataFrame) -> (pd.Timestamp, pd.Timestamp)
    """
    dates = valid_feature_dates(ref_ticker, start_date, end_date, **shared_cache, precomputed_prices=prices)
    return compute_global_split_dates(dates)


def prepare_pooled_training_data(tickers, start_date, end_date, shared_cache=None, prices_by_ticker=None,
                                 garch_cache_dir=None, garch_read_only=False):
    """
    설명: 분할 경계·종목별 baseline·pooled 시퀀스(train/val/test)·train으로 맞춘 scaler를 만들어 dict로 돌려준다(학습 직전까지).
    주의 사항: 일일 학습과 주간 튜닝이 같은 함수를 쓴다. 캐시 인자가 없으면 DB를 읽고, GARCH 재적합 시 캐시 파일을 쓴다.
    입력 타입->반환 타입: (list[str], str, str, dict|None, dict|None, str|None, bool) -> dict
    """
    if shared_cache is None:
        shared_cache = load_shared_cache(start_date, end_date)
    if prices_by_ticker is None:
        prices_by_ticker = load_price_cache(tickers, start_date, end_date)

    train_end, val_end = compute_split_dates(tickers[0], start_date, end_date, shared_cache,
                                             prices_by_ticker[tickers[0]])

    baselines_by_ticker = {}
    for ticker in tickers:
        baselines_by_ticker[ticker] = compute_ticker_baselines(
            ticker, start_date, end_date, train_end, val_end, prices=prices_by_ticker[ticker],
            garch_cache_dir=garch_cache_dir, garch_read_only=garch_read_only,
        )

    def hybrid_build_fn(ticker, s, e, precomputed_indicators=None, true_kospi=None):
        """
        설명: 종목 하나의 GARCH σ 포함 학습 frame을 만든다(build_pooled_sequences에 넘기는 build_fn).
        주의 사항: build_pooled_sequences가 인자 이름(precomputed_indicators·true_kospi)을 보고 캐시를 넘긴다 — 이름을 바꾸지 말 것.
        입력 타입->반환 타입: (str, str, str, dict|None, pd.Series|None) -> (pd.DataFrame, dict)
        """
        b = baselines_by_ticker[ticker]
        return build_merged_dataset_v2_volatility_hybrid(
            ticker, s, e, precomputed_sigma=b["sigma_full"], garch_params=b["params"],
            precomputed_indicators=precomputed_indicators, true_kospi=true_kospi,
            precomputed_prices=prices_by_ticker[ticker],
        )

    splits, feature_cols, ticker_to_id, (train_end2, val_end2) = build_pooled_sequences(
        tickers, start_date, end_date, lookback=LOOKBACK, build_fn=hybrid_build_fn,
        shared_cache=shared_cache,
    )
    assert train_end2 == train_end and val_end2 == val_end
    X_train, y_train, tid_train, _, _ = splits["train"]
    X_val, y_val, tid_val, _, _ = splits["val"]

    n_train_by_ticker = {t: int((tid_train == tid).sum()) for t, tid in ticker_to_id.items()}
    thin = {t: n for t, n in n_train_by_ticker.items() if n < MIN_TRAIN_SEQUENCES}
    if thin:
        print(f"⚠️ train 시퀀스가 {MIN_TRAIN_SEQUENCES}개 미만인 종목(임베딩이 사실상 학습 안 될 위험): {thin}")
        print("   — 종목_월간갱신.py 게이팅을 통과한 종목이라면 나타나면 안 되는 상태다. 재검토 필요.")

    scaler = FeatureScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    return {"splits": splits, "feature_cols": feature_cols, "ticker_to_id": ticker_to_id,
            "baselines_by_ticker": baselines_by_ticker, "train_end": train_end, "val_end": val_end,
            "scaler": scaler, "X_train_s": X_train_s, "X_val_s": X_val_s,
            "n_train_by_ticker": n_train_by_ticker}


def train_daily_pooled_model(tickers, start_date, end_date, train_config=DEFAULT_TRAIN_CONFIG, shared_cache=None,
                             prices_by_ticker=None, garch_cache_dir=None, garch_read_only=False):
    """
    설명: 전 종목 baseline을 계산하고 하나의 pooled 데이터셋으로 LSTM을 학습한 뒤 게이트까지 판정한 결과 dict를 돌려준다.
    주의 사항: 캐시 인자가 없으면 DB를 읽는다. GARCH 재적합 시 캐시 파일을 쓴다. train 시퀀스가 적은 종목은 경고만 낸다.
    입력 타입->반환 타입: (list[str], str, str, TrainConfig, dict|None, dict|None, str|None, bool) -> dict
    """
    data = prepare_pooled_training_data(tickers, start_date, end_date, shared_cache=shared_cache,
                                        prices_by_ticker=prices_by_ticker, garch_cache_dir=garch_cache_dir,
                                        garch_read_only=garch_read_only)
    return train_and_gate(data, len(tickers), train_config)


def train_and_gate(data, num_stocks, train_config):
    """
    설명: prepare_pooled_training_data 결과로 train 학습(val로 early stopping) → val 게이트 판정까지 하는 일일 학습 경로.
    주의 사항: 주간 튜닝의 최종 후보 게이트 판정도 이 함수를 그대로 쓴다(따로 구현하지 말 것). 시드는 train_config.seed로 고정된다.
    입력 타입->반환 타입: (dict, int, TrainConfig) -> dict
    """
    splits, scaler = data["splits"], data["scaler"]
    X_train_s, X_val_s = data["X_train_s"], data["X_val_s"]
    _, y_train, tid_train, _, _ = splits["train"]
    _, y_val, tid_val, _, _ = splits["val"]
    baselines_by_ticker = data["baselines_by_ticker"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = train_model(
        X_train_s, y_train, tid_train, X_val_s, y_val, tid_val,
        num_stocks, LOOKBACK, train_config, device, label="pooled-volatility-hybrid",
    )

    gate = evaluate_gate(result["model"], result["val_loader"], device, baselines_by_ticker)

    return {
        "model": result["model"], "scaler": scaler, "feature_cols": data["feature_cols"],
        "ticker_to_id": data["ticker_to_id"], "device": device, "gate": gate,
        "baselines_by_ticker": baselines_by_ticker,
        "train_end": data["train_end"], "val_end": data["val_end"],
        "n_train": len(X_train_s), "n_val": len(X_val_s),
        "n_train_by_ticker": data["n_train_by_ticker"],
        "history": result["history"], "best_epoch": result["best_epoch"],
        "best_val_loss": result["best_val_loss"],
        "overfit_ratio_at_end": result["overfit_ratio_at_end"],
        "train_config": train_config,
    }


# ── 전체 재학습(서빙 모델) ────────────────────────────────────────────────────────

def final_refit_enabled(flag=None):
    """
    설명: 전체 재학습을 할지 정한다. flag를 주면 그 값, 없으면 환경변수 FINAL_REFIT("0"이면 끔, 그 밖은 켬).
    주의 사항: 끄면 게이트 모델이 서빙되고 체크포인트 meta·MLflow 기록이 이 단계 추가 전과 같다.
    입력 타입->반환 타입: (bool|None) -> bool
    """
    if flag is not None:
        return bool(flag)
    return os.environ.get(FINAL_REFIT_ENV, "1") != "0"


def refit_on_all_data(data, num_stocks, train_config, n_epochs, device):
    """
    설명: prepare_pooled_training_data의 train+val+test 시퀀스 전부로 scaler를 다시 맞추고 n_epochs epoch 재학습한 결과를 돌려준다.
    주의 사항: 검증 구간이 없어 early stopping 대신 게이트 모델의 best epoch 수를 받는다. 이 모델은 test를 학습하므로 test 구간 평가는 in-sample이다.
    입력 타입->반환 타입: (dict, int, TrainConfig, int, torch.device) -> dict
    """
    t0 = time.time()
    parts = [data["splits"][s] for s in ("train", "val", "test")]
    X_all = np.concatenate([p[0] for p in parts], axis=0)
    y_all = np.concatenate([p[1] for p in parts], axis=0)
    tid_all = np.concatenate([p[2] for p in parts], axis=0)
    last_target_date = max(np.asarray(p[3]).astype("datetime64[D]").max() for p in parts)
    scaler = FeatureScaler().fit(X_all)
    X_all_s = scaler.transform(X_all).astype(np.float32)
    result = train_fixed_epochs(X_all_s, y_all, tid_all, num_stocks, LOOKBACK, train_config, device, n_epochs,
                                label="pooled-volatility-refit")
    return {"model": result["model"], "scaler": scaler, "history": result["history"], "epochs": result["epochs"],
            "n_rows": int(len(X_all)), "n_by_split": {s: int(len(p[0])) for s, p in zip(("train", "val", "test"), parts)},
            "last_target_date": str(last_target_date), "elapsed_sec": round(time.time() - t0, 1)}


# ── 입력 점검: 주가지수 신선도(경고만, 파이프라인은 멈추지 않음) ───────────────────────────
INDEX_FRESHNESS_CODES = ("KOSPI", "KOSDAQ")


def check_market_index_freshness(end_date):
    """
    설명: 주가 최신 거래일(<= end_date) 대비 KOSPI·KOSDAQ가 몇 거래일 뒤처졌는지 세어 {코드: 거래일 수}로 돌려준다.
    주의 사항: DB 읽기. 뒤처지면 경고만 출력한다. 결측일 지수 피처는 마지막 값으로 채워지고 excess_return_z가 오염된다.
    입력 타입->반환 타입: (str) -> dict[str, int]
    """
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패 — check_market_index_freshness")
        with conn.cursor() as cur:
            cur.execute(
                "SELECT indicator_code, MAX(date) FROM market_indicators "
                "WHERE indicator_code = ANY(%s) AND date <= %s GROUP BY indicator_code",
                (list(INDEX_FRESHNESS_CODES), end_date),
            )
            latest = dict(cur.fetchall())
            lag = {}
            for code in INDEX_FRESHNESS_CODES:
                last = latest.get(code)
                cur.execute(
                    "SELECT COUNT(DISTINCT date), MAX(date) FROM daily_stock_prices "
                    "WHERE date <= %s AND (%s::date IS NULL OR date > %s::date)",
                    (end_date, last, last),
                )
                n_behind, stock_last = cur.fetchone()
                lag[code] = int(n_behind)
                if n_behind > 0:
                    print(f"⚠️ {code} 지수 결측: market_indicators 최신={last}, 주가 최신={stock_last} — "
                          f"{n_behind}거래일째 결측. 결측일 KOSPI/KOSDAQ 피처는 마지막 값으로 채워지고 "
                          f"excess_return_z_lag1은 KOSPI 수익률 0으로 계산되어 예측이 오염될 수 있음.")
    return lag


# ── 입력 점검: 종목별 주가 최신일 정렬 ──────────────────────────────────────────────
# 공통일 = 종목별 최신일의 최빈값(동률이면 불일치). 이탈이 MAX_DEVIANT_TICKERS 이하면 그 종목만 저장에서 빼고 진행,
# 초과·동률이면 재조회 후에도 같을 때 학습 전에 중단한다.
MAX_DEVIANT_TICKERS = 5
PRICE_ALIGNMENT_RETRIES = 3          # 첫 조회 뒤 재조회 횟수(최대 대기 = RETRIES x INTERVAL)
PRICE_ALIGNMENT_RETRY_INTERVAL_SEC = 300


def summarize_price_alignment(latest_by_ticker, tickers, max_deviant=MAX_DEVIANT_TICKERS):
    """
    설명: 종목별 최신일로 공통일·이탈 종목·날짜 분포·가격 없는 종목·진행 가능 여부를 판정해 dict로 돌려준다.
    주의 사항: DB 없이 쓰는 순수 함수. 최빈 날짜가 동률이거나 전부 없으면 common_date=None(진행 불가).
    입력 타입->반환 타입: (dict[str, date|None], list[str], int) -> dict
    """
    missing = [t for t in tickers if latest_by_ticker.get(t) is None]
    distribution = {}
    for t in tickers:
        d = latest_by_ticker.get(t)
        if d is not None:
            distribution[d] = distribution.get(d, 0) + 1
    top = max(distribution.values(), default=0)
    modes = [d for d, n in distribution.items() if n == top]
    common_date = modes[0] if len(modes) == 1 else None
    deviants = ({t: latest_by_ticker.get(t) for t in tickers if latest_by_ticker.get(t) != common_date}
                if common_date is not None else {})
    return {"ok": common_date is not None and len(deviants) <= max_deviant,
            "common_date": common_date, "deviants": deviants,
            "distribution": dict(sorted(distribution.items())), "missing": missing}


def _fetch_latest_price_dates(tickers, end_date):
    """
    설명: daily_stock_prices에서 종목별 최신 날짜(<= end_date)를 {종목: 날짜}로 읽는다.
    주의 사항: DB 읽기. 행이 없는 종목은 결과에 없다.
    입력 타입->반환 타입: (list[str], str) -> dict[str, date]
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ticker, MAX(date) FROM daily_stock_prices "
                "WHERE ticker = ANY(%s) AND date <= %s GROUP BY ticker",
                (list(tickers), end_date),
            )
            return dict(cur.fetchall())


def check_price_dates_aligned(tickers, end_date, retries=PRICE_ALIGNMENT_RETRIES,
                              interval_sec=PRICE_ALIGNMENT_RETRY_INTERVAL_SEC,
                              max_deviant=MAX_DEVIANT_TICKERS,
                              fetch=_fetch_latest_price_dates, sleep=time.sleep):
    """
    설명: 종목별 주가 최신일 정렬을 점검해 진행 가능하면 {"common_date", "deviants"}를 돌려준다.
    주의 사항: 불가하면 interval_sec마다 retries번 다시 조회하며 기다리고, 끝까지 불가하면 RuntimeError(학습 전 중단).
    입력 타입->반환 타입: (list[str], str, int, int, int, Callable, Callable) -> dict
    """
    for attempt in range(retries + 1):
        summary = summarize_price_alignment(fetch(tickers, end_date), tickers, max_deviant)
        dist = ", ".join(f"{d}: {n}종목" for d, n in summary["distribution"].items())
        if summary["ok"]:
            deviants = summary["deviants"]
            print(f"[PRICE-ALIGN] 공통 최신일 {summary['common_date']} — "
                  f"{len(tickers) - len(deviants)}/{len(tickers)}종목 일치")
            if deviants:
                listing = ", ".join(f"{t}={d}" for t, d in deviants.items())
                print(f"[PRICE-ALIGN] ⚠️ 이탈 {len(deviants)}종목(허용 {max_deviant}) — 이번 실행에서 예측을 "
                      f"저장하지 않음(학습 데이터는 그대로): {listing}")
            return {"common_date": summary["common_date"], "deviants": deviants}
        reason = ("최빈 최신일 동률" if summary["common_date"] is None
                  else f"이탈 {len(summary['deviants'])}종목 > 허용 {max_deviant}")
        print(f"[PRICE-ALIGN] 주가 최신일 불일치 (조회 {attempt + 1}/{retries + 1}, {reason}) — "
              f"분포 {{{dist}}}, 가격 없음 {len(summary['missing'])}종목 {summary['missing'][:10]}")
        if attempt < retries:
            print(f"[PRICE-ALIGN] {interval_sec}초 뒤 재조회합니다(수집 진행 중일 수 있음).")
            sleep(interval_sec)
    raise RuntimeError(
        f"[PRICE-ALIGN] 재조회 {retries}회 후에도 종목별 주가 최신일 불일치({reason}) — 학습·예측·저장을 "
        f"하지 않고 중단합니다. 분포 {{{dist}}}, 가격 없음 {summary['missing']}"
    )


def run_pooled_volatility_pipeline(tickers, start_date, end_date, train_config=None, sinks=None,
                                   checkpoint_dir=None, garch_cache_dir=None, garch_read_only=False, log_dir=None,
                                   version=None, tuning_path=None, final_refit=None):
    """
    설명: 입력 점검 → baseline·학습·게이트 → 전체 재학습 → 체크포인트 → 종목별 예측·저장 → 실행 기록을 하고 {version, gate, predictions, serving_model}을 돌려준다.
    주의 사항: 기본 sinks는 운영 DB·MLflow에 쓴다. train_config를 안 주면 채택된 튜닝 값(tuning_path, 기본 checkpoints/tuning)을, 없으면 기본값을 두 단계 모두에 쓴다.
    final_refit(None이면 환경변수 FINAL_REFIT)이 켜지면 서빙은 재학습 모델, gate_passed는 게이트 모델 결과다. 재학습이 실패하면 [REFIT-FAIL] 후 게이트 모델로 서빙한다.
    입력 타입->반환 타입: (list[str], str, str, TrainConfig|None, list|None, str|None, str|None, bool, str|None, str|None, str|None, bool|None) -> dict
    """
    if sinks is None:
        sinks = [store.DbSink(), store.MlflowSink()]
    checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
    log_dir = log_dir or LOG_DIR
    # 소수 이탈은 그 종목 저장만 빼고 진행, 다수 이탈은 학습 전에 중단.
    price_deviants = check_price_dates_aligned(tickers, end_date)["deviants"]
    # 지수 결측이 하나라도 있으면 이번 실행의 예측 전부를 input_data_suspect=True로 기록한다.
    input_data_suspect = any(n > 0 for n in check_market_index_freshness(end_date).values())
    # 거시지표·KOSPI·종목 가격은 한 번만 읽어 학습과 예측이 같은 스냅숏을 쓴다.
    shared_cache = load_shared_cache(start_date, end_date)
    prices_by_ticker = load_price_cache(tickers, start_date, end_date)
    # 튜닝 값: 직접 준 train_config가 우선. 없으면 채택된 주간 튜닝 값, 그것도 없거나 깨졌으면 기본값(실패하지 않음).
    tuned_params_id = None
    if train_config is None:
        train_config, tuned_params_id = tuning.resolve_train_config(DEFAULT_TRAIN_CONFIG, tuning_path)
    data = prepare_pooled_training_data(tickers, start_date, end_date, shared_cache=shared_cache,
                                        prices_by_ticker=prices_by_ticker, garch_cache_dir=garch_cache_dir,
                                        garch_read_only=garch_read_only)
    train_out = train_and_gate(data, len(tickers), train_config)
    feature_cols, ticker_to_id = train_out["feature_cols"], train_out["ticker_to_id"]
    device, gate = train_out["device"], train_out["gate"]
    baselines_by_ticker = train_out["baselines_by_ticker"]
    # 서빙 모델: 기본은 게이트 모델. 전체 재학습이 성공하면 재학습 모델로 바꾼다(gate_passed는 그대로 게이트 모델 결과).
    model, scaler = train_out["model"], train_out["scaler"]

    model_kwargs = _model_kwargs(len(tickers), train_out["train_config"])
    extra_meta = {
        "tickers": tickers, "ticker_to_id": ticker_to_id,
        "gate": {k: v for k, v in gate.items()},
        "n_train": train_out["n_train"], "n_val": train_out["n_val"],
        "n_train_by_ticker": train_out["n_train_by_ticker"],
        "train_start": start_date, "train_end": end_date,
        "split_train_end": str(train_out["train_end"].date()),
        "split_val_end": str(train_out["val_end"].date()),
    }
    if tuned_params_id is not None:
        extra_meta["tuned_params_id"] = tuned_params_id

    refit_summary = None
    gate_ckpt = None
    if final_refit_enabled(final_refit):
        try:
            refit = refit_on_all_data(data, len(tickers), train_out["train_config"], train_out["best_epoch"], device)
        except Exception as e:
            # 재학습은 서빙 모델을 바꾸는 부가 단계라, 실패하면 이미 게이트를 판정한 모델로 서빙한다(기록은 meta·MLflow에 남김).
            print(traceback.format_exc())
            error = redact(f"{type(e).__name__}: {e}")
            print(f"[REFIT-FAIL] ⚠️ 전체 재학습 실패 — 게이트 모델(train→val)로 서빙한다: {error}")
            refit_summary = {"serving_model": store.SERVING_GATE_MODEL, "error": error}
            extra_meta["serving_model"] = store.SERVING_GATE_MODEL
            extra_meta["final_refit"] = {"error": error}
        else:
            refit_summary = {"serving_model": store.SERVING_FINAL_REFIT, **{k: v for k, v in refit.items()
                                                                           if k not in ("model", "scaler")}}
            gate_ckpt = {"model": model, "scaler": scaler,
                         "extra_meta": {**extra_meta, "serving_model": store.SERVING_GATE_MODEL,
                                        "role": "gate_model"}}
            model, scaler = refit["model"], refit["scaler"]
            extra_meta["serving_model"] = store.SERVING_FINAL_REFIT
            extra_meta["gate_model_dir"] = store.GATE_MODEL_SUBDIR
            extra_meta["final_refit"] = {"epochs": refit["epochs"], "n_rows": refit["n_rows"],
                                         "n_by_split": refit["n_by_split"],
                                         "last_target_date": refit["last_target_date"],
                                         "scaler_fit": "train+val+test", "elapsed_sec": refit["elapsed_sec"]}
            print(f"[REFIT] 전체 재학습 완료 — {refit['n_rows']}행(train {refit['n_by_split']['train']} + "
                  f"val {refit['n_by_split']['val']} + test {refit['n_by_split']['test']}, 마지막 target "
                  f"{refit['last_target_date']}), {refit['epochs']} epoch, {refit['elapsed_sec']}초 — 서빙 모델 = 재학습 모델")

    if gate_ckpt is not None:
        # 게이트 모델은 서빙 버전 폴더 안 gate_model/에 보관한다(백테스트 재생이 out-of-sample 날짜를 이 모델로 다시 예측).
        version = version or store.new_version()
        gate_ckpt["extra_meta"]["serving_version"] = version
        store.save_checkpoint(
            gate_ckpt["model"], gate_ckpt["scaler"], feature_cols, model_kwargs, os.path.join(checkpoint_dir, version),
            version=store.GATE_MODEL_SUBDIR, model_class_name=MODEL_CLASS_NAME, extra_meta=gate_ckpt["extra_meta"],
            update_latest=False,
        )
    version = store.save_checkpoint(
        model, scaler, feature_cols, model_kwargs, checkpoint_dir, version=version,
        model_class_name=MODEL_CLASS_NAME, extra_meta=extra_meta,
    )
    ckpt = {"version": version, "checkpoint_dir": checkpoint_dir, "model": model, "scaler": scaler,
            "feature_cols": feature_cols, "model_kwargs": model_kwargs, "model_class": MODEL_CLASS_NAME,
            "extra_meta": extra_meta}
    if gate_ckpt is not None:
        ckpt["gate_model"] = gate_ckpt
    for sink in sinks:
        sink.on_checkpoint(ckpt)

    gate_str = "통과" if gate["passed"] else "미통과"
    print(f"\n📊 pooled 하이브리드 변동성 모델 학습 완료(버전 {version}, "
          f"train={train_out['n_train']}/val={train_out['n_val']}) — 배포 게이트 {gate_str}")
    print(f"   val RMSE: hybrid={gate['hybrid_rmse']:.4f}  GARCH={gate['garch_rmse']:.4f}  "
          f"SMA20={gate['sma_rmse']:.4f}  Parkinson-SMA20={gate['parkinson_rmse']:.4f}")
    print(f"   개별 판정: vs GARCH={gate['beats_garch']}  vs SMA20={gate['beats_sma']}  "
          f"vs Parkinson-SMA20={gate['beats_parkinson']}")

    if not gate["passed"]:
        failed_against = [
            name for name, beat in (
                ("GARCH", gate["beats_garch"]),
                ("SMA20", gate["beats_sma"]),
                ("Parkinson-SMA20", gate["beats_parkinson"]),
            ) if not beat
        ]
        print(f"\n⚠️⚠️⚠️ 배포 게이트 미통과 — 다음 baseline을 이기지 못함: {', '.join(failed_against)} ⚠️⚠️⚠️")
        print("⚠️ 이번 예측은 저장되지만 gate_passed=False로 표시됩니다 — 참고용으로만 사용하세요.")

    results = {}
    for ticker in tickers:
        if ticker in price_deviants:
            results[ticker] = f"제외 (가격 정렬 이탈, 최신일 {price_deviants[ticker]})"
            continue
        r = store.run_isolated(
            store.predict_and_save_for_ticker, ticker, start_date, end_date, LOOKBACK,
            model, scaler, feature_cols, ticker_to_id, device, version, gate, baselines_by_ticker, sinks,
            input_data_suspect=input_data_suspect, shared_cache=shared_cache,
            prices=prices_by_ticker[ticker], label=ticker, log_dir=log_dir,
        )
        results[ticker] = r["result"] if r["status"] == "success" else f"실패 ({r['error']})"

    n_success, n_failed, n_excluded = store.count_prediction_results(results)
    print(f"\n=== 종목별 예측 결과 ({n_success}/{len(tickers)} 성공, 실패 {n_failed}, 제외 {n_excluded}) ===")
    for ticker, r in results.items():
        print(f"  {ticker}: {r}")
    if n_failed:
        print(f"⚠️ {n_failed}개 종목 예측 실패 — 가격예측/logs/ 에서 상세 로그를 확인하세요.")

    payload = store.build_run_payload(tickers, version, train_out, gate, results, model_kwargs,
                                      input_data_suspect=input_data_suspect, tuned_params_id=tuned_params_id,
                                      refit=refit_summary)
    for sink in sinks:
        sink.log_run(payload)
    serving_model = refit_summary["serving_model"] if refit_summary else store.SERVING_GATE_MODEL
    return {"version": version, "gate": gate, "predictions": results, "serving_model": serving_model}
