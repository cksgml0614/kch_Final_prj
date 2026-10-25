# 대시보드/백테스트_재생.py — 현재 체크포인트로 최근 N거래일을 다시 예측해 로컬 파일에만 저장한다(2026-10-18, 수동 실행용).
# 실행(프로젝트 루트에서): python -m 대시보드.백테스트_재생 [--version 20261017_200618] [--days 15]
# 출력: 대시보드/cache/backtest_replay.parquet, backtest_replay.meta.json (cache/.gitignore가 커밋을 막는다).
# 규칙:
# - DB에는 쓰지 않는다. model_predictions는 (ticker, target_date) UPSERT라 재생값을 쓰면 운영 행을 덮어쓴다.
#   이 프로세스의 DB 연결은 모두 read_only로 열고(get_db_connection 교체), 끝에 rollback한다.
# - 학습하지 않는다. 체크포인트(model.pt·scaler.npz·meta.json)와 운영 GARCH 캐시는 읽기만 한다(garch_read_only).
# - 피처·baseline을 새로 구현하지 않는다. 서빙 창은 split_dataset.build_next_day_merged_window_volatility_hybrid,
#   예측은 store.predict_and_save_for_ticker와 같은 순서(창 → float32 → scaler.transform → model), baseline 3종은
#   baselines.compute_ticker_baselines의 next_day 값(운영이 저장하는 값과 같은 계산)을 쓴다.
# - 실측은 model_predictions_realized 뷰의 SQL을 (ticker, prediction_date) 목록에 그대로 적용해 DB에서 계산한다.
# - prediction_date마다 가격은 그날까지로 자른다. 지표 캐시는 published_date < 거래일 asof 조인이라 이후 공표분이
#   붙지 않는다(누수 확인은 검증 스크립트가 날짜를 잘라 다시 불러 비교한다).
# - 대상 날짜는 체크포인트 split_val_end보다 뒤(out-of-sample)인 날짜만 쓴다.
# - 서빙 체크포인트가 전체 재학습 모델(meta serving_model=final_refit)이면 그 모델은 대상 날짜를 이미 학습했다. 그래서 같은 버전
#   폴더의 gate_model/(train→val 게이트 모델)로 재생한다. gate_model/이 없으면 in-sample 재생을 하지 않고 멈춘다.

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import db_manager  # noqa: E402

_open_connection = db_manager.get_db_connection


def get_read_only_connection():
    """
    설명: db_manager로 연결을 열고 read_only로 바꿔 돌려준다(이 프로세스의 모든 DB 접근이 이 함수를 거친다).
    주의 사항: 아래 가격예측 모듈 import보다 먼저 db_manager.get_db_connection을 바꿔야 그 모듈들에도 적용된다.
    입력 타입->반환 타입: () -> psycopg.Connection
    """
    conn = _open_connection()
    conn.read_only = True
    return conn


db_manager.get_db_connection = get_read_only_connection

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from constants import STOCK_INITIAL_LOAD_START  # noqa: E402
from 가격예측.baselines import GARCH_PARAM_CACHE_DIR, compute_ticker_baselines  # noqa: E402
from 가격예측.model import PooledLSTMRegressor  # noqa: E402
from 가격예측.sequence_dataset import FeatureScaler  # noqa: E402
from 가격예측.split_dataset import build_next_day_merged_window_volatility_hybrid  # noqa: E402
from 가격예측.store import GATE_MODEL_SUBDIR, SERVING_FINAL_REFIT, SERVING_GATE_MODEL, next_weekday  # noqa: E402
from 가격예측.가격예측_변동성_공통 import CHECKPOINT_DIR, LOOKBACK, load_shared_cache  # noqa: E402
from 시장지표.feature_loader import load_price_cache  # noqa: E402

KST = timezone(timedelta(hours=9))
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
OUT_PARQUET = os.path.join(CACHE_DIR, "backtest_replay.parquet")
OUT_META = os.path.join(CACHE_DIR, "backtest_replay.meta.json")
DEFAULT_DAYS = 15
MIN_TICKERS_FOR_TRADING_DAY = 50  # 이만큼 종목 행이 있는 날을 거래일로 본다

# model_predictions_realized 뷰(Database/가격예측/뷰_생성.sql)의 실측 정의를 그대로 옮긴 것. p(ticker, prediction_date)만 바꿨다.
REALIZED_SQL = """
SELECT p.ticker, p.prediction_date,
       nxt.date AS realized_date,
       CASE WHEN nxt.date IS NOT NULL AND prev.close > 0 AND nxt.close > 0
            THEN abs(ln(nxt.close::float8 / prev.close::float8)) * 100 END AS realized_volatility,
       CASE WHEN nxt.date IS NOT NULL THEN 'confirmed' ELSE 'pending' END AS realized_status,
       (nxt.volume = 0) AS target_halted
FROM unnest(%s::varchar[], %s::date[]) AS p(ticker, prediction_date)
LEFT JOIN LATERAL (
    SELECT d.date, d.close, d.volume FROM daily_stock_prices d
    WHERE d.ticker = p.ticker AND d.date > p.prediction_date
    ORDER BY d.date LIMIT 1) nxt ON true
LEFT JOIN LATERAL (
    SELECT d.close FROM daily_stock_prices d
    WHERE d.ticker = p.ticker AND d.date < nxt.date
    ORDER BY d.date DESC LIMIT 1) prev ON true
"""


def load_checkpoint(checkpoint_root, version, device):
    """
    설명: 체크포인트 폴더에서 재생에 쓸 모델·scaler·meta를 읽어 평가 모드 모델과 함께 돌려준다(replay_model: 실제로 읽은 모델).
    주의 사항: 파일 읽기만. version이 None이면 latest.txt를 따른다. 전체 재학습 서빙 모델이면 gate_model/을 읽고, 없으면 FileNotFoundError.
    입력 타입->반환 타입: (str, str|None, torch.device) -> dict
    """
    if version is None:
        with open(os.path.join(checkpoint_root, "latest.txt"), encoding="utf-8") as f:
            version = f.read().strip()
    vdir = os.path.join(checkpoint_root, version)
    with open(os.path.join(vdir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    replay_model = meta.get("serving_model", SERVING_GATE_MODEL)
    if replay_model == SERVING_FINAL_REFIT:
        vdir = os.path.join(vdir, meta.get("gate_model_dir", GATE_MODEL_SUBDIR))
        if not os.path.isfile(os.path.join(vdir, "meta.json")):
            raise FileNotFoundError(f"{version}: 전체 재학습 모델인데 게이트 모델({vdir})이 없다 — in-sample 재생을 하지 않는다")
        with open(os.path.join(vdir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        replay_model = SERVING_GATE_MODEL
        print(f"서빙 모델은 전체 재학습 모델이라 게이트 모델({vdir})로 재생한다")
    model = PooledLSTMRegressor(n_features=len(meta["feature_cols"]), **meta["model_kwargs"]).to(device)
    model.load_state_dict(torch.load(os.path.join(vdir, "model.pt"), map_location=device, weights_only=True))
    model.eval()
    scaler = FeatureScaler.load(os.path.join(vdir, "scaler.npz"))
    return {"version": version, "meta": meta, "model": model, "scaler": scaler, "replay_model": replay_model}


def slice_prices(prices, start_date, end_date):
    """
    설명: 가격 캐시 한 종목분을 end_date까지로 자르고 get_features가 확인하는 구간 표시(attrs span)를 맞춘다.
    주의 사항: 원본을 바꾸지 않는다(복사본).
    입력 타입->반환 타입: (pd.DataFrame, str, str) -> pd.DataFrame
    """
    out = prices[prices["date"] <= pd.Timestamp(end_date)].copy()
    out.attrs["span"] = (str(pd.Timestamp(start_date).date()), str(pd.Timestamp(end_date).date()))
    return out


def predict_one(ticker, start_date, prediction_date, ckpt, shared_cache, prices, garch_cache_dir, device):
    """
    설명: 한 종목·한 prediction_date의 예측값과 baseline 3종을 운영 서빙 경로로 계산한다.
    주의 사항: prices는 prediction_date까지 잘린 캐시여야 한다. 마지막 확정일이 prediction_date와 다르면 ValueError.
    입력 타입->반환 타입: (str, str, str, dict, dict, pd.DataFrame, str, torch.device) -> dict
    """
    meta = ckpt["meta"]
    b = compute_ticker_baselines(
        ticker, start_date, prediction_date, pd.Timestamp(meta["split_train_end"]), pd.Timestamp(meta["split_val_end"]),
        prices=prices, garch_cache_dir=garch_cache_dir, garch_read_only=True,
    )
    window_df, last_confirmed = build_next_day_merged_window_volatility_hybrid(
        ticker, start_date, prediction_date, LOOKBACK, b["params"], **shared_cache, precomputed_prices=prices,
    )
    if pd.Timestamp(last_confirmed).date() != pd.Timestamp(prediction_date).date():
        raise ValueError(f"{ticker}: 마지막 확정일 {pd.Timestamp(last_confirmed).date()} != prediction_date {prediction_date}")
    X_next = window_df[meta["feature_cols"]].values[np.newaxis, :, :].astype(np.float32)
    X_next_s = ckpt["scaler"].transform(X_next)
    tid = np.array([meta["ticker_to_id"][ticker]], dtype=np.int64)
    with torch.no_grad():
        pred = ckpt["model"](torch.from_numpy(X_next_s).to(device), torch.from_numpy(tid).to(device)).cpu().numpy()[0]
    nd = b["next_day"]
    return {"predicted_volatility": float(pred), "garch_baseline": nd["garch"], "sma20_baseline": nd["sma20"],
            "parkinson_sma20_baseline": nd["parkinson"]}


def replay_dates(prediction_dates, ckpt, shared_cache, prices_by_ticker, start_date, garch_cache_dir, device,
                 log=print):
    """
    설명: prediction_date 목록 × 체크포인트 종목 전부를 예측해 행 목록과 실패 목록을 돌려준다.
    주의 사항: 종목 단위로 격리한다(실패는 failures에 남기고 요약에 출력). DB 쓰기 없음.
    입력 타입->반환 타입: (list[str], dict, dict, dict, str, str, torch.device, callable) -> (list[dict], list[dict])
    """
    rows, failures = [], []
    for pdate in prediction_dates:
        t0 = time.time()
        for ticker in ckpt["meta"]["tickers"]:
            try:
                prices = slice_prices(prices_by_ticker[ticker], start_date, pdate)
                r = predict_one(ticker, start_date, pdate, ckpt, shared_cache, prices, garch_cache_dir, device)
            except Exception as e:  # 종목 단위 격리 — 원인은 failures와 요약에 남긴다
                failures.append({"ticker": ticker, "prediction_date": pdate, "error": f"{type(e).__name__}: {e}"})
                continue
            pd_date = pd.Timestamp(pdate).date()
            rows.append({"ticker": ticker, "prediction_date": pd_date, "target_date": next_weekday(pd_date), **r})
        log(f"  prediction_date {pdate}: {sum(1 for x in rows if str(x['prediction_date']) == pdate)}행 "
            f"({time.time() - t0:.0f}초)")
    return rows, failures


def fetch_realized(tickers, prediction_dates):
    """
    설명: (ticker, prediction_date) 목록에 model_predictions_realized 뷰와 같은 SQL로 실측을 계산해 DataFrame으로 돌려준다.
    주의 사항: DB 읽기(read_only, 끝에 rollback).
    입력 타입->반환 타입: (list[str], list[date]) -> pd.DataFrame
    """
    conn = db_manager.get_db_connection()
    try:
        cur = conn.execute(REALIZED_SQL, (list(tickers), list(prediction_dates)))
        cols = [d.name for d in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=cols)
        conn.rollback()
    finally:
        conn.close()
    return df


def trading_days(prices_by_ticker):
    """
    설명: 가격 캐시에서 MIN_TICKERS_FOR_TRADING_DAY개 이상 종목에 행이 있는 날짜를 오름차순으로 돌려준다.
    주의 사항: 없음.
    입력 타입->반환 타입: (dict) -> list[pd.Timestamp]
    """
    counts = pd.concat([df["date"] for df in prices_by_ticker.values() if not df.empty]).value_counts()
    return sorted(counts[counts >= MIN_TICKERS_FOR_TRADING_DAY].index)


def git_commit():
    """
    설명: 이 파일이 있는 저장소의 HEAD 커밋 해시(작업 트리에 변경이 있으면 '+dirty')를 돌려준다.
    주의 사항: git이 없으면 'unknown'.
    입력 타입->반환 타입: () -> str
    """
    try:
        h = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", "."], capture_output=True, text=True).stdout.strip()
        return h + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def main():
    """
    설명: 대상 날짜를 정하고 재생·실측 계산 뒤 parquet·meta.json을 대시보드/cache/에 쓴다.
    주의 사항: 쓰기는 대시보드/cache/ 안뿐. 실패 종목이 있으면 요약에 출력하고 exit 1.
    입력 타입->반환 타입: () -> None
    """
    ap = argparse.ArgumentParser(description="현재 체크포인트로 최근 거래일 백테스트 재생(로컬 파일 저장, DB 쓰기 없음)")
    ap.add_argument("--version", default=None, help="체크포인트 버전(기본: latest.txt)")
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS, help="실측이 확정된 최근 평가일 수(기본 15)")
    ap.add_argument("--checkpoint-root", default=os.environ.get("REPLAY_CHECKPOINT_ROOT", CHECKPOINT_DIR))
    ap.add_argument("--garch-cache-dir", default=os.environ.get("REPLAY_GARCH_CACHE_DIR", GARCH_PARAM_CACHE_DIR))
    args = ap.parse_args()

    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint_root, args.version, device)
    meta = ckpt["meta"]
    tickers = meta["tickers"]
    start_date = STOCK_INITIAL_LOAD_START
    val_end = pd.Timestamp(meta["split_val_end"])

    latest_price = None
    conn = db_manager.get_db_connection()
    try:
        latest_price = conn.execute("SELECT max(date) FROM daily_stock_prices WHERE ticker = ANY(%s)",
                                    (list(tickers),)).fetchone()[0]
        conn.rollback()
    finally:
        conn.close()
    end_date = str(latest_price)
    print(f"체크포인트 {ckpt['version']} (split_val_end {meta['split_val_end']}), 데이터 마지막 확정일 {end_date}, 장치 {device}")

    shared_cache = load_shared_cache(start_date, end_date)
    prices_by_ticker = load_price_cache(tickers, start_date, end_date)
    days = trading_days(prices_by_ticker)
    eval_days = days[-args.days:]
    # 평가일 E의 prediction_date = 그 직전 거래일. 마지막 거래일도 넣는다(실측 대기 1일, 재현 확인용).
    pdates = [days[days.index(e) - 1] for e in eval_days] + [days[-1]]
    oos = [d for d in pdates if d > val_end]
    dropped = [str(d.date()) for d in pdates if d <= val_end]
    if dropped:
        print(f"⚠️ split_val_end 이전 날짜는 제외: {dropped}")
    pdates = [str(d.date()) for d in oos]
    print(f"대상 prediction_date {len(pdates)}개: {pdates[0]} ~ {pdates[-1]}")

    rows, failures = replay_dates(pdates, ckpt, shared_cache, prices_by_ticker, start_date, args.garch_cache_dir, device)
    df = pd.DataFrame(rows)
    realized = fetch_realized(df["ticker"].tolist(), df["prediction_date"].tolist())
    df = df.merge(realized, on=["ticker", "prediction_date"], how="left", validate="one_to_one")
    df["model_version"] = ckpt["version"]

    garch_fitted = []
    for t in tickers:
        p = os.path.join(args.garch_cache_dir, f"{t}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                garch_fitted.append(json.load(f).get("fitted_at"))
    confirmed = df[df["realized_status"] == "confirmed"]
    out_meta = {
        "generated_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_version": ckpt["version"],
        "replay_model": ckpt["replay_model"],
        "split_train_end": meta["split_train_end"], "split_val_end": meta["split_val_end"],
        "prediction_dates": pdates,
        "eval_period": [str(confirmed["realized_date"].min()), str(confirmed["realized_date"].max())] if len(confirmed) else None,
        "n_eval_days": int(confirmed["realized_date"].nunique()),
        "data_last_confirmed_date": end_date,
        "code_commit": git_commit(),
        "garch_cache_fitted_at": [min(garch_fitted), max(garch_fitted)] if garch_fitted else None,
        "device": str(device),
        "n_rows": int(len(df)), "n_failed": len(failures), "failures": failures[:20],
        "dropped_before_val_end": dropped,
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    ignore = os.path.join(CACHE_DIR, ".gitignore")
    if not os.path.exists(ignore):
        with open(ignore, "w", encoding="utf-8") as f:
            f.write("*\n")
    df.to_parquet(OUT_PARQUET, index=False)
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(out_meta, f, ensure_ascii=False, indent=1)
    print(f"저장: {OUT_PARQUET} ({len(df)}행), 평가일 {out_meta['n_eval_days']}개 {out_meta['eval_period']}, "
          f"{out_meta['elapsed_sec']}초")
    if failures:
        print(f"⚠️ 실패 {len(failures)}건(종목·날짜): " + "; ".join(f"{x['ticker']} {x['prediction_date']} {x['error']}"
                                                         for x in failures[:10]))
        sys.exit(1)


if __name__ == "__main__":
    main()
