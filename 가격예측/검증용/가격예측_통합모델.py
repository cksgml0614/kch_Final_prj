# 가격예측_통합모델.py
# Task T 다종목 pooled **방향** 예측 모델(2026-09-06) — 8종목 cross-sectional pooling +
# 종목 임베딩.
#
# ⚠️⚠️ 2026-09-06 폐기 확정: Task T는 방향(상승/하락) 예측을 완전히 접고 변동성 예측으로
# 전환했다 — 단일종목·pooled 방향 예측 전부 random walk/동전던지기/셔플 검증에 실패했다.
# 상세 사유·수치는 `결과_TaskT_방향예측_폐기.md` 참고. **이 파일은 더 이상 운영 파이프라인이
# 아니며 재실행 대상도 아니다** — 다만 pooled_dataset.build_pooled_sequences/
# train_common.train_pooled_transformer 등 하이브리드 변동성 모델이 그대로 재사용 중인
# 공용 인프라의 하이퍼파라미터 상수(D_MODEL 등)를 이 파일에서 계속 참조하는 다른 스크립트가
# 있어(가격예측_변동성_일일수집.py 이전의 실험 스크립트들), 삭제하지 않고 남겨뒀다 — 완전
# 정리 여부는 CLAUDE.md/사람 결정 대기(파일 정리 목록 참고).
#
# 원래 가격예측_공통.py(단일 종목 방향 파이프라인)를 참조했으나 그 파일이 쓰래기통/으로
# 이동돼(2026-09-06) next_weekday()/save_prediction()을 아래에 인라인 복사했다 — 이 두
# 함수 자체는 값 변경 없이 그대로 옮긴 것이다. save_prediction()이 쓰는 model_predictions
# 스키마는 변동성 예측용으로 이미 교체돼(predicted_return 컬럼 자체가 없어짐) 이 함수를
# 실제로 호출하면 실패한다 — 정상이다, 이 파일은 재실행 대상이 아니기 때문이다.

import argparse
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch

from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
from db_manager import get_db_connection
from 가격예측.pooled_dataset import build_pooled_sequences
from 가격예측.sequence_dataset import FeatureScaler
from 가격예측.split_dataset import build_next_day_merged_window
from 가격예측.train_common import (
    evaluate_pooled_predictions,
    passes_deployment_gate,
    run_isolated,
    save_checkpoint,
    train_pooled_transformer,
)


def next_weekday(d):
    """가격예측_공통.py(폐기, 쓰래기통 이동)에서 인라인 복사 — 값 변경 없음."""
    wd = d.weekday()
    if wd == 4:
        return d + timedelta(days=3)
    if wd == 5:
        return d + timedelta(days=2)
    return d + timedelta(days=1)


def save_prediction(ticker, prediction_date, target_date, predicted_return, model_version, gate):
    """가격예측_공통.py(폐기, 쓰래기통 이동)에서 인라인 복사 — 값 변경 없음. ⚠️ 구 스키마
    (predicted_return 기반)를 참조하므로 현재 model_predictions(변동성 예측용으로 교체됨)
    에는 실행 시 실패한다 — 이 파일 자체가 재실행 대상이 아니므로 의도적으로 고치지 않았다."""
    query = """
        INSERT INTO model_predictions
            (ticker, target_date, prediction_date, predicted_return, model_version,
             gate_passed, gate_precision, gate_recall)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker, target_date) DO UPDATE
            SET prediction_date = EXCLUDED.prediction_date,
                predicted_return = EXCLUDED.predicted_return,
                model_version = EXCLUDED.model_version,
                gate_passed = EXCLUDED.gate_passed,
                gate_precision = EXCLUDED.gate_precision,
                gate_recall = EXCLUDED.gate_recall
    """
    gate_precision = None if np.isnan(gate["precision"]) else float(gate["precision"])
    gate_recall = None if np.isnan(gate["recall"]) else float(gate["recall"])
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패 — save_prediction")
        with conn.cursor() as cur:
            cur.execute(query, (
                ticker, target_date, prediction_date, float(predicted_return), model_version,
                bool(gate["passed"]), gate_precision, gate_recall,
            ))
        conn.commit()

LOOKBACK = 20
EMBEDDING_DIM = 8  # 종목 수(8개) 대비 과하지 않은 크기로 선택 — 너무 크면 종목별 과적합 위험

# 단일 종목 파이프라인(가격예측_공통.py)과 동일한 TASK_T 확정 하이퍼파라미터를 그대로 쓴다 —
# pooled 실험의 목적이 "구조 자체의 성능 개선"이 아니라 "cross-sectional pooling의 효과 확인"
# 이므로, 비교 대상을 최대한 통제하기 위해 인코더 하이퍼파라미터는 손대지 않았다.
D_MODEL = 32
NHEAD = 2
NUM_LAYERS = 2
DIM_FEEDFORWARD = 64
DROPOUT = 0.3
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
SMOKE_EPOCHS = 2
MAX_EPOCHS = 60
PATIENCE = 8
SEED = 42

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "pooled")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")


def _model_kwargs(num_stocks):
    return dict(
        lookback=LOOKBACK, num_stocks=num_stocks, embedding_dim=EMBEDDING_DIM,
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT,
    )


def train_pooled_model(tickers, start_date, end_date, seed=SEED):
    """8종목(또는 지정된 종목들)을 하나의 pooled 데이터셋으로 묶어 학습한다.
    반환: model, scaler, feature_cols, ticker_to_id, device, gate, (train_end, val_end), n_train/n_val."""
    splits, feature_cols, ticker_to_id, (train_end, val_end) = build_pooled_sequences(
        tickers, start_date, end_date, lookback=LOOKBACK
    )
    X_train, y_train, tid_train, d_train, _ = splits["train"]
    X_val, y_val, tid_val, d_val, _ = splits["val"]

    scaler = FeatureScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = train_pooled_transformer(
        X_train_s, y_train, tid_train, X_val_s, y_val, tid_val,
        len(feature_cols), LOOKBACK, len(tickers), EMBEDDING_DIM,
        D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT,
        BATCH_SIZE, LR, WEIGHT_DECAY, SMOKE_EPOCHS, MAX_EPOCHS, PATIENCE,
        device, seed=seed, label="pooled",
    )

    val_preds, val_actuals, val_tids = evaluate_pooled_predictions(result["model"], result["val_loader"], device)
    # 전체 풀 기준 배포 게이트(요구사항 6의 "종목별 + 전체 평균" 보고와는 별개 — 여기서는
    # 체크포인트에 태깅할 단일 게이트 판정만 계산한다. 종목별 세부 검증은 검증 스크립트가 담당).
    gate = passes_deployment_gate(val_preds, val_actuals)

    return {
        "model": result["model"], "scaler": scaler, "feature_cols": feature_cols,
        "ticker_to_id": ticker_to_id, "device": device, "gate": gate,
        "train_end": train_end, "val_end": val_end,
        "n_train": len(X_train), "n_val": len(X_val),
        "val_preds": val_preds, "val_actuals": val_actuals, "val_tids": val_tids,
    }


def predict_and_save_for_ticker(ticker, start_date, end_date, model, scaler, feature_cols,
                                 ticker_to_id, device, version, gate):
    """pooled 모델로 한 종목의 다음 거래일을 예측하고 model_predictions에 저장한다.
    run_isolated로 감싸 호출하는 것을 전제로 예외를 그대로 던진다."""
    window_df, last_confirmed_date = build_next_day_merged_window(ticker, start_date, end_date, LOOKBACK)
    X_next = window_df[feature_cols].values[np.newaxis, :, :].astype(np.float32)
    X_next_s = scaler.transform(X_next)
    tid = np.array([ticker_to_id[ticker]], dtype=np.int64)

    model.eval()
    with torch.no_grad():
        pred = model(
            torch.from_numpy(X_next_s).to(device), torch.from_numpy(tid).to(device)
        ).cpu().numpy()[0]

    prediction_date = pd.Timestamp(last_confirmed_date).date()
    target_date = next_weekday(prediction_date)
    save_prediction(ticker, prediction_date, target_date, float(pred), version, gate)

    return {
        "predicted_return": float(pred),
        "prediction_date": str(prediction_date),
        "target_date": str(target_date),
    }


def run_pooled_pipeline(tickers, start_date, end_date):
    """전체 pooled 파이프라인: 학습(8종목 통합) -> 체크포인트 저장(checkpoints/pooled/) ->
    종목별 다음 거래일 예측(run_isolated로 종목별 격리) -> model_predictions 저장.

    학습 자체는 격리하지 않는다 — 모델이 하나라 학습이 실패하면 전체가 실패하는 게 맞다
    (단일 종목 파이프라인처럼 "한 종목만 실패, 나머지는 계속"이 구조적으로 불가능함).
    예측 단계만 종목별로 run_isolated로 감싼다 — 한 종목의 피처 조회 실패가 다른 종목의
    예측까지 막지 않게 하기 위함."""
    train_out = train_pooled_model(tickers, start_date, end_date)
    model, scaler, feature_cols = train_out["model"], train_out["scaler"], train_out["feature_cols"]
    ticker_to_id, device, gate = train_out["ticker_to_id"], train_out["device"], train_out["gate"]

    version = save_checkpoint(
        model, scaler, feature_cols, _model_kwargs(len(tickers)), CHECKPOINT_DIR,
        model_class_name="PooledTransformerRegressor",
        extra_meta={
            "tickers": tickers, "ticker_to_id": ticker_to_id,
            "gate_passed": gate["passed"],
            "gate_precision": None if np.isnan(gate["precision"]) else gate["precision"],
            "gate_recall": None if np.isnan(gate["recall"]) else gate["recall"],
            "n_train": train_out["n_train"], "n_val": train_out["n_val"],
            "train_start": start_date, "train_end": end_date,
            "split_train_end": str(train_out["train_end"].date()),
            "split_val_end": str(train_out["val_end"].date()),
        },
    )

    results = {}
    for ticker in tickers:
        r = run_isolated(
            predict_and_save_for_ticker, ticker, start_date, end_date,
            model, scaler, feature_cols, ticker_to_id, device, version, gate,
            label=ticker, log_dir=LOG_DIR,
        )
        results[ticker] = r["result"] if r["status"] == "success" else f"실패 ({r['error']})"

    gate_str = "통과" if gate["passed"] else "미통과"
    print(f"\n✅ pooled 모델 학습 완료(버전 {version}, train={train_out['n_train']}/val={train_out['n_val']}) "
          f"— 배포 게이트 {gate_str}")
    print("\n=== 종목별 예측 결과 ===")
    for ticker, r in results.items():
        print(f"  {ticker}: {r}")

    return {"version": version, "gate_passed": gate["passed"], "predictions": results}


def main():
    parser = argparse.ArgumentParser(
        description="Task T 다종목 pooled 모델 — 8종목 통합 학습 + 종목별 예측 저장"
    )
    parser.add_argument("--tickers", nargs="+", default=None,
                         help="예: --tickers 005930 035420 (생략 시 constants.ACTIVE_TICKERS 전체)")
    parser.add_argument("--start", type=str, default=STOCK_INITIAL_LOAD_START,
                         help="학습 시작일 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None,
                         help="학습에 사용할 마지막 날짜 YYYY-MM-DD (생략 시 오늘)")
    args = parser.parse_args()

    tickers = args.tickers if args.tickers else ACTIVE_TICKERS
    end_date = args.end if args.end else date.today().isoformat()

    if len(tickers) < 2:
        print("⚠️ pooled 모델은 2종목 이상이 전제입니다 — 단일 종목은 가격예측_일일수집.py를 쓰세요.")
        return

    run_pooled_pipeline(tickers, args.start, end_date)


if __name__ == "__main__":
    main()
