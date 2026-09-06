# 가격예측_공통.py
# Task T 자동화 파이프라인 — 데이터 로드~피처 생성~학습~예측 공용 로직(2026-09-06 설계 확정).
#
# 확정된 설계 결정:
#   - V2(비율 기반 피처: close_return/hl_range_ratio/open_gap_ratio/log1p(volume)/국고채
#     스프레드) 채택. V1(레벨 기반, dataset_builder.build_base_dataset)은 완전 폐기 —
#     diagnose_baseline.py 등 Task T-1 진단 스크립트의 역사적 재현을 위해 코드 자체는
#     남겨뒀지만 이 파일은 build_base_dataset_v2 계열만 쓴다.
#   - 스트레스 구간 분리 폐기. split_dataset.split_full_period()로 전체 기간을 그대로
#     70/15/15 분할한다(뉴스 트랙 D-3와 동일 논리 — 특정 구간 하드코딩 예외는 다종목 확장과
#     충돌). STRESS_PERIOD_START/END 상수와 analysis/detect_stress_period.py는 Task T-1
#     진단 기록으로만 보존하며 이 파일은 참조하지 않는다.
#   - V2 단일 test 평가는 생략 — 정확도 검증 개념 대신, 파이프라인에 넣을 백테스트 기능으로
#     대체하기로 함(이번 단계에서는 미구현). split_full_period()가 반환하는 test 조각은
#     당장 이 파일에서 쓰지 않고 향후 백테스트 기능을 위해 남겨둔다.
#
# 재학습 주기: "매일 전체 재학습"(트리거 로직 없음). 근거는 run_daily_pipeline() 주석 참고.

import os
from datetime import timedelta

import numpy as np
import pandas as pd
import torch

from db_manager import get_db_connection
from 가격예측.sequence_dataset import FeatureScaler, build_sequences, split_sequences_by_date
from 가격예측.split_dataset import build_merged_dataset_v2, build_next_day_merged_window, split_full_period
from 가격예측.train_common import (
    evaluate_predictions,
    passes_deployment_gate,
    save_checkpoint,
    train_transformer,
)

LOOKBACK = 20

# TASK_T 확정 하이퍼파라미터(model.py 주석 근거: "encoder 1~2layer, d_model 32~64, head 2~4,
# dropout 0.2~0.3") — train_v2.py가 채택했던 값 그대로 이어받는다. 진단 체인에서 시험한
# 축소판(d16/L1) 등은 "우연한 baseline 돌파"로 판정났으므로(seed_stability_check.py) 채택하지 않는다.
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

CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints")


def checkpoint_dir_for(ticker):
    return os.path.join(CHECKPOINT_ROOT, ticker)


def _model_kwargs():
    return dict(
        lookback=LOOKBACK, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT,
    )


def build_dataset(ticker, start_date, end_date):
    """V2 피처 + 전체 기간 70/15/15 분할(스트레스 구간 분리 없음).
    반환: merged, feature_cols, (train_df, val_df, test_df) — test_df는 이 파일에서 쓰지
    않는다(위 파일 docstring 참고, 향후 백테스트 기능용으로 보존)."""
    merged, meta = build_merged_dataset_v2(ticker, start_date, end_date)
    feature_cols = [c for c in merged.columns if c != "target"]
    train_df, val_df, test_df = split_full_period(merged)
    return merged, feature_cols, (train_df, val_df, test_df)


def _build_train_val_sequences(merged, feature_cols, train_df, val_df, lookback=LOOKBACK):
    """merged 전체(연속 구간)로 시퀀스를 만든 뒤 train/val 경계로 자른다. train_df 마지막
    (lookback-1)개는 val 시퀀스의 히스토리로 재사용된다 — 표준 시계열 관행(sequence_dataset.py
    문서 참고, 이미 확정된 과거 정보 재사용이라 누수가 아님)."""
    X_all, y_all, dates_all = build_sequences(merged, feature_cols, lookback)
    splits = split_sequences_by_date(X_all, y_all, dates_all, train_df.index.max(), val_df.index.max())
    return splits["train"], splits["val"]


def train_daily_model(ticker, start_date, end_date, seed=SEED):
    """오늘자 학습 1회 실행 — V2 피처, 전체 기간에서 train으로 학습, val로 early stopping +
    배포 게이트 판정. 반환 dict: model, scaler, feature_cols, device, gate, val_preds/actuals."""
    merged, feature_cols, (train_df, val_df, test_df) = build_dataset(ticker, start_date, end_date)
    (X_train, y_train, d_train), (X_val, y_val, d_val) = _build_train_val_sequences(
        merged, feature_cols, train_df, val_df
    )

    scaler = FeatureScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = train_transformer(
        X_train_s, y_train, X_val_s, y_val, len(feature_cols), LOOKBACK,
        D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT,
        BATCH_SIZE, LR, WEIGHT_DECAY, SMOKE_EPOCHS, MAX_EPOCHS, PATIENCE,
        device, seed=seed, label=ticker,
    )

    val_preds, val_actuals = evaluate_predictions(result["model"], result["val_loader"], device)
    gate = passes_deployment_gate(val_preds, val_actuals)

    return {
        "ticker": ticker, "model": result["model"], "scaler": scaler,
        "feature_cols": feature_cols, "device": device,
        "val_preds": val_preds, "val_actuals": val_actuals, "gate": gate,
        "train_result": result, "n_train": len(X_train), "n_val": len(X_val),
    }


def next_weekday(d):
    """실제 거래소 캘린더가 없어 채택한 단순 근사 — 공휴일은 반영하지 못한다. 이 근사가 틀려
    target_date가 실제 거래일과 어긋나는 경우, actual_return 백필 시 그냥 매치되는 실제
    거래일이 없어 NULL로 남는다(조용히 틀린 값이 채워지는 게 아니라 비어있는 형태로 자명하게
    드러나므로, 잘못된 값을 저장하는 것보다 안전한 실패 모드로 판단해 채택)."""
    wd = d.weekday()  # Mon=0 .. Sun=6
    if wd == 4:  # 금요일 -> 월요일
        return d + timedelta(days=3)
    if wd == 5:  # 토요일 -> 월요일
        return d + timedelta(days=2)
    return d + timedelta(days=1)


def predict_next_day(model, scaler, feature_cols, device, ticker, start_date, end_date, lookback=LOOKBACK):
    """가장 최근 lookback 피처 시퀀스로 다음 거래일 수익률을 예측한다.
    반환: (predicted_return: float, prediction_date: date, target_date: date)"""
    window_df, last_confirmed_date = build_next_day_merged_window(ticker, start_date, end_date, lookback)
    X_next = window_df[feature_cols].values[np.newaxis, :, :].astype(np.float32)
    X_next_s = scaler.transform(X_next)

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X_next_s).to(device)).cpu().numpy()[0]

    prediction_date = pd.Timestamp(last_confirmed_date).date()
    target_date = next_weekday(prediction_date)
    return float(pred), prediction_date, target_date


def save_prediction(ticker, prediction_date, target_date, predicted_return, model_version, gate):
    """model_predictions에 UPSERT((ticker, target_date) 기준) — 같은 target_date에 대해
    재실행(예: 수동 재시도)하면 최신 값으로 덮어쓴다. 배포 게이트 미통과여도 예측은 그대로
    저장하고 gate_passed로만 표시한다(2026-09-06 확정) — 예측을 완전히 막으면 무인 운영 중
    하루치가 비어버리는 트레이드오프를 피하기 위함. T-1 결론(가격+거시지표는 이미 대체로
    무작위 수준)을 고려하면 게이트 통과 여부가 "이 날은 위험/안전"을 가르는 게 아니라
    "이 학습 실행이 최소 품질 기준을 만족했는가"를 기록하는 감사 정보에 가깝다 — 다운스트림이
    필요하면 WHERE gate_passed로 걸러 쓰면 된다."""
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


def run_daily_pipeline(ticker, start_date, end_date):
    """하루치 파이프라인 전체: 전체 재학습 -> 체크포인트 저장(버전 태깅) -> 배포 게이트 판정
    -> 다음 거래일 예측 -> model_predictions 저장. run_isolated()로 감싸 호출하는 것을
    전제로 예외를 그대로 던진다(여기서 삼키지 않음 — 격리 책임은 호출부에 있음).

    재학습 주기 = "매일 전체 재학습", 트리거 로직 없음(2026-09-06 확정). 근거: 이 모델은
    파라미터 1~2만개 수준의 극소형 Transformer라 학습 자체가 초~수십 초 단위로 끝난다.
    "성능 저하 감지 시에만 재학습" 같은 조건부 트리거는 (a) "저하"를 무엇으로 정의할지부터
    논쟁적이고 (b) 그 판단 로직 자체가 새로운 버그 지점이 되는데, 아낄 수 있는 비용은
    사실상 없다(재학습이 이미 거의 무료 수준) — 최소 복잡도를 택하는 쪽이 합리적이라고 판단했다.
    """
    train_out = train_daily_model(ticker, start_date, end_date)
    model, scaler, feature_cols, device = (
        train_out["model"], train_out["scaler"], train_out["feature_cols"], train_out["device"]
    )
    gate = train_out["gate"]

    version = save_checkpoint(
        model, scaler, feature_cols, _model_kwargs(), checkpoint_dir_for(ticker),
        extra_meta={
            "ticker": ticker,
            "gate_passed": gate["passed"],
            "gate_precision": None if np.isnan(gate["precision"]) else gate["precision"],
            "gate_recall": None if np.isnan(gate["recall"]) else gate["recall"],
            "n_train": train_out["n_train"], "n_val": train_out["n_val"],
            "train_start": start_date, "train_end": end_date,
        },
    )

    predicted_return, prediction_date, target_date = predict_next_day(
        model, scaler, feature_cols, device, ticker, start_date, end_date
    )
    save_prediction(ticker, prediction_date, target_date, predicted_return, version, gate)

    gate_str = "통과" if gate["passed"] else "미통과"
    precision_str = "nan" if np.isnan(gate["precision"]) else f"{gate['precision']:.2%}"
    recall_str = "nan" if np.isnan(gate["recall"]) else f"{gate['recall']:.2%}"
    print(
        f"✅ [{ticker}] 학습 완료(버전 {version}, train={train_out['n_train']}/val={train_out['n_val']}) "
        f"— 배포 게이트 {gate_str}(precision={precision_str} recall={recall_str}) | "
        f"{prediction_date} 기준 -> {target_date} 예측수익률={predicted_return:+.4%}"
    )

    return {
        "ticker": ticker, "model_version": version, "gate_passed": gate["passed"],
        "prediction_date": str(prediction_date), "target_date": str(target_date),
        "predicted_return": predicted_return,
    }
