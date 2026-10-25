# store.py — 저장과 예측: 체크포인트, model_registry 등록, 종목별 다음 거래일 예측과 model_predictions 저장, MLflow 실행 기록.
# 파이프라인 단계: 저장+예측. 주요 진입: save_checkpoint, predict_and_save_for_ticker, build_run_payload, Sink 3종.
# Sink는 저장 대상을 바꿔 끼우는 인터페이스다(on_checkpoint, save_prediction, log_run). 운영 기본값은 [DbSink(), MlflowSink()].
# 배경: 결정기록.md의 "예측은 게이트와 무관하게 저장한다", "model_predictions UPSERT와 target_date 규칙",
#       "MLflow 기록과 model_registry 등록은 부가 기록"

import hashlib
import json
import os
import traceback
from datetime import date, datetime, timedelta, timezone

import mlflow
import numpy as np
import pandas as pd
import requests
import torch

from config import redact
from db_manager import get_db_connection
from 가격예측.model import MODEL_CLASS_NAME, MODEL_FAMILY
from 가격예측.split_dataset import build_next_day_merged_window_volatility_hybrid

# 서버는 같은 PC의 127.0.0.1에만 바인딩된다. 환경변수로 덮어쓸 수 있다.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = "일일_자동화"
# 기록 직전 /health 확인 타임아웃(초). 응답이 없으면 기록을 건너뛴다.
MLFLOW_HEALTH_TIMEOUT_SEC = 3
# MLflow HTTP 재시도 상한(setdefault라 환경변수로 덮어쓸 수 있다). 기본값이면 서버가 없을 때 수 분을 기다린다.
MLFLOW_HTTP_LIMITS = {
    "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "3",
    "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR": "1",
    "MLFLOW_HTTP_REQUEST_TIMEOUT": "15",
}

# 서빙 모델 표시 값(체크포인트 meta serving_model, MLflow tag·param serving_model). 대시보드.백테스트_재생이 이 값을 본다.
SERVING_FINAL_REFIT = "final_refit"   # train+val+test 전체 재학습 모델
SERVING_GATE_MODEL = "gate_model"     # 게이트 판정에 쓴 train→val 모델(재학습 실패 시 서빙)
GATE_MODEL_SUBDIR = "gate_model"      # 재학습 모델 버전 폴더 안에 게이트 모델을 보관하는 하위 폴더

# model_registry 자동 등록 값. 끄려면 환경변수 MODEL_REGISTRY_AUTO=0.
REGISTRY_SOURCE = "live"
REGISTRY_GATE_DEFINITION = "tickermean"
REGISTRY_MISSING_SHOW = 10  # [REGISTRY-MISSING] 줄에 보여줄 최대 version 수


# ── 모델 체크포인트 저장 ──────────────────────────────────────────────────────────

def new_version():
    """
    설명: 체크포인트 버전 문자열(UTC 현재 시각 YYYYMMDD_HHMMSS)을 만든다.
    주의 사항: model_predictions.model_version·model_registry와 같은 형식이다(형식을 바꾸지 말 것).
    입력 타입->반환 타입: () -> str
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def save_checkpoint(model, scaler, feature_cols, model_kwargs, checkpoint_dir, version=None,
                     extra_meta=None, model_class_name=None, update_latest=True):
    """
    설명: state_dict·scaler·meta.json을 새 버전 폴더(checkpoint_dir/<version>/)에 저장하고 latest.txt를 그 버전으로 바꾼다.
    주의 사항: 파일 쓰기. version 생략 시 new_version(). 이전 버전은 덮어쓰지 않는다. update_latest=False면 latest.txt를 건드리지 않는다(게이트 모델 보관용).
    입력 타입->반환 타입: (nn.Module, FeatureScaler, list[str], dict, str, str|None, dict|None, str|None, bool) -> str
    """
    if version is None:
        version = new_version()
    version_dir = os.path.join(checkpoint_dir, version)
    os.makedirs(version_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(version_dir, "model.pt"))
    scaler.save(os.path.join(version_dir, "scaler.npz"))

    meta = {
        "version": version,
        "feature_cols": list(feature_cols),
        "model_kwargs": model_kwargs,
        "model_class": model_class_name or type(model).__name__,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(os.path.join(version_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if update_latest:
        with open(os.path.join(checkpoint_dir, "latest.txt"), "w", encoding="utf-8") as f:
            f.write(version)

    return version


def state_dict_sha256(model):
    """
    설명: state_dict의 키 이름과 텐서 바이트를 키 순서대로 이어 SHA256을 만든다.
    주의 사항: 학습 결과 동일성 비교용(CollectSink). 장치·dtype이 다르면 해시가 달라진다.
    입력 타입->반환 타입: (nn.Module) -> str
    """
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def register_model_version(version, checkpoint_dir, source=REGISTRY_SOURCE):
    """
    설명: 체크포인트 meta.json으로 model_registry에 1행을 넣고, registry에 없는 version 목록을 경고로 출력한다.
    주의 사항: DB 쓰기. 실패해도 예외를 던지지 않고 [REGISTRY-SKIP] 경고만 남긴다. 빠진 version을 자동 보충하지 않는다.
    입력 타입->반환 타입: (str, str, str) -> None
    """
    try:
        if os.environ.get("MODEL_REGISTRY_AUTO", "1") == "0":
            print(f"[REGISTRY-OFF] model_registry 자동 등록 꺼짐(MODEL_REGISTRY_AUTO=0) — version {version} 등록 안 함")
            return
        with open(os.path.join(checkpoint_dir, version, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        if meta["model_class"] != MODEL_CLASS_NAME:
            raise ValueError(f"예상하지 못한 model_class {meta['model_class']!r}")
        family = MODEL_FAMILY
        row = (version, family, meta["model_class"], "checkpoint_meta", source, REGISTRY_GATE_DEFINITION,
               meta["train_end"], meta["split_train_end"], meta["split_val_end"], meta["n_train"])
        local_versions = {d for d in os.listdir(checkpoint_dir)
                          if os.path.isfile(os.path.join(checkpoint_dir, d, "meta.json"))}
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO model_registry (model_version, model_family, model_class, family_basis, source, "
                    "gate_definition, train_end, split_train_end, split_val_end, n_train) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (model_version) DO NOTHING", row)
                inserted = cur.rowcount
                cur.execute("SELECT model_version FROM model_registry")
                registered = {r[0] for r in cur.fetchall()}
                cur.execute("SELECT DISTINCT model_version FROM model_predictions")
                predicted = {r[0] for r in cur.fetchall()}
            conn.commit()
        print(f"[registry] {version} {'신규 등록' if inserted == 1 else '이미 등록돼 있음'}({family})")
        missing = sorted((local_versions | predicted) - registered)
        if missing:
            shown = ", ".join(missing[:REGISTRY_MISSING_SHOW]) + (" …" if len(missing) > REGISTRY_MISSING_SHOW else "")
            print(f"[REGISTRY-MISSING] ⚠️ model_registry에 없는 version {len(missing)}개: {shown}")
    except Exception as e:
        # 예외 원문에 접속 정보가 섞일 수 있어 클래스명만 남기고 redact를 거친다.
        print(redact(f"[REGISTRY-SKIP] ⚠️ model_registry 등록 실패({type(e).__name__}) — 예측·저장은 계속된다. "
                     f"version {version}은 registry에 없으므로 수동 등록이 필요하다."))


# ── 무인 실행 예외 격리 ───────────────────────────────────────────────────────────

def run_isolated(fn, *args, label="", log_dir=None, **kwargs):
    """
    설명: fn을 실행하고 성공·실패를 status dict({label, status, result, error, traceback})로 돌려준다.
    주의 사항: 예외를 다시 던지지 않는다(종목별 루프 계속용). 실패하면 traceback을 출력하고 log_dir에 파일로 쓴다.
    입력 타입->반환 타입: (Callable, *args, str, str|None, **kwargs) -> dict
    """
    try:
        result = fn(*args, **kwargs)
        return {"label": label, "status": "success", "result": result, "error": None, "traceback": None}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"❌ [{label}] 실패: {e}")
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(log_dir, f"error_{label}_{ts}.log")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(tb)
            print(f"   상세 로그: {log_path}")
        return {"label": label, "status": "failed", "result": None, "error": str(e), "traceback": tb}


# ── 다음 거래일 예측 + 저장 ───────────────────────────────────────────────────────

def next_weekday(d):
    """
    설명: 다음 평일을 돌려준다(금·토 → 월).
    주의 사항: 공휴일을 모른다. 휴장일이 target_date가 될 수 있다(실측은 model_predictions_realized 뷰가 다음 거래일로 계산).
    입력 타입->반환 타입: (date) -> date
    """
    wd = d.weekday()  # Mon=0 .. Sun=6
    if wd == 4:  # 금요일 -> 월요일
        return d + timedelta(days=3)
    if wd == 5:  # 토요일 -> 월요일
        return d + timedelta(days=2)
    return d + timedelta(days=1)


def _trailing_predicted_volatility_mean(ticker, before_target_date, n=60):
    """
    설명: model_predictions에서 종목의 직전 n개 target_date 예측 변동성 평균을 구한다(조기경보 계산용).
    주의 사항: DB 읽기. 이력이 10개 미만이면 None(조기경보 False). 조기경보는 설명용이며 검증된 신호가 아니다.
    입력 타입->반환 타입: (str, date, int) -> float|None
    """
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패 — _trailing_predicted_volatility_mean")
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT predicted_volatility FROM model_predictions
                WHERE ticker = %s AND target_date < %s
                ORDER BY target_date DESC LIMIT %s
                """,
                (ticker, before_target_date, n),
            )
            rows = cur.fetchall()
    if len(rows) < 10:  # 이력이 너무 적으면 판단을 보류(과거 데이터 부족 -> 조기경보 False로 안전하게)
        return None
    return float(np.mean([r[0] for r in rows]))


def save_prediction(ticker, target_date, prediction_date, predicted_volatility,
                     garch_baseline, sma20_baseline, parkinson_sma20_baseline,
                     is_early_warning, model_version, gate, input_data_suspect=False):
    """
    설명: 예측 1행을 model_predictions에 (ticker, target_date) 기준으로 UPSERT한다.
    주의 사항: DB 쓰기. 같은 키를 다시 계산하면 예측값·게이트·suspect를 덮어쓴다(이전 값은 남지 않음, created_at은 최초 값).
    입력 타입->반환 타입: (str, date, date, float, float, float, float, bool, str, dict, bool) -> None
    """
    query = """
        INSERT INTO model_predictions
            (ticker, target_date, prediction_date, predicted_volatility,
             garch_baseline, sma20_baseline, parkinson_sma20_baseline,
             is_early_warning, model_version,
             gate_passed, gate_vs_garch, gate_vs_sma20, gate_vs_parkinson, input_data_suspect)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker, target_date) DO UPDATE
            SET prediction_date = EXCLUDED.prediction_date,
                predicted_volatility = EXCLUDED.predicted_volatility,
                garch_baseline = EXCLUDED.garch_baseline,
                sma20_baseline = EXCLUDED.sma20_baseline,
                parkinson_sma20_baseline = EXCLUDED.parkinson_sma20_baseline,
                is_early_warning = EXCLUDED.is_early_warning,
                model_version = EXCLUDED.model_version,
                gate_passed = EXCLUDED.gate_passed,
                gate_vs_garch = EXCLUDED.gate_vs_garch,
                gate_vs_sma20 = EXCLUDED.gate_vs_sma20,
                gate_vs_parkinson = EXCLUDED.gate_vs_parkinson,
                input_data_suspect = EXCLUDED.input_data_suspect
    """
    with get_db_connection() as conn:
        if not conn:
            raise RuntimeError("DB 연결 실패 — save_prediction")
        with conn.cursor() as cur:
            cur.execute(query, (
                ticker, target_date, prediction_date, float(predicted_volatility),
                float(garch_baseline), float(sma20_baseline), float(parkinson_sma20_baseline),
                bool(is_early_warning), model_version,
                bool(gate["passed"]), bool(gate["beats_garch"]), bool(gate["beats_sma"]), bool(gate["beats_parkinson"]),
                bool(input_data_suspect),
            ))
        conn.commit()


def predict_and_save_for_ticker(ticker, start_date, end_date, lookback, model, scaler, feature_cols,
                                 ticker_to_id, device, version, gate, baselines_by_ticker, sinks,
                                 input_data_suspect=False, shared_cache=None, prices=None):
    """
    설명: 한 종목의 다음 거래일 변동성을 예측하고 예측 행을 sinks에 넘긴 뒤 요약 dict를 돌려준다.
    주의 사항: 예외를 그대로 던진다(run_isolated로 감싸 부를 것). 예측값이 NaN/inf/음수면 저장하지 않고 ValueError.
    입력 타입->반환 타입: (str, str, str, int, nn.Module, FeatureScaler, list, dict, device, str, dict, dict, list, bool, dict|None, pd.DataFrame|None) -> dict
    """
    params = baselines_by_ticker[ticker]["params"]
    window_df, last_confirmed_date = build_next_day_merged_window_volatility_hybrid(
        ticker, start_date, end_date, lookback, params, **(shared_cache or {}),
        precomputed_prices=prices,
    )
    X_next = window_df[feature_cols].values[np.newaxis, :, :].astype(np.float32)
    X_next_s = scaler.transform(X_next)
    tid = np.array([ticker_to_id[ticker]], dtype=np.int64)

    model.eval()
    with torch.no_grad():
        pred = model(
            torch.from_numpy(X_next_s).to(device), torch.from_numpy(tid).to(device)
        ).cpu().numpy()[0]

    predicted_volatility = float(pred)
    # 확실한 결함(NaN/inf/음수)만 막고 상한은 두지 않는다(급변장의 큰 예측은 정상일 수 있음).
    if not np.isfinite(predicted_volatility) or predicted_volatility < 0:
        raise ValueError(
            f"[{ticker}] 예측값 비정상(NaN/inf/음수) — predicted_volatility={predicted_volatility!r}. "
            f"저장을 건너뜁니다."
        )

    prediction_date = pd.Timestamp(last_confirmed_date).date()
    target_date = next_weekday(prediction_date)

    trailing_mean = _trailing_predicted_volatility_mean(ticker, target_date)
    is_early_warning = bool(trailing_mean is not None and predicted_volatility > 1.5 * trailing_mean)

    next_day = baselines_by_ticker[ticker]["next_day"]
    row = dict(
        ticker=ticker, target_date=target_date, prediction_date=prediction_date,
        predicted_volatility=predicted_volatility, garch_baseline=next_day["garch"],
        sma20_baseline=next_day["sma20"], parkinson_sma20_baseline=next_day["parkinson"],
        is_early_warning=is_early_warning, model_version=version, gate=gate,
        input_data_suspect=input_data_suspect,
    )
    for sink in sinks:
        sink.save_prediction(row)

    return {
        "predicted_volatility": predicted_volatility,
        "garch_baseline": next_day["garch"], "sma20_baseline": next_day["sma20"],
        "parkinson_sma20_baseline": next_day["parkinson"],
        "is_early_warning": is_early_warning,
        "prediction_date": str(prediction_date), "target_date": str(target_date),
    }


def count_prediction_results(results):
    """
    설명: 종목별 결과를 (성공, 실패, 제외) 수로 센다. 성공=dict, 실패="실패 ..." 문자열, 제외="제외 ..." 문자열.
    주의 사항: 결과 문자열의 접두어("실패"·"제외")로 판정하므로 오케스트레이터의 문구와 맞아야 한다.
    입력 타입->반환 타입: (dict) -> (int, int, int)
    """
    n_success = sum(1 for v in results.values() if isinstance(v, dict))
    n_failed = sum(1 for v in results.values() if isinstance(v, str) and v.startswith("실패"))
    n_excluded = sum(1 for v in results.values() if isinstance(v, str) and v.startswith("제외"))
    return n_success, n_failed, n_excluded


# ── 실행 기록(MLflow) ─────────────────────────────────────────────────────────────

def build_run_payload(tickers, version, train_out, gate, prediction_results, model_kwargs, input_data_suspect=False,
                      tuned_params_id=None, refit=None):
    """
    설명: MLflow에 남길 run 이름·태그·파라미터·손실 수열·지표를 dict로 만든다(run 이름 "pd<prediction_date>_v<version>").
    주의 사항: 태그·파라미터·지표 이름은 바꾸지 말 것. n_input_suspect는 0 또는 성공 종목 수. 채택된 튜닝 값으로 돈 실행만 param tuned_params_id를 더한다.
    refit(전체 재학습 결과 요약)이 있으면 서빙 모델 표시(tag·param serving_model)와 재학습 기록을 더한다. 없으면(FINAL_REFIT=0) 예전과 같다.
    입력 타입->반환 타입: (list[str], str, dict, dict, dict, dict, bool, str|None, dict|None) -> dict
    """
    cfg = train_out["train_config"]
    pred_dates = sorted({v["prediction_date"] for v in prediction_results.values() if isinstance(v, dict)})
    pred_date_tag = ",".join(pred_dates) if pred_dates else "none"
    n_success, n_failed, n_excluded = count_prediction_results(prediction_results)
    payload = {
        "run_name": f"pd{pred_date_tag}_v{version}",
        "tags": {
            "n_tickers": len(tickers), "version": version,
            "model_family": MODEL_FAMILY,
            "prediction_date": pred_date_tag,
            "run_date": date.today().isoformat(),
            "input_data_suspect": str(bool(input_data_suspect)).lower(),
        },
        "params": {
            "MODEL_ARCHITECTURE": MODEL_FAMILY,
            "model_class": MODEL_CLASS_NAME,
            **{f"model_{k}": v for k, v in model_kwargs.items()},
            "BATCH_SIZE": cfg.batch_size, "LR": cfg.lr, "WEIGHT_DECAY": cfg.weight_decay,
            "SMOKE_EPOCHS": cfg.smoke_epochs, "MAX_EPOCHS": cfg.max_epochs, "PATIENCE": cfg.patience,
            "SEED": cfg.seed,
        },
        "steps": [(name, loss, epoch) for epoch, train_loss, val_loss in train_out["history"]
                  for name, loss in (("train_loss", train_loss), ("val_loss", val_loss))],
        "metrics": [
            {
                "best_val_loss": train_out["best_val_loss"],
                "overfit_ratio_at_end": train_out["overfit_ratio_at_end"],
                "hybrid_rmse_tickermean": gate["hybrid_rmse"],
                "garch_rmse": gate["garch_rmse"],
                "sma_rmse": gate["sma_rmse"],
                "parkinson_rmse": gate["parkinson_rmse"],
                "gate_passed": int(gate["passed"]),
            },
            {
                "n_predictions_success": n_success, "n_predictions_failed": n_failed,
                "n_predictions_excluded": n_excluded,
                "n_input_suspect": n_success if input_data_suspect else 0,
            },
        ],
    }
    if tuned_params_id is not None:
        payload["params"]["tuned_params_id"] = tuned_params_id
    if refit is not None:
        payload["tags"]["serving_model"] = refit["serving_model"]
        payload["params"]["serving_model"] = refit["serving_model"]
        if refit["serving_model"] == SERVING_FINAL_REFIT:
            payload["params"]["refit_epochs"] = refit["epochs"]
            payload["params"]["refit_n_rows"] = refit["n_rows"]
            payload["params"]["refit_last_target_date"] = refit["last_target_date"]
            payload["steps"] += [("refit_train_loss", loss, epoch) for epoch, loss in refit["history"]]
            payload["metrics"].append({"refit_elapsed_sec": refit["elapsed_sec"],
                                       "refit_final_train_loss": refit["history"][-1][1]})
        else:
            payload["params"]["refit_error"] = refit["error"][:250]
    return payload


def _mlflow_health_problem(uri):
    """
    설명: MLflow 서버 /health를 짧은 타임아웃으로 확인해 정상이면 None, 아니면 원인 요약(예외 클래스명·HTTP 상태)을 돌려준다.
    주의 사항: 네트워크 요청. URL·예외 원문은 돌려주지 않는다.
    입력 타입->반환 타입: (str) -> str|None
    """
    try:
        r = requests.get(uri.rstrip("/") + "/health", timeout=MLFLOW_HEALTH_TIMEOUT_SEC)
    except requests.RequestException as e:
        return type(e).__name__
    return None if r.status_code == 200 else f"HTTP {r.status_code}"


# ── Sink ──────────────────────────────────────────────────────────────────────

class Sink:
    """
    설명: 저장 대상 인터페이스(on_checkpoint, save_prediction, log_run). 기본 구현은 아무것도 하지 않는다.
    주의 사항: 오케스트레이터가 sinks 목록을 돌며 같은 이름의 메서드를 부른다(메서드 이름을 바꾸지 말 것).
    입력 타입->반환 타입: () -> Sink
    """

    def on_checkpoint(self, ckpt):
        """
        설명: 체크포인트 저장 직후 호출된다(기본: 아무것도 안 함).
        주의 사항: 없음.
        입력 타입->반환 타입: (dict) -> None
        """
        pass

    def save_prediction(self, row):
        """
        설명: 종목 1개 예측 행을 받는다(기본: 아무것도 안 함).
        주의 사항: 없음.
        입력 타입->반환 타입: (dict) -> None
        """
        pass

    def log_run(self, payload):
        """
        설명: 실행 기록 payload를 받는다(기본: 아무것도 안 함).
        주의 사항: 없음.
        입력 타입->반환 타입: (dict) -> None
        """
        pass


class DbSink(Sink):
    """
    설명: 운영 DB에 쓴다(체크포인트 직후 model_registry 등록, 예측 행 model_predictions UPSERT).
    주의 사항: DB 쓰기. 백테스트 등 운영 외 실행에서는 쓰지 말 것.
    입력 타입->반환 타입: (str) -> DbSink
    """

    def __init__(self, registry_source=REGISTRY_SOURCE):
        """
        설명: model_registry.source에 넣을 값을 받는다.
        주의 사항: 운영 기본값은 'live'.
        입력 타입->반환 타입: (str) -> None
        """
        self.registry_source = registry_source

    def on_checkpoint(self, ckpt):
        """
        설명: 체크포인트 version을 model_registry에 등록한다.
        주의 사항: DB 쓰기(실패해도 예외 없이 경고만).
        입력 타입->반환 타입: (dict) -> None
        """
        register_model_version(ckpt["version"], ckpt["checkpoint_dir"], source=self.registry_source)

    def save_prediction(self, row):
        """
        설명: 예측 행을 model_predictions에 UPSERT한다.
        주의 사항: DB 쓰기.
        입력 타입->반환 타입: (dict) -> None
        """
        save_prediction(**row)


class MlflowSink(Sink):
    """
    설명: MLflow run을 기록한다(게이트 통과 여부와 무관하게 항상).
    주의 사항: 네트워크 쓰기. 서버가 없거나 실패해도 파이프라인을 멈추지 않는다([MLFLOW-SKIP] 또는 경고 출력).
    입력 타입->반환 타입: (str, str|None) -> MlflowSink
    """

    def __init__(self, experiment=MLFLOW_EXPERIMENT_NAME, tracking_uri=None):
        """
        설명: 실험 이름과 tracking URI를 받는다.
        주의 사항: 실험은 이름으로 찾는다(운영 "일일_자동화").
        입력 타입->반환 타입: (str, str|None) -> None
        """
        self.experiment = experiment
        self.tracking_uri = tracking_uri or MLFLOW_TRACKING_URI

    def log_run(self, payload):
        """
        설명: /health를 확인한 뒤 payload의 run 이름·태그·파라미터·손실 수열·지표를 MLflow에 기록한다.
        주의 사항: 네트워크 쓰기. HTTP 재시도 상한 환경변수를 setdefault한다. 모든 예외를 잡아 경고만 출력한다.
        입력 타입->반환 타입: (dict) -> None
        """
        try:
            for k, v in MLFLOW_HTTP_LIMITS.items():
                os.environ.setdefault(k, v)
            if self.tracking_uri.startswith("http"):
                problem = _mlflow_health_problem(self.tracking_uri)
                if problem:
                    print(f"[MLFLOW-SKIP] ⚠️ MLflow 서버 응답 없음 — 기록 건너뜀({problem}). "
                          f"예측 저장은 끝났고, 이번 실행의 MLflow run은 남지 않는다.")
                    return
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment)
            with mlflow.start_run(run_name=payload["run_name"]):
                mlflow.set_tags(payload["tags"])
                mlflow.log_params(payload["params"])
                for name, value, step in payload["steps"]:
                    mlflow.log_metric(name, value, step=step)
                for metrics in payload["metrics"]:
                    mlflow.log_metrics(metrics)
        except Exception as e:
            print(f"⚠️ MLflow 로깅 실패(파이프라인은 계속 진행됩니다) — {e}")


class CollectSink(Sink):
    """
    설명: 아무 데도 쓰지 않고 체크포인트 요약·예측 행·MLflow 기록을 메모리에 모은다(점검·비교용).
    주의 사항: 값은 DB·MLflow에 들어가는 형태로 바꿔 담는다(태그·파라미터 문자열, 지표 float, 날짜 문자열).
    입력 타입->반환 타입: (str) -> CollectSink
    """

    def __init__(self, experiment=MLFLOW_EXPERIMENT_NAME):
        """
        설명: 빈 수집 상태(predictions, registry, checkpoint, mlflow)로 만든다.
        주의 사항: 없음.
        입력 타입->반환 타입: (str) -> None
        """
        self.experiment = experiment
        self.predictions, self.registry = [], []
        self.checkpoint = None
        self.gate_checkpoint = None
        self.mlflow = None

    def on_checkpoint(self, ckpt):
        """
        설명: version과 서빙 체크포인트 요약(state_dict 해시, scaler, 피처, model_kwargs, model_class, extra_meta)을 담는다.
        주의 사항: 전체 재학습으로 게이트 모델을 따로 보관했으면 그 요약(state_hash·scaler)을 gate_checkpoint에 담는다.
        입력 타입->반환 타입: (dict) -> None
        """
        self.registry.append(ckpt["version"])
        self.checkpoint = {
            "state_hash": state_dict_sha256(ckpt["model"]),
            "scaler_mean": np.asarray(ckpt["scaler"].mean_).copy(),
            "scaler_std": np.asarray(ckpt["scaler"].std_).copy(),
            "feature_cols": list(ckpt["feature_cols"]), "model_kwargs": dict(ckpt["model_kwargs"]),
            "model_class": ckpt["model_class"], "extra_meta": dict(ckpt["extra_meta"]),
        }
        gate = ckpt.get("gate_model")
        if gate is not None:
            self.gate_checkpoint = {
                "state_hash": state_dict_sha256(gate["model"]),
                "scaler_mean": np.asarray(gate["scaler"].mean_).copy(),
                "scaler_std": np.asarray(gate["scaler"].std_).copy(),
                "extra_meta": dict(gate["extra_meta"]),
            }

    def save_prediction(self, row):
        """
        설명: 예측 행을 model_predictions 열 형태의 dict로 바꿔 predictions에 더한다.
        주의 사항: 없음.
        입력 타입->반환 타입: (dict) -> None
        """
        g = row["gate"]
        self.predictions.append(dict(
            ticker=row["ticker"], target_date=str(row["target_date"]), prediction_date=str(row["prediction_date"]),
            predicted_volatility=float(row["predicted_volatility"]), garch_baseline=float(row["garch_baseline"]),
            sma20_baseline=float(row["sma20_baseline"]), parkinson_sma20_baseline=float(row["parkinson_sma20_baseline"]),
            is_early_warning=bool(row["is_early_warning"]), model_version=row["model_version"],
            gate_passed=bool(g["passed"]), gate_vs_garch=bool(g["beats_garch"]), gate_vs_sma20=bool(g["beats_sma"]),
            gate_vs_parkinson=bool(g["beats_parkinson"]), input_data_suspect=bool(row["input_data_suspect"])))

    def log_run(self, payload):
        """
        설명: payload를 MLflow에 기록될 형태(실험, run 이름, 태그, 파라미터, 지표, 손실 수열)로 바꿔 mlflow에 담는다.
        주의 사항: 없음.
        입력 타입->반환 타입: (dict) -> None
        """
        metrics = {}
        for m in payload["metrics"]:
            metrics.update({k: float(v) for k, v in m.items()})
        self.mlflow = {
            "experiment": self.experiment, "run_name": payload["run_name"],
            "tags": {k: str(v) for k, v in payload["tags"].items()},
            "params": {k: str(v) for k, v in payload["params"].items()},
            "metrics": metrics, "steps": [(k, float(v), s) for k, v, s in payload["steps"]],
        }
