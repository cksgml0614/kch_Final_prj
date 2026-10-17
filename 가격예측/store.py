# store.py — 저장(+예측) 단계: 체크포인트, model_registry, model_predictions, 종목별 다음 거래일 예측, MLflow 기록.
# 2026-10-17 정리에서 train_common.py(체크포인트·예외 격리)와 가격예측_변동성_공통.py(registry·예측·저장·MLflow)에서
# 옮겼다. SQL·출력 문구·MLflow 이름(실험·run·태그·파라미터·지표)은 그대로다.
#
# Sink: 저장 대상을 바꿔 끼우는 최소 인터페이스(이 모듈과 오케스트레이터에서만 쓴다).
#   on_checkpoint(ckpt)   체크포인트를 저장한 직후(DbSink: model_registry 등록)
#   save_prediction(row)  종목 1개 예측 행(DbSink: model_predictions UPSERT)
#   log_run(payload)      실행 기록(MlflowSink: MLflow run)
# 운영 기본값은 [DbSink(), MlflowSink()]. CollectSink는 아무 데도 쓰지 않고 메모리에 모은다(정리 전후 비교·점검용).

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

# MLflow 일일 자동화 로깅 (2026-09-14 신규). 자동화 스크립트는 서버와 같은 컴퓨터에서
# 돌기 때문에 기본값은 127.0.0.1이다. 2026-10-11부터 서버를 127.0.0.1에만 바인딩하므로
# (mlflow_server_start.bat) 다른 기기에서는 UI에 접속할 수 없다 — 원격 조회가 필요하면 인증을
# 갖춘 터널을 따로 검토한다. 환경변수로 덮어쓸 수 있게 해 서버 위치가 바뀌어도 코드 수정 없이
# 대응 가능하게 함.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = "일일_자동화"
# 2026-10-11: 로깅 직전 서버 확인 타임아웃(초). 서버가 없거나 멈췄으면 기록을 건너뛴다(MlflowSink.log_run).
MLFLOW_HEALTH_TIMEOUT_SEC = 3
# MLflow HTTP 재시도 상한(setdefault — 사람이 환경변수로 덮어쓸 수 있다). MLflow 기본값(재시도 7, 대기 계수 2,
# 타임아웃 120초)이면 서버가 없을 때 약 4분 24초(2026-10-11 실측), 멈췄을 때 약 20분(추정)을 기다린다.
MLFLOW_HTTP_LIMITS = {
    "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "3",
    "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR": "1",
    "MLFLOW_HTTP_REQUEST_TIMEOUT": "15",
}

# 2026-10-11: 체크포인트를 저장한 직후 그 version을 model_registry에 등록한다(register_model_version).
# 일일 경로 기본값은 source='live'. 끄려면 환경변수 MODEL_REGISTRY_AUTO=0.
REGISTRY_SOURCE = "live"
REGISTRY_GATE_DEFINITION = "tickermean"
REGISTRY_MISSING_SHOW = 10  # [REGISTRY-MISSING] 줄에 보여줄 최대 version 수


# ── 모델 체크포인트 저장 (2026-09-06) ─────────────────────────────────────────────
# 버전 디렉터리(UTC 날짜+시각)를 매번 새로 만들어 이전 버전을 덮어쓰지 않는다(롤백 가능) — latest.txt로
# 최신 버전을 가리킨다(심볼릭 링크는 Windows 권한 문제가 있을 수 있어 피함).

def save_checkpoint(model, scaler, feature_cols, model_kwargs, checkpoint_dir, version=None,
                     extra_meta=None, model_class_name=None):
    """model.state_dict() + FeatureScaler + 피처 목록 + 모델 하이퍼파라미터 + 메타데이터를
    버전 디렉터리(checkpoint_dir/<version>/)에 저장한다. version 생략 시 UTC 날짜+시각
    (YYYYMMDD_HHMMSS)으로 자동 태깅 — 같은 날 여러 번 실행해도 초 단위까지 있어 겹치지 않는다.
    model_class_name 생략 시 model의 실제 클래스 이름을 자동으로 기록한다.
    반환: 저장된 version 문자열."""
    if version is None:
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
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

    with open(os.path.join(checkpoint_dir, "latest.txt"), "w", encoding="utf-8") as f:
        f.write(version)

    return version


def state_dict_sha256(model):
    """state_dict 내용 해시(키 정렬, 키 이름 + 텐서 바이트). CollectSink가 학습 결과 동일성 비교에 쓴다."""
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def register_model_version(version, checkpoint_dir, source=REGISTRY_SOURCE):
    """체크포인트 meta.json으로 model_registry에 1행을 INSERT ... ON CONFLICT DO NOTHING 한다(2026-10-11).
    이어서 로컬 체크포인트 version과 model_predictions의 version 중 registry에 없는 것을 찾아 경고한다
    (자동 보충 등록은 하지 않는다 — 시험 실행 체크포인트를 'live'로 잘못 넣을 수 있어 사람이 보고 넣는다).
    부가 기록이라 실패해도 예외를 던지지 않는다. 줄 맨 앞 [REGISTRY-SKIP]을 붙인 경고만 남기고 돌아간다.
    경고에는 예외 클래스명만 넣고(원문에 접속 정보가 섞일 수 있음), 출력 전에 redact를 거친다."""
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
        print(redact(f"[REGISTRY-SKIP] ⚠️ model_registry 등록 실패({type(e).__name__}) — 예측·저장은 계속된다. "
                     f"version {version}은 registry에 없으므로 수동 등록이 필요하다."))


# ── 무인 실행 예외 격리 (2026-09-06) ─────────────────────────────────────────────
# 예외를 print만 하고 삼키지 않는다. 종목 단위로 격리해 하나가 실패해도 나머지가 계속 진행되게 하되,
# 실패는 콘솔 + 로그 파일에 traceback까지 남긴다.

def run_isolated(fn, *args, label="", log_dir=None, **kwargs):
    """fn(*args, **kwargs)을 실행하고 성공/실패를 status dict로 반환한다. 예외가 나면
    traceback을 콘솔에 출력하고(log_dir이 주어지면 파일로도 저장) 예외를 다시 던지지
    않는다 — 호출부(오케스트레이션 스크립트의 종목별 루프)가 다음 항목을 계속 처리할 수 있다.

    반환: {"label", "status"("success"/"failed"), "result", "error", "traceback"}"""
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
    """옛 방향예측 파이프라인에서 쓰던 근사를 값 변경 없이 그대로 가져옴 — 실제 거래소 캘린더가 없어 공휴일은
    반영 못 한다. target_date가 실제 거래일과 어긋나면 실측은 model_predictions_realized 뷰가 prediction_date
    다음 첫 거래일로 계산한다(CLAUDE.md "휴장일 target_date")."""
    wd = d.weekday()  # Mon=0 .. Sun=6
    if wd == 4:  # 금요일 -> 월요일
        return d + timedelta(days=3)
    if wd == 5:  # 토요일 -> 월요일
        return d + timedelta(days=2)
    return d + timedelta(days=1)


def _trailing_predicted_volatility_mean(ticker, before_target_date, n=60):
    """model_predictions에서 이 종목의 직전 n개 target_date(< before_target_date)
    predicted_volatility 평균을 가져온다. is_early_warning 계산용.

    ⚠️ 이 규칙(예측σ > 직전60일 평균의 1.5배) 자체의 조기경보 "능력"은 pre-cleanup 태그 시점의 결과_변동성_조기경보_
    검증.md 원 검증이 GARCH 라벨링 버그로 무효화된 뒤 재검증에서 우연 수준으로 확인됐다
    (CLAUDE.md 참고). 여기서 채우는 is_early_warning은 감사/기록용 descriptive 값일 뿐,
    검증된 운영 신호로 취급하면 안 된다."""
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
    """model_predictions에 UPSERT((ticker, target_date) 기준) — 재실행 시 최신 값으로 덮어씀.
    input_data_suspect(2026-09-27): 입력 데이터 오염 의심 플래그 — 재실행으로 예측값이 바뀌면
    플래그도 그 실행 기준으로 함께 덮어쓴다(깨끗한 재실행 후에도 true가 남지 않도록)."""
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
    """pooled 모델로 한 종목의 다음 거래일 변동성을 예측하고 sinks에 예측 행을 넘긴다(DbSink면 model_predictions
    저장). run_isolated로 감싸 호출하는 것을 전제로 예외를 그대로 던진다(격리 책임은 호출부에 있음).

    shared_cache(선택, 2026-10-05): load_shared_cache() 결과. None이면 종목마다 거시지표·KOSPI를 다시 조회한다.
    prices(선택, 2026-10-05): load_price_cache(...)[ticker]. None이면 가격을 직접 조회한다."""
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
    # sanity check(2026-09-14, 사람 결정): NaN/inf/음수만 차단 — 변동성은 음수가 될 수 없는
    # 값이라 확실한 결함 신호지만, 상한선은 일부러 두지 않는다. 실제 급변장에서는 크게 튀는
    # 예측이 정상일 수 있어 임의 상한이 오히려 정상 예측을 막을 위험이 있다는 판단.
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


def _count_prediction_results(results):
    """종목별 결과를 (성공, 실패, 제외) 수로 센다. 성공 = dict, 실패 = "실패 ..." 문자열,
    제외 = "제외 ..." 문자열(가격 정렬 이탈로 예측을 저장하지 않은 종목, 2026-10-17)."""
    n_success = sum(1 for v in results.values() if isinstance(v, dict))
    n_failed = sum(1 for v in results.values() if isinstance(v, str) and v.startswith("실패"))
    n_excluded = sum(1 for v in results.values() if isinstance(v, str) and v.startswith("제외"))
    return n_success, n_failed, n_excluded


# ── 실행 기록(MLflow) ─────────────────────────────────────────────────────────────

def build_run_payload(tickers, version, train_out, gate, prediction_results, model_kwargs, input_data_suspect=False):
    """MLflow에 남길 실행 기록을 dict로 만든다(2026-10-17, 옛 _log_mlflow_run의 내용 부분). 이름은 그대로다:
    - run 이름 "pd<prediction_date>_v<체크포인트 버전>", 실행일은 run_date 태그(2026-10-11 정정).
    - params: 모델 생성자 인자(model_* — 체크포인트 meta와 같은 출처)와 MODEL_ARCHITECTURE(= MODEL_FAMILY),
      학습 하이퍼파라미터는 train_out["train_config"](TrainConfig)에서 가져오되 이름(BATCH_SIZE 등)은 그대로.
    - tag model_family = MODEL_FAMILY('lstm'), model_registry.model_family와 같은 어휘.
    - 하이브리드 게이트 RMSE는 hybrid_rmse_tickermean으로만 기록한다(옛 hybrid_rmse는 2026-09-28부터 정의 변경).
    - input_data_suspect는 실행 단위 플래그라 n_input_suspect는 0 또는 저장 성공 종목 수다."""
    cfg = train_out["train_config"]
    pred_dates = sorted({v["prediction_date"] for v in prediction_results.values() if isinstance(v, dict)})
    pred_date_tag = ",".join(pred_dates) if pred_dates else "none"
    n_success, n_failed, n_excluded = _count_prediction_results(prediction_results)
    return {
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


def _mlflow_health_problem(uri):
    """MLflow 서버 /health를 MLFLOW_HEALTH_TIMEOUT_SEC 안에 확인한다(2026-10-11). 정상(HTTP 200)이면 None,
    아니면 원인 요약(예외 클래스명 또는 HTTP 상태)을 돌려준다. URL·예외 원문은 돌려주지 않는다."""
    try:
        r = requests.get(uri.rstrip("/") + "/health", timeout=MLFLOW_HEALTH_TIMEOUT_SEC)
    except requests.RequestException as e:
        return type(e).__name__
    return None if r.status_code == 200 else f"HTTP {r.status_code}"


# ── Sink ──────────────────────────────────────────────────────────────────────

class Sink:
    """저장 대상 인터페이스. 기본 구현은 아무것도 하지 않는다."""

    def on_checkpoint(self, ckpt):
        pass

    def save_prediction(self, row):
        pass

    def log_run(self, payload):
        pass


class DbSink(Sink):
    """운영 DB: model_registry 등록(체크포인트 직후)과 model_predictions UPSERT."""

    def __init__(self, registry_source=REGISTRY_SOURCE):
        self.registry_source = registry_source

    def on_checkpoint(self, ckpt):
        register_model_version(ckpt["version"], ckpt["checkpoint_dir"], source=self.registry_source)

    def save_prediction(self, row):
        save_prediction(**row)


class MlflowSink(Sink):
    """MLflow run 기록. 게이트 통과 여부와 무관하게 항상 기록한다(2026-09-14 — MLflow는 "무슨 일이 있었는지 전부
    보는" 관측 도구). 전체를 try/except로 감싸 MLflow가 꺼져 있어도 파이프라인 본체는 멈추지 않는다.
    2026-10-11 지연 상한: 시작할 때 /health를 3초 안에 확인하고, 응답이 없으면 줄 맨 앞에 [MLFLOW-SKIP]을 붙인
    경고를 출력한 뒤 기록 없이 돌아간다. 확인을 통과한 뒤 실패해도 MLFLOW_HTTP_LIMITS로 재시도를 줄여 상한을 둔다."""

    def __init__(self, experiment=MLFLOW_EXPERIMENT_NAME, tracking_uri=None):
        self.experiment = experiment
        self.tracking_uri = tracking_uri or MLFLOW_TRACKING_URI

    def log_run(self, payload):
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
    """아무 데도 쓰지 않고 메모리에 모은다(정리 전후 비교·점검용). 값은 MLflow·DB에 들어가는 형태로 정규화한다
    (태그·파라미터는 문자열, 지표는 float, 날짜는 문자열)."""

    def __init__(self, experiment=MLFLOW_EXPERIMENT_NAME):
        self.experiment = experiment
        self.predictions, self.registry = [], []
        self.checkpoint = None
        self.mlflow = None

    def on_checkpoint(self, ckpt):
        self.registry.append(ckpt["version"])
        self.checkpoint = {
            "state_hash": state_dict_sha256(ckpt["model"]),
            "scaler_mean": np.asarray(ckpt["scaler"].mean_).copy(),
            "scaler_std": np.asarray(ckpt["scaler"].std_).copy(),
            "feature_cols": list(ckpt["feature_cols"]), "model_kwargs": dict(ckpt["model_kwargs"]),
            "model_class": ckpt["model_class"], "extra_meta": dict(ckpt["extra_meta"]),
        }

    def save_prediction(self, row):
        g = row["gate"]
        self.predictions.append(dict(
            ticker=row["ticker"], target_date=str(row["target_date"]), prediction_date=str(row["prediction_date"]),
            predicted_volatility=float(row["predicted_volatility"]), garch_baseline=float(row["garch_baseline"]),
            sma20_baseline=float(row["sma20_baseline"]), parkinson_sma20_baseline=float(row["parkinson_sma20_baseline"]),
            is_early_warning=bool(row["is_early_warning"]), model_version=row["model_version"],
            gate_passed=bool(g["passed"]), gate_vs_garch=bool(g["beats_garch"]), gate_vs_sma20=bool(g["beats_sma"]),
            gate_vs_parkinson=bool(g["beats_parkinson"]), input_data_suspect=bool(row["input_data_suspect"])))

    def log_run(self, payload):
        metrics = {}
        for m in payload["metrics"]:
            metrics.update({k: float(v) for k, v in m.items()})
        self.mlflow = {
            "experiment": self.experiment, "run_name": payload["run_name"],
            "tags": {k: str(v) for k, v in payload["tags"].items()},
            "params": {k: str(v) for k, v in payload["params"].items()},
            "metrics": metrics, "steps": [(k, float(v), s) for k, v, s in payload["steps"]],
        }
