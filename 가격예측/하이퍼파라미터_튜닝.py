# 하이퍼파라미터_튜닝.py — 주 1회 하이퍼파라미터 탐색·채점·채택과, 일일 학습이 읽는 튜닝 결과 파일(best_params.json) 관리.
# 파이프라인 단계: 튜닝(시퀀스와 학습 사이, 주 1회). 주요 진입: run_tuning(주간_튜닝.py), resolve_train_config(일일 학습).
# 채점: 일일 학습과 같은 train으로 후보당 1회 학습 → 검증 구간을 시간순 3조각으로 나눈 종목평균 RMSE의 평균. 테스트 구간은 기록만.
# 배경: 결정기록.md의 "주간 튜닝: 검증 3조각 채점과 채택 규칙", "튜닝 값이 없거나 깨지면 일일 학습은 기본값"

import dataclasses
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone

import mlflow
import numpy as np
import torch

from 가격예측.evaluate import evaluate_gate, evaluate_pooled_predictions, tickermean_rmse
from 가격예측.model import PooledLSTMRegressor
from 가격예측.train_common import make_pooled_loader, train_model

KST = timezone(timedelta(hours=9))
_HERE = os.path.dirname(os.path.abspath(__file__))
TUNING_DIR = os.path.join(_HERE, "checkpoints", "tuning")
BEST_PARAMS_FILE = "best_params.json"
HISTORY_DIR = "history"
HISTORY_KEEP = 8
MLFLOW_TUNING_EXPERIMENT = "주간_튜닝"

N_TRIALS = 12          # 0번 = 기본 설정 재채점, 1~11번 = 시드 고정 무작위 후보
SEARCH_SEED = 20261018
N_VAL_CHUNKS = 3
# 탐색 범위: 운영 기본값(hidden 32, 층 2, dropout 0.3, lr 1e-3, wd 1e-4) 주변의 좁은 범위.
# 정수는 목록에서 고르고, dropout은 균등, lr·wd는 로그 균등(유효숫자 2자리로 반올림).
SEARCH_SPACE = {
    "hidden_size": (24, 32, 48),
    "num_layers": (1, 2, 3),
    "dropout": (0.2, 0.4),
    "lr": (5e-4, 2e-3),
    "weight_decay": (3e-5, 3e-4),
}
TUNED_FIELDS = tuple(SEARCH_SPACE)


# ── 결과 파일 읽기(일일 학습이 부른다 — 어떤 경우에도 예외를 밖으로 내지 않는다) ──────────────────

def default_best_params_path():
    """
    설명: 운영 튜닝 결과 파일 경로(가격예측/checkpoints/tuning/best_params.json)를 돌려준다.
    주의 사항: 파일이 있는지는 보지 않는다.
    입력 타입->반환 타입: () -> str
    """
    return os.path.join(TUNING_DIR, BEST_PARAMS_FILE)


def _validated_params(params):
    """
    설명: 튜닝 파일의 params가 탐색 대상 5개를 빠짐없이, 올바른 타입·범위로 가졌는지 검사해 그대로 돌려준다.
    주의 사항: 어긋나면 ValueError. 정수 칸에 bool·실수가 오면 거부한다.
    입력 타입->반환 타입: (dict) -> dict
    """
    if not isinstance(params, dict) or set(params) != set(TUNED_FIELDS):
        raise ValueError(f"params 키가 {list(TUNED_FIELDS)}와 다름: {params!r}")
    for k in ("hidden_size", "num_layers"):
        if type(params[k]) is not int or params[k] < 1:
            raise ValueError(f"{k}는 1 이상 정수여야 함: {params[k]!r}")
    for k in ("dropout", "lr", "weight_decay"):
        if isinstance(params[k], bool) or not isinstance(params[k], (int, float)) or not np.isfinite(params[k]):
            raise ValueError(f"{k}는 유한한 수여야 함: {params[k]!r}")
    if not (0 <= params["dropout"] < 1 and params["lr"] > 0 and params["weight_decay"] >= 0):
        raise ValueError(f"범위 밖 값: {params!r}")
    return params


def resolve_train_config(default_config, path=None):
    """
    설명: 채택된 튜닝 값이 있으면 그 값을 덮은 TrainConfig와 id를, 없거나 adopted=false·깨진 파일이면 (default_config, None)을 돌려준다.
    주의 사항: 예외를 던지지 않는다(튜닝 파일 때문에 일일 학습이 실패하면 안 됨). 로그는 "[TUNE]"로 시작한다.
    입력 타입->반환 타입: (TrainConfig, str|None) -> (TrainConfig, str|None)
    """
    path = path or default_best_params_path()
    try:
        if not os.path.exists(path):
            print(f"[TUNE] 없음→기본값 (튜닝 파일 없음: {path})")
            return default_config, None
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        if rec.get("adopted") is not True:
            print(f"[TUNE] 없음→기본값 (채택된 튜닝 값 없음: adopted={rec.get('adopted')!r}, id={rec.get('id')})")
            return default_config, None
        params = _validated_params(rec.get("params"))
        tuned_id = rec.get("id")
        if not isinstance(tuned_id, str) or not tuned_id:
            raise ValueError(f"id가 비어 있음: {tuned_id!r}")
        config = dataclasses.replace(default_config, **params)
        print(f"[TUNE] 채택값 사용 id={tuned_id} {params}")
        return config, tuned_id
    except Exception as e:
        print(f"[TUNE] ⚠️ 튜닝 파일을 읽지 못함 — 기본값으로 진행({type(e).__name__}: {e})")
        return default_config, None


# ── 탐색 후보 ───────────────────────────────────────────────────────────────────

def _round_sig(x, digits=2):
    """
    설명: x를 유효숫자 digits자리로 반올림한 float을 돌려준다.
    주의 사항: 없음.
    입력 타입->반환 타입: (float, int) -> float
    """
    return float(f"{x:.{digits - 1}e}")


def sample_candidates(n, seed=SEARCH_SEED):
    """
    설명: SEARCH_SPACE에서 시드 고정 무작위 후보 n개를 {인자: 값} 목록으로 만든다.
    주의 사항: 전역 난수 상태를 건드리지 않는다(별도 Generator). 같은 seed면 항상 같은 후보다.
    입력 타입->반환 타입: (int, int) -> list[dict]
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        lo_d, hi_d = SEARCH_SPACE["dropout"]
        out.append({
            "hidden_size": int(rng.choice(SEARCH_SPACE["hidden_size"])),
            "num_layers": int(rng.choice(SEARCH_SPACE["num_layers"])),
            "dropout": round(float(rng.uniform(lo_d, hi_d)), 2),
            "lr": _round_sig(float(np.exp(rng.uniform(*np.log(SEARCH_SPACE["lr"]))))),
            "weight_decay": _round_sig(float(np.exp(rng.uniform(*np.log(SEARCH_SPACE["weight_decay"]))))),
        })
    return out


# ── 채점 ───────────────────────────────────────────────────────────────────────

def val_chunk_masks(val_dates, n_chunks=N_VAL_CHUNKS):
    """
    설명: 검증 행의 날짜를 고유 날짜 기준 시간순 n_chunks조각(같은 길이, 나머지는 앞 조각부터 1일씩)으로 나눈 행 마스크 목록과 경계를 돌려준다.
    주의 사항: val_dates는 val loader 순서(shuffle 없음)와 같은 배열이어야 한다.
    입력 타입->반환 타입: (np.ndarray, int) -> (list[np.ndarray], list[(str, str)])
    """
    dates = np.asarray(val_dates).astype("datetime64[D]")
    chunks = np.array_split(np.unique(dates), n_chunks)
    masks = [np.isin(dates, c) for c in chunks]
    bounds = [(str(c[0]), str(c[-1])) for c in chunks]
    return masks, bounds


def score_model(model, val_loader, device, masks, baselines_by_ticker):
    """
    설명: 학습된 모델의 검증 조각별·전체 종목평균 RMSE와 게이트 판정을 dict로 돌려준다.
    주의 사항: 전체 점수와 게이트는 일일 학습의 evaluate_gate와 같은 함수로 계산한다.
    입력 타입->반환 타입: (nn.Module, DataLoader, torch.device, list[np.ndarray], dict) -> dict
    """
    preds, actuals, tids = evaluate_pooled_predictions(model, val_loader, device)
    chunks = [tickermean_rmse(preds[m], actuals[m], tids[m]) for m in masks]
    gate = evaluate_gate(model, val_loader, device, baselines_by_ticker)
    return {"chunks": chunks, "mean": float(np.mean(chunks)), "overall": gate["hybrid_rmse"], "gate": gate}


def test_score(state_dict, model_args, X_test, y_test, tid_test, batch_size, device):
    """
    설명: 저장해 둔 가중치로 모델을 다시 만들어 테스트(백테스트) 구간 종목평균 RMSE를 돌려준다.
    주의 사항: 기록 전용 — 채택 판단에 쓰지 않는다.
    입력 타입->반환 타입: (dict, dict, np.ndarray, np.ndarray, np.ndarray, int, torch.device) -> float
    """
    model = PooledLSTMRegressor(**model_args).to(device)
    model.load_state_dict(state_dict)
    loader = make_pooled_loader(X_test, y_test, tid_test, batch_size, shuffle=False)
    preds, actuals, tids = evaluate_pooled_predictions(model, loader, device)
    return tickermean_rmse(preds, actuals, tids)


def adoption_decision(default, best):
    """
    설명: 채택 규칙 (a) 3조각 평균이 기본보다 낮음 (b) 3조각 중 2개 이상에서 낮음 (c) 게이트 통과를 각각 판정한다.
    주의 사항: 셋 다 참일 때만 adopted=True.
    입력 타입->반환 타입: (dict, dict) -> dict
    """
    n_better = sum(b < d for b, d in zip(best["chunks"], default["chunks"]))
    a = best["mean"] < default["mean"]
    b = n_better >= 2
    c = bool(best["gate"]["passed"])
    return {"a_mean_lower": a, "b_chunks_lower": b, "n_chunks_lower": n_better, "c_gate_passed": c,
            "adopted": a and b and c}


# ── 결과 파일 쓰기 ──────────────────────────────────────────────────────────────

def code_commit():
    """
    설명: 이 코드 트리의 git 커밋 해시(12자리, 작업 트리가 더러우면 "+dirty")를 돌려준다.
    주의 사항: git이 없거나 실패하면 "unknown".
    입력 타입->반환 타입: () -> str
    """
    root = os.path.dirname(_HERE)
    try:
        head = subprocess.run(["git", "-C", root, "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain"], capture_output=True, text=True,
                               timeout=10, check=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def _write_json(path, obj):
    """
    설명: obj를 JSON으로 임시 파일에 쓴 뒤 path로 바꿔 끼운다(읽는 쪽이 반쯤 쓴 파일을 보지 않게).
    주의 사항: 파일 쓰기.
    입력 타입->반환 타입: (str, dict) -> None
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def save_result(record, out_dir=None):
    """
    설명: 이번 튜닝 기록을 history/에 남기고, 채택됐거나 이전 채택값이 없을 때만 best_params.json을 바꾼다(최근 8개 보관).
    주의 사항: 채택되지 않았고 이전 채택값이 있으면 best_params.json은 그대로(이전 값 유지). 돌려주는 값은 실제 적용 상태 설명.
    입력 타입->반환 타입: (dict, str|None) -> str
    """
    out_dir = out_dir or TUNING_DIR
    hist = os.path.join(out_dir, HISTORY_DIR)
    os.makedirs(hist, exist_ok=True)
    best_path = os.path.join(out_dir, BEST_PARAMS_FILE)
    previous = None
    if os.path.exists(best_path):
        try:
            with open(best_path, encoding="utf-8") as f:
                previous = json.load(f)
        except Exception:
            previous = None
    keep_previous = (not record["adopted"]) and previous is not None and previous.get("adopted") is True
    record = dict(record, kept_previous_id=previous.get("id") if keep_previous else None)
    _write_json(os.path.join(hist, f"{record['id']}.json"), record)
    if keep_previous:
        state = f"채택 안 됨 — 이전 채택값 유지(id={previous.get('id')})"
    else:
        _write_json(best_path, record)
        state = "채택 — best_params.json 교체" if record["adopted"] else "채택 안 됨 — 이전 채택값 없음, 기본값 유지"
    files = sorted((f for f in os.listdir(hist) if f.endswith(".json")),
                   key=lambda f: os.path.getmtime(os.path.join(hist, f)))
    for f in files[:-HISTORY_KEEP]:
        os.remove(os.path.join(hist, f))
    return state


# ── MLflow ───────────────────────────────────────────────────────────────────────

def log_tuning_run(record, tracking_uri, experiment=MLFLOW_TUNING_EXPERIMENT):
    """
    설명: 실험 "주간_튜닝"에 run "tune_YYYY-MM-DD"와 시도별 중첩 run을 기록한다.
    주의 사항: 네트워크 쓰기. 서버가 없거나 실패해도 예외를 내지 않는다(결과 파일은 이미 저장된 뒤에 부른다).
    입력 타입->반환 타입: (dict, str, str) -> None
    """
    from 가격예측 import store
    try:
        for k, v in store.MLFLOW_HTTP_LIMITS.items():
            os.environ.setdefault(k, v)
        if tracking_uri.startswith("http"):
            problem = store._mlflow_health_problem(tracking_uri)
            if problem:
                print(f"[MLFLOW-SKIP] ⚠️ MLflow 서버 응답 없음 — 튜닝 기록 건너뜀({problem}). 결과 파일은 저장됐다.")
                return
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        d, b = record["default"], record["best"]
        with mlflow.start_run(run_name=f"tune_{record['tuned_at_kst'][:10]}"):
            mlflow.set_tags({"tuned_params_id": record["id"], "adopted": str(record["adopted"]).lower(),
                             "code_commit": record["code_commit"], "end_date": record["end_date"],
                             "best_trial": b["trial"]})
            mlflow.log_params({"n_trials": record["n_trials"], "search_seed": record["search_seed"],
                               **{f"space_{k}": str(v) for k, v in SEARCH_SPACE.items()},
                               **{f"best_{k}": v for k, v in b["params"].items()}})
            mlflow.log_metrics({"default_val_score": d["mean"], "best_val_score": b["mean"],
                                "default_test_score": record["test"]["default"], "best_test_score": record["test"]["best"],
                                "adopted": int(record["adopted"])})
            for t in record["trials"]:
                with mlflow.start_run(run_name=f"trial_{t['trial']:02d}", nested=True):
                    mlflow.log_params({**t["params"], "is_default": t["trial"] == 0})
                    mlflow.log_metrics({**{f"val_chunk_{i + 1}": v for i, v in enumerate(t["chunks"])},
                                        "val_mean": t["mean"], "val_overall": t["overall"],
                                        "gate_passed": int(t["gate"]["passed"]), "best_epoch": t["best_epoch"],
                                        "train_sec": t["train_sec"]})
    except Exception as e:
        print(f"⚠️ MLflow 튜닝 기록 실패(결과 파일은 저장됨) — {e}")


# ── 튜닝 실행 ─────────────────────────────────────────────────────────────────────

def run_tuning(tickers, start_date, end_date, n_trials=N_TRIALS, seed=SEARCH_SEED, out_dir=None,
               tracking_uri=None, garch_cache_dir=None, garch_read_only=True, experiment=MLFLOW_TUNING_EXPERIMENT):
    """
    설명: 일일 학습과 같은 데이터로 0번(기본 설정)과 무작위 후보를 채점하고, 채택 규칙을 판정해 결과 파일·MLflow에 남긴 기록을 돌려준다.
    주의 사항: DB 읽기만. GARCH 캐시는 기본 읽기 전용(일일 실행이 먼저 갱신한다). 학습은 후보당 1회, 테스트 구간은 기본·최선만 1번씩.
    입력 타입->반환 타입: (list[str], str, str, int, int, str|None, str|None, str|None, bool, str) -> dict
    """
    from 가격예측 import 가격예측_변동성_공통 as pipeline  # 순환 import 방지(공통이 이 모듈의 resolve_train_config를 쓴다)
    from 가격예측 import store
    from 시장지표.feature_loader import load_price_cache

    t_start = time.time()
    default_config = pipeline.DEFAULT_TRAIN_CONFIG
    shared = pipeline.load_shared_cache(start_date, end_date)
    prices = load_price_cache(tickers, start_date, end_date)
    data = pipeline.prepare_pooled_training_data(tickers, start_date, end_date, shared_cache=shared,
                                                 prices_by_ticker=prices, garch_cache_dir=garch_cache_dir,
                                                 garch_read_only=garch_read_only)
    splits, scaler, baselines = data["splits"], data["scaler"], data["baselines_by_ticker"]
    _, y_train, tid_train, _, _ = splits["train"]
    _, y_val, tid_val, val_dates, _ = splits["val"]
    X_test, y_test, tid_test, test_dates, _ = splits["test"]
    X_test_s = scaler.transform(X_test).astype(np.float32)
    masks, chunk_bounds = val_chunk_masks(val_dates)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_prep = time.time() - t_start
    print(f"[TUNE] 데이터 준비 {n_prep:.0f}s — train {len(y_train)} / val {len(y_val)} / test {len(y_test)}, "
          f"검증 조각 {chunk_bounds}")

    candidates = [{k: getattr(default_config, k) for k in TUNED_FIELDS}] + sample_candidates(n_trials - 1, seed)
    trials, states = [], {}
    for i, params in enumerate(candidates):
        config = dataclasses.replace(default_config, **params)
        t0 = time.time()
        result = train_model(data["X_train_s"], y_train, tid_train, data["X_val_s"], y_val, tid_val, len(tickers),
                             pipeline.LOOKBACK, config, device, verbose=False, label=f"tune-{i:02d}")
        s = score_model(result["model"], result["val_loader"], device, masks, baselines)
        elapsed = time.time() - t0
        states[i] = {k: v.detach().cpu().clone() for k, v in result["model"].state_dict().items()}
        trials.append({"trial": i, "params": params, "chunks": s["chunks"], "mean": s["mean"], "overall": s["overall"],
                       "gate": s["gate"], "best_epoch": result["best_epoch"], "last_epoch": result["last_epoch"],
                       "train_sec": round(elapsed, 1)})
        print(f"[TUNE] 시도 {i:02d} {params} → 조각 {[round(c, 6) for c in s['chunks']]} 평균 {s['mean']:.6f} "
              f"전체 {s['overall']:.6f} 게이트 {'통과' if s['gate']['passed'] else '미통과'} "
              f"(best epoch {result['best_epoch']}, {elapsed:.0f}s)", flush=True)
        del result
        if device.type == "cuda":
            torch.cuda.empty_cache()

    default = trials[0]
    best = min(trials[1:], key=lambda t: (t["mean"], t["trial"])) if len(trials) > 1 else default
    decision = adoption_decision(default, best)

    def model_args(params):
        """
        설명: 후보 params로 PooledLSTMRegressor 생성자 인자를 만든다(일일 학습의 train_model과 같은 구성).
        주의 사항: embedding_dim은 탐색하지 않으므로 기본값.
        입력 타입->반환 타입: (dict) -> dict
        """
        return dict(n_features=X_test.shape[2], lookback=pipeline.LOOKBACK, num_stocks=len(tickers),
                    embedding_dim=default_config.embedding_dim, hidden_size=params["hidden_size"],
                    num_layers=params["num_layers"], dropout=params["dropout"])
    test = {name: test_score(states[t["trial"]], model_args(t["params"]), X_test_s, y_test, tid_test,
                             default_config.batch_size, device) for name, t in (("default", default), ("best", best))}

    now = datetime.now(KST)
    params_hash = hashlib.sha256(json.dumps(best["params"], sort_keys=True).encode()).hexdigest()[:8]
    record = {
        "id": f"tune_{now:%Y%m%d_%H%M%S}_{params_hash}",
        "adopted": decision["adopted"], "params": best["params"], "decision": decision,
        "tuned_at_kst": now.isoformat(timespec="seconds"), "end_date": str(end_date),
        "data_last_date": str(np.asarray(test_dates).astype("datetime64[D]").max()),
        "split": {"train_end": str(data["train_end"].date()), "val_end": str(data["val_end"].date()),
                  "val_chunks": chunk_bounds},
        "default": {k: default[k] for k in ("trial", "params", "chunks", "mean", "overall")},
        "best": {k: best[k] for k in ("trial", "params", "chunks", "mean", "overall")} | {"gate": best["gate"]},
        "test": test, "test_note": "백테스트(테스트) 구간 종목평균 RMSE — 기록 전용, 채택 판단에 쓰지 않음",
        "trials": trials, "n_trials": n_trials, "search_seed": seed,
        "search_space": {k: list(v) for k, v in SEARCH_SPACE.items()},
        "code_commit": code_commit(), "elapsed_sec": round(time.time() - t_start, 1), "prep_sec": round(n_prep, 1),
    }
    state = save_result(record, out_dir)
    print(f"[TUNE] 판정: (a) 평균 낮음 {decision['a_mean_lower']} (b) 조각 {decision['n_chunks_lower']}/3 낮음 "
          f"{decision['b_chunks_lower']} (c) 게이트 {decision['c_gate_passed']} → {state}")
    print(f"[TUNE] 테스트 구간(기록 전용): 기본 {test['default']:.6f} / 최선(시도 {best['trial']:02d}) {test['best']:.6f}")
    log_tuning_run(record, tracking_uri or store.MLFLOW_TRACKING_URI, experiment)
    return record
