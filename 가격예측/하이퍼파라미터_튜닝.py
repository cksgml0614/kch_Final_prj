# 하이퍼파라미터_튜닝.py — 주 1회 하이퍼파라미터 탐색·채점·채택과, 일일 학습이 읽는 튜닝 결과 파일(best_params.json) 관리.
# 파이프라인 단계: 튜닝(시퀀스와 학습 사이, 주 1회). 주요 진입: run_tuning(주간_튜닝.py), resolve_train_config(일일 학습).
# 선택: train 구간 안의 시간순 expanding-window 3폴드 점수(종목평균 RMSE)로만 고른다. 게이트 val은 최종 판정(c)에만, 테스트는 기록만.
# 배경: 결정기록.md의 "주간 튜닝: train 안 3폴드로 고르고 val은 게이트에만", "튜닝 값이 없거나 깨지면 일일 학습은 기본값"

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

from 가격예측.evaluate import evaluate_pooled_predictions, tickermean_rmse
from 가격예측.sequence_dataset import FeatureScaler
from 가격예측.train_common import make_pooled_loader, train_model

KST = timezone(timedelta(hours=9))
_HERE = os.path.dirname(os.path.abspath(__file__))
TUNING_DIR = os.path.join(_HERE, "checkpoints", "tuning")
BEST_PARAMS_FILE = "best_params.json"
HISTORY_DIR = "history"
HISTORY_KEEP = 8
MLFLOW_TUNING_EXPERIMENT = "주간_튜닝"

N_TRIALS = 12          # 0번 = 기본 설정, 1~11번 = 시드 고정 무작위 후보(채택값이 있으면 "현재 적용 값"을 따로 하나 더 채점)
SEARCH_SEED = 20261018
# 폴드: train target 날짜의 끝쪽에 같은 길이(각 15%) 평가 조각 3개. 폴드 k는 조각 k 앞 날짜 전부로 학습(expanding).
# 학습 끝과 평가 시작 사이는 FOLD_PURGE_DAYS 거래일을 비운다(= LOOKBACK — 평가 시퀀스의 입력 창이 학습 target일과 겹치지 않게).
# 폴드 안 early stopping은 일일 학습과 같은 구조: 폴드 학습 날짜의 마지막 15%를 멈춤 판단 전용으로 떼고, scaler는 그 앞에만 맞춘다.
N_FOLDS = 3
FOLD_EVAL_RATIO = 0.15
FOLD_STOP_RATIO = 0.15
FOLD_PURGE_DAYS = 20
MIN_REL_IMPROVEMENT = 0.005   # 채택 규칙 (a): 폴드 평균이 기준보다 0.5% 이상 낮아야 한다
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


def read_adopted(path=None):
    """
    설명: 결과 파일에서 채택된 값을 읽어 (params, id, 설명)을 돌려준다. 채택값이 없거나 파일이 없거나 깨졌으면 (None, None, 설명).
    주의 사항: 예외를 던지지 않는다. 설명 첫 단어는 "없음"(파일 없음·adopted≠true) 또는 "깨짐"(읽기·검사 실패).
    입력 타입->반환 타입: (str|None) -> (dict|None, str|None, str)
    """
    path = path or default_best_params_path()
    try:
        if not os.path.exists(path):
            return None, None, f"없음 (튜닝 파일 없음: {path})"
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        if rec.get("adopted") is not True:
            return None, None, f"없음 (채택된 튜닝 값 없음: adopted={rec.get('adopted')!r}, id={rec.get('id')})"
        params = _validated_params(rec.get("params"))
        tuned_id = rec.get("id")
        if not isinstance(tuned_id, str) or not tuned_id:
            raise ValueError(f"id가 비어 있음: {tuned_id!r}")
        return dict(params), tuned_id, f"채택값 id={tuned_id}"
    except Exception as e:
        return None, None, f"깨짐 ({type(e).__name__}: {e})"


def resolve_train_config(default_config, path=None):
    """
    설명: 채택된 튜닝 값이 있으면 그 값을 덮은 TrainConfig와 id를, 없거나 adopted=false·깨진 파일이면 (default_config, None)을 돌려준다.
    주의 사항: 예외를 던지지 않는다(튜닝 파일 때문에 일일 학습이 실패하면 안 됨). 로그는 "[TUNE]"로 시작한다.
    입력 타입->반환 타입: (TrainConfig, str|None) -> (TrainConfig, str|None)
    """
    params, tuned_id, note = read_adopted(path)
    if params is None:
        if note.startswith("깨짐"):
            print(f"[TUNE] ⚠️ 튜닝 파일을 읽지 못함 — 기본값으로 진행{note[2:]}")
        else:
            print(f"[TUNE] 없음→기본값{note[2:]}")
        return default_config, None
    print(f"[TUNE] 채택값 사용 id={tuned_id} {params}")
    return dataclasses.replace(default_config, **params), tuned_id


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


# ── 폴드 ───────────────────────────────────────────────────────────────────────

def build_folds(train_dates, n_folds=N_FOLDS, eval_ratio=FOLD_EVAL_RATIO, stop_ratio=FOLD_STOP_RATIO,
                purge=FOLD_PURGE_DAYS):
    """
    설명: train target 날짜로 시간순 expanding-window 폴드 경계(학습·멈춤 판단·간격·평가 날짜 범위)를 만든다.
    주의 사항: 날짜는 고유 target일 기준. 평가 조각은 같은 길이이고 마지막 조각은 train 마지막 날에서 끝난다. 겹침은 assert로 막는다.
    입력 타입->반환 타입: (np.ndarray, int, float, float, int) -> list[dict]
    """
    dates = np.unique(np.asarray(train_dates).astype("datetime64[D]"))
    n = len(dates)
    n_eval = int(round(n * eval_ratio))
    folds = []
    for k in range(n_folds):
        e0 = n - (n_folds - k) * n_eval
        fit_end = e0 - purge - 1
        n_stop = int(round((fit_end + 1) * stop_ratio))
        assert fit_end - n_stop > 0 and e0 > fit_end, "폴드를 만들 train 날짜가 부족함"
        folds.append({
            "fold": k + 1, "train": (dates[0], dates[fit_end - n_stop]), "stop": (dates[fit_end - n_stop + 1], dates[fit_end]),
            "purge": (dates[fit_end + 1], dates[e0 - 1]), "eval": (dates[e0], dates[e0 + n_eval - 1]),
            "n_dates": {"train": fit_end - n_stop + 1, "stop": n_stop, "purge": e0 - fit_end - 1, "eval": n_eval},
        })
    return folds


def fold_bounds_text(folds):
    """
    설명: 폴드 경계를 결과 파일·로그용 문자열 dict 목록으로 바꾼다.
    주의 사항: 없음.
    입력 타입->반환 타입: (list[dict]) -> list[dict]
    """
    return [{"fold": f["fold"], **{k: [str(f[k][0]), str(f[k][1])] for k in ("train", "stop", "purge", "eval")},
             "n_dates": f["n_dates"]} for f in folds]


def score_on_folds(config, train_arrays, folds, num_stocks, lookback, device):
    """
    설명: 한 설정을 폴드마다 학습(멈춤 판단 구간으로 early stopping)하고 평가 조각의 종목평균 RMSE를 구해 dict로 돌려준다.
    주의 사항: scaler는 폴드 학습 구간에만 맞춘다. 평가 조각은 학습·멈춤 판단에 쓰지 않는다. 시드는 config.seed로 폴드마다 고정.
    입력 타입->반환 타입: (TrainConfig, tuple, list[dict], int, int, torch.device) -> dict
    """
    X, y, tid, dates = train_arrays
    d = np.asarray(dates).astype("datetime64[D]")
    scores, epochs, t0 = [], [], time.time()
    for f in folds:
        tr = d <= f["train"][1]
        st = (d >= f["stop"][0]) & (d <= f["stop"][1])
        ev = (d >= f["eval"][0]) & (d <= f["eval"][1])
        scaler = FeatureScaler().fit(X[tr])
        Xs = {k: scaler.transform(X[m]).astype(np.float32) for k, m in (("tr", tr), ("st", st), ("ev", ev))}
        result = train_model(Xs["tr"], y[tr], tid[tr], Xs["st"], y[st], tid[st], num_stocks, lookback, config, device,
                             verbose=False, label=f"fold{f['fold']}")
        loader = make_pooled_loader(Xs["ev"], y[ev], tid[ev], config.batch_size, shuffle=False)
        p, a, t = evaluate_pooled_predictions(result["model"], loader, device)
        scores.append(tickermean_rmse(p, a, t))
        epochs.append(result["best_epoch"])
        del result
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"fold_scores": scores, "fold_mean": float(np.mean(scores)), "fold_best_epochs": epochs,
            "train_sec": round(time.time() - t0, 1)}


def compare_to_reference(cand, ref, min_rel=MIN_REL_IMPROVEMENT):
    """
    설명: 채택 규칙 (a)·(b)를 판정한다 — (a) 폴드 평균이 기준보다 min_rel 이상 상대적으로 낮음 (b) 폴드 중 과반(3개 중 2개 이상)에서 낮음.
    주의 사항: (c) 게이트는 따로 판정한다. 상대 개선 = (기준 - 후보) / 기준.
    입력 타입->반환 타입: (dict, dict, float) -> dict
    """
    rel = (ref["fold_mean"] - cand["fold_mean"]) / ref["fold_mean"]
    n_lower = sum(c < r for c, r in zip(cand["fold_scores"], ref["fold_scores"]))
    return {"rel_improvement": rel, "min_rel_improvement": min_rel, "a_fold_mean_improved": rel >= min_rel,
            "n_folds_lower": n_lower, "b_folds_lower": n_lower * 2 > len(ref["fold_scores"])}


def test_score(model, X_test, y_test, tid_test, batch_size, device):
    """
    설명: 일일 학습 경로로 학습한 모델의 테스트(백테스트) 구간 종목평균 RMSE를 돌려준다.
    주의 사항: 기록 전용 — 채택 판단에 쓰지 않는다.
    입력 타입->반환 타입: (nn.Module, np.ndarray, np.ndarray, np.ndarray, int, torch.device) -> float
    """
    loader = make_pooled_loader(X_test, y_test, tid_test, batch_size, shuffle=False)
    preds, actuals, tids = evaluate_pooled_predictions(model, loader, device)
    return tickermean_rmse(preds, actuals, tids)


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
    설명: 이번 기록을 history/에 남기고, state가 "kept"(현재 적용 값 유지)가 아니면 best_params.json을 이 기록으로 바꾼다(최근 8개 보관).
    주의 사항: adopted=false로 바뀌면(state "reverted"·"none") 다음 일일 학습은 기본값이다.
    입력 타입->반환 타입: (dict, str|None) -> None
    """
    out_dir = out_dir or TUNING_DIR
    hist = os.path.join(out_dir, HISTORY_DIR)
    os.makedirs(hist, exist_ok=True)
    _write_json(os.path.join(hist, f"{record['id']}.json"), record)
    if record["state"] != "kept":
        _write_json(os.path.join(out_dir, BEST_PARAMS_FILE), record)
    files = sorted((f for f in os.listdir(hist) if f.endswith(".json")),
                   key=lambda f: os.path.getmtime(os.path.join(hist, f)))
    for f in files[:-HISTORY_KEEP]:
        os.remove(os.path.join(hist, f))


def decide_state(applied_id, revert, decision, gate_passed):
    """
    설명: 채택 규칙 결과로 이번 튜닝의 상태(adopted·kept·reverted·none)와 사유 문장을 정한다.
    주의 사항: 후보 채택이 되돌림보다 우선한다(되돌린 뒤 기준이 기본값으로 바뀐 상태에서 후보가 이기면 adopted).
    입력 타입->반환 타입: (str|None, bool, dict, bool) -> (str, str)
    """
    ok_ab = decision["a_fold_mean_improved"] and decision["b_folds_lower"]
    if ok_ab and gate_passed:
        return "adopted", "후보가 기준을 (a)·(b)·(c) 모두 충족"
    why = ("(a) 상대 개선 {:.3%} < {:.1%}".format(decision["rel_improvement"], decision["min_rel_improvement"])
           if not decision["a_fold_mean_improved"] else
           f"(b) 낮은 폴드 {decision['n_folds_lower']}개" if not decision["b_folds_lower"] else "(c) 게이트 미통과")
    if revert:
        return "reverted", f"현재 적용 값({applied_id})이 기본값보다 규칙상 나빠 기본값으로 되돌림; 후보 기각: {why}"
    if applied_id:
        return "kept", f"후보 기각({why}) — 현재 적용 값({applied_id}) 유지"
    return "none", f"후보 기각({why}) — 채택값 없음, 기본값 유지"


# ── MLflow ───────────────────────────────────────────────────────────────────────

def log_tuning_run(record, tracking_uri, experiment=MLFLOW_TUNING_EXPERIMENT):
    """
    설명: 실험 "주간_튜닝"에 부모 run "tune_YYYY-MM-DD"(태그 run_kind=tuning_parent)와 시도별 중첩 run(run_kind=tuning_trial)을 기록한다.
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
        dflt, best, dec = record["default"], record["best"], record["decision"]
        metrics = {"default_fold_score": dflt["fold_mean"], "reference_fold_score": record["reference"]["fold_mean"],
                   "best_fold_score": best["fold_mean"], "best_fold_rel_improvement": dec["rel_improvement"],
                   "best_n_folds_lower": dec["n_folds_lower"], "default_gate_rmse": dflt["gate_rmse"],
                   "best_gate_rmse": best["gate_rmse"], "default_test_score": record["test"]["default"],
                   "best_test_score": record["test"]["best"], "adopted": int(record["adopted"])}
        if record["current"] is not None:
            metrics.update({"current_fold_score": record["current"]["fold_mean"],
                            "current_gate_rmse": record["current"]["gate_rmse"],
                            "current_test_score": record["test"]["current"]})
        with mlflow.start_run(run_name=f"tune_{record['tuned_at_kst'][:10]}"):
            mlflow.set_tags({"run_kind": "tuning_parent", "adopted": str(record["adopted"]).lower(),
                             "state": record["state"], "tuned_params_id": record["id"],
                             "reference": record["reference"]["label"], "code_commit": record["code_commit"],
                             "end_date": record["end_date"], "best_trial": best["trial"]})
            mlflow.log_params({"n_trials": record["n_trials"], "search_seed": record["search_seed"], "n_folds": N_FOLDS,
                               "fold_purge_days": FOLD_PURGE_DAYS, "min_rel_improvement": MIN_REL_IMPROVEMENT,
                               **{f"space_{k}": str(v) for k, v in SEARCH_SPACE.items()},
                               **{f"best_{k}": v for k, v in best["params"].items()}})
            mlflow.log_metrics(metrics)
            for t in record["trials"]:
                with mlflow.start_run(run_name=f"trial_{t['trial']:02d}_{t['label']}", nested=True):
                    mlflow.set_tags({"run_kind": "tuning_trial", "trial_label": t["label"]})
                    mlflow.log_params(t["params"])
                    mlflow.log_metrics({**{f"fold_{i + 1}_score": v for i, v in enumerate(t["fold_scores"])},
                                        **{f"fold_{i + 1}_best_epoch": v for i, v in enumerate(t["fold_best_epochs"])},
                                        "fold_mean_score": t["fold_mean"], "train_sec": t["train_sec"]})
    except Exception as e:
        print(f"⚠️ MLflow 튜닝 기록 실패(결과 파일은 저장됨) — {e}")


# ── 튜닝 실행 ─────────────────────────────────────────────────────────────────────

def run_tuning(tickers, start_date, end_date, n_trials=N_TRIALS, seed=SEARCH_SEED, out_dir=None,
               tracking_uri=None, garch_cache_dir=None, garch_read_only=True, experiment=MLFLOW_TUNING_EXPERIMENT):
    """
    설명: train 안 3폴드로 기본값·현재 적용 값·무작위 후보를 채점하고, 최선 후보를 일일 학습 경로로 학습해 게이트를 본 뒤 채택 여부를 정해 기록한다.
    주의 사항: DB 읽기만. GARCH 캐시는 기본 읽기 전용(일일 실행이 먼저 갱신한다). val은 게이트(c)에만, 테스트 구간은 기록에만 쓴다.
    입력 타입->반환 타입: (list[str], str, str, int, int, str|None, str|None, str|None, bool, str) -> dict
    """
    from 가격예측 import 가격예측_변동성_공통 as pipeline  # 순환 import 방지(공통이 이 모듈의 resolve_train_config를 쓴다)
    from 가격예측 import store
    from 시장지표.feature_loader import load_price_cache

    assert FOLD_PURGE_DAYS >= pipeline.LOOKBACK, "폴드 간격은 LOOKBACK 이상이어야 한다"
    t_start = time.time()
    out_dir = out_dir or TUNING_DIR
    default_config = pipeline.DEFAULT_TRAIN_CONFIG
    default_params = {k: getattr(default_config, k) for k in TUNED_FIELDS}
    applied_params, applied_id, note = read_adopted(os.path.join(out_dir, BEST_PARAMS_FILE))
    print(f"[TUNE] 현재 적용 값: {note}")
    if applied_params == default_params:
        applied_params, applied_id = None, None  # 기본값과 같으면 기준은 기본값 하나

    shared = pipeline.load_shared_cache(start_date, end_date)
    prices = load_price_cache(tickers, start_date, end_date)
    data = pipeline.prepare_pooled_training_data(tickers, start_date, end_date, shared_cache=shared,
                                                 prices_by_ticker=prices, garch_cache_dir=garch_cache_dir,
                                                 garch_read_only=garch_read_only)
    X_tr, y_tr, tid_tr, d_tr, _ = data["splits"]["train"]
    X_test, y_test, tid_test, test_dates, _ = data["splits"]["test"]
    X_test_s = data["scaler"].transform(X_test).astype(np.float32)
    folds = build_folds(d_tr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_prep = time.time() - t_start
    print(f"[TUNE] 데이터 준비 {n_prep:.0f}s — train {len(y_tr)} / test {len(y_test)}, 폴드 {fold_bounds_text(folds)}")

    plan = [("default", default_params)]
    if applied_params is not None:
        plan.append(("current", applied_params))
    plan += [("random", p) for p in sample_candidates(n_trials - 1, seed)]
    trials = []
    for i, (label, params) in enumerate(plan):
        s = score_on_folds(dataclasses.replace(default_config, **params), (X_tr, y_tr, tid_tr, d_tr), folds,
                           len(tickers), pipeline.LOOKBACK, device)
        trials.append({"trial": i, "label": label, "params": params, **s})
        print(f"[TUNE] 시도 {i:02d} {label} {params} → 폴드 {[round(x, 6) for x in s['fold_scores']]} "
              f"평균 {s['fold_mean']:.6f} (best epoch {s['fold_best_epochs']}, {s['train_sec']:.0f}s)", flush=True)

    t_default = trials[0]
    t_current = trials[1] if applied_params is not None else None
    revert_check = compare_to_reference(t_default, t_current) if t_current else None
    revert = bool(revert_check and revert_check["a_fold_mean_improved"] and revert_check["b_folds_lower"])
    reference = t_default if (t_current is None or revert) else t_current
    randoms = [t for t in trials if t["label"] == "random"]
    best = min(randoms, key=lambda t: (t["fold_mean"], t["trial"]))
    decision = compare_to_reference(best, reference)

    # 일일 학습 경로(train 학습 → val 게이트)로 기본값·현재 적용 값·최선 후보를 학습한다. val은 여기(c)에서만, 테스트는 기록만.
    finals = {}
    for name, t in (("default", t_default), ("current", t_current), ("best", best)):
        if t is None:
            continue
        out = pipeline.train_and_gate(data, len(tickers), dataclasses.replace(default_config, **t["params"]))
        finals[name] = {"gate": out["gate"], "test": test_score(out["model"], X_test_s, y_test, tid_test,
                                                                default_config.batch_size, out["device"])}
        del out
    decision["c_gate_passed"] = bool(finals["best"]["gate"]["passed"])
    state, reason = decide_state(applied_id, revert, decision, decision["c_gate_passed"])
    adopted = state == "adopted"

    def summary(t, name):
        """
        설명: 시도 하나의 폴드 점수와 일일 경로 게이트 값을 결과 파일용 dict로 만든다.
        주의 사항: name은 finals의 키(default·current·best).
        입력 타입->반환 타입: (dict, str) -> dict
        """
        g = finals[name]["gate"]
        return {"trial": t["trial"], "label": t["label"], "params": t["params"], "fold_scores": t["fold_scores"],
                "fold_mean": t["fold_mean"], "gate_rmse": g["hybrid_rmse"], "gate_passed": g["passed"], "gate": g}

    now = datetime.now(KST)
    chosen = best["params"]
    params_hash = hashlib.sha256(json.dumps(chosen, sort_keys=True).encode()).hexdigest()[:8]
    record = {
        "id": f"tune_{now:%Y%m%d_%H%M%S}_{params_hash}", "adopted": adopted, "state": state, "state_reason": reason,
        "params": chosen, "applied_before": {"id": applied_id, "params": applied_params} if applied_id else None,
        "decision": decision, "revert_check": revert_check,
        "reference": {"label": reference["label"], "trial": reference["trial"], "fold_mean": reference["fold_mean"]},
        "default": summary(t_default, "default"), "current": summary(t_current, "current") if t_current else None,
        "best": summary(best, "best"),
        "test": {k: v["test"] for k, v in finals.items()},
        "test_note": "백테스트(테스트) 구간 종목평균 RMSE — 기록 전용, 채택 판단에 쓰지 않음",
        "tuned_at_kst": now.isoformat(timespec="seconds"), "end_date": str(end_date),
        "data_last_date": str(np.asarray(test_dates).astype("datetime64[D]").max()),
        "split": {"train_end": str(data["train_end"].date()), "val_end": str(data["val_end"].date())},
        "folds": fold_bounds_text(folds), "trials": trials, "n_trials": n_trials, "search_seed": seed,
        "search_space": {k: list(v) for k, v in SEARCH_SPACE.items()},
        "fold_settings": {"n_folds": N_FOLDS, "eval_ratio": FOLD_EVAL_RATIO, "stop_ratio": FOLD_STOP_RATIO,
                          "purge_days": FOLD_PURGE_DAYS, "min_rel_improvement": MIN_REL_IMPROVEMENT},
        "code_commit": code_commit(), "elapsed_sec": round(time.time() - t_start, 1), "prep_sec": round(n_prep, 1),
    }
    save_result(record, out_dir)
    print(f"[TUNE] 판정: 기준={reference['label']} (a) 상대 개선 {decision['rel_improvement']:.3%} "
          f"(≥{MIN_REL_IMPROVEMENT:.1%}: {decision['a_fold_mean_improved']}) (b) 폴드 {decision['n_folds_lower']}/{N_FOLDS} "
          f"낮음 {decision['b_folds_lower']} (c) 게이트 {decision['c_gate_passed']} → {state}: {reason}")
    print(f"[TUNE] 게이트 RMSE(일일 경로): 기본 {finals['default']['gate']['hybrid_rmse']:.6f} / 최선(시도 {best['trial']:02d}) "
          f"{finals['best']['gate']['hybrid_rmse']:.6f}; 테스트(기록 전용): " +
          " / ".join(f"{k} {v['test']:.6f}" for k, v in finals.items()))
    log_tuning_run(record, tracking_uri or store.MLFLOW_TRACKING_URI, experiment)
    return record
