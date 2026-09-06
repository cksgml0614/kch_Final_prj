# train_common.py — Task T-1 학습 루프 공통 유틸.
# train.py(체크포인트4, v1, 폐기됨)와 train_v2.py(피처 재설계 이후)가 동일 로직을 공유했다.
#
# 2026-09-06(Task T 자동화 착수, 설계 결정 확정): 진단 도구들이 공유하던 이 파일을
# 운영 파이프라인의 공용 인프라로 확장한다 — 체크포인트 저장/로드, 배포 게이트 판정,
# 무인 실행 예외 격리를 추가했다. train_transformer()/evaluate_predictions() 등 기존
# 함수는 동작을 바꾸지 않았다(진단 스크립트들이 그대로 계속 참조 가능). confusion_at_
# threshold()는 diagnose_down_class.py에서 이 파일로 옮겨왔다(자동화 파이프라인이
# 일회성 진단 스크립트를 import하지 않도록) — diagnose_down_class.py/seed_stability_
# check.py는 이제 여기서 import해서 쓴다.
#
# 2026-09-06(Task T 다종목 pooled 모델, 별도 실험 트랙): PooledTransformerRegressor
# (forward(x, ticker_ids) — 기존 TransformerRegressor와 시그니처가 다름) 학습을 위해
# pooled 전용 학습/평가 루프(run_pooled_epoch/evaluate_pooled_predictions/
# train_pooled_transformer)를 추가했다. 기존 run_epoch/evaluate_predictions/
# train_transformer는 단일 종목 파이프라인이 계속 쓰므로 값을 바꾸지 않고 별도 함수로
# 분리했다(코드 중복이 있지만, 두 모델의 forward 시그니처 자체가 달라 억지로 하나로
# 합치면 오히려 두 경로 모두 이해하기 어려워진다고 판단). save_checkpoint/load_checkpoint는
# model_class_name/model_class 파라미터로 두 모델 클래스를 모두 지원하도록 확장했다 —
# 기존 호출부(가격예측_공통.py)는 인자를 안 줘도 동작이 그대로다(TransformerRegressor가
# 기본값).

import json
import os
import random
import traceback
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from 가격예측.model import PooledTransformerRegressor, TransformerRegressor
from 가격예측.sequence_dataset import FeatureScaler

_MODEL_CLASSES = {
    "TransformerRegressor": TransformerRegressor,
    "PooledTransformerRegressor": PooledTransformerRegressor,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, criterion, optimizer, device, train_mode):
    model.train(train_mode)
    total_loss, n = 0.0, 0
    with torch.set_grad_enabled(train_mode):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(xb)
            n += len(xb)
    return total_loss / n


def evaluate_predictions(model, loader, device):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
            actuals.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(actuals)


def directional_accuracy(preds, actuals):
    return float((np.sign(preds) == np.sign(actuals)).mean())


def majority_baseline_accuracy(actuals):
    up = float((actuals > 0).mean())
    down = float((actuals < 0).mean())
    if up >= down:
        return "항상 상승 예측", up
    return "항상 하락 예측", down


def train_transformer(X_train, y_train, X_val, y_val, n_features, lookback,
                       d_model, nhead, num_layers, dim_feedforward, dropout,
                       batch_size, lr, weight_decay, smoke_epochs, max_epochs, patience,
                       device, seed=42, verbose=True, label=""):
    """test는 절대 건드리지 않는다 — train/val 로더만 받아서 학습·검증한다."""
    set_seed(seed)
    model = TransformerRegressor(n_features, lookback, d_model, nhead, num_layers,
                                  dim_feedforward, dropout).to(device)

    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    prefix = f"[{label}] " if label else ""

    if verbose:
        print(f"{prefix}스모크 테스트 ({smoke_epochs} epoch)")
    for epoch in range(1, smoke_epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, False)
        history.append((epoch, train_loss, val_loss))
        finite = np.isfinite(train_loss) and np.isfinite(val_loss)
        if verbose:
            print(f"{prefix}  epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} finite={finite}")
        if not finite:
            raise RuntimeError(f"{label}: 스모크 테스트 중 loss가 NaN/Inf")

    if verbose:
        print(f"{prefix}본 학습 (최대 {max_epochs} epoch, patience={patience})")
    best_val_loss = float("inf")
    best_state = None
    best_epoch = smoke_epochs
    patience_counter = 0

    for epoch in range(smoke_epochs + 1, smoke_epochs + 1 + max_epochs):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, False)
        history.append((epoch, train_loss, val_loss))
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
            marker = " *best*"
        else:
            patience_counter += 1
        if verbose:
            print(f"{prefix}  epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}{marker}")
        if patience_counter >= patience:
            if verbose:
                print(f"{prefix}  early stopping (patience={patience}, best epoch={best_epoch})")
            break

    model.load_state_dict(best_state)
    last_epoch, last_train_loss, last_val_loss = history[-1]

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "last_epoch": last_epoch,
        "last_train_loss": last_train_loss,
        "last_val_loss": last_val_loss,
        "val_loader": val_loader,
        "overfit_ratio_at_end": last_val_loss / last_train_loss if last_train_loss > 0 else float("inf"),
    }


# ── 하락 클래스 confusion matrix (diagnose_down_class.py에서 이관, 2026-09-06) ──────

def confusion_at_threshold(preds, actuals, threshold=0.0):
    """'하락'(actual<0)을 양성 클래스로: pred < threshold면 '하락 예측'.
    diagnose_down_class.py(2026-08-17 진단)에서 그대로 옮겨왔다 — 자동화 파이프라인의
    배포 게이트(passes_deployment_gate)가 이 계산을 재사용한다."""
    pred_down = preds < threshold
    actual_down = actuals < 0

    TP = int(np.sum(pred_down & actual_down))
    FP = int(np.sum(pred_down & ~actual_down))
    FN = int(np.sum(~pred_down & actual_down))
    TN = int(np.sum(~pred_down & ~actual_down))

    precision = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
    recall = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
    beta = 2
    f2 = ((1 + beta**2) * precision * recall / (beta**2 * precision + recall)
          if (precision + recall) > 0 else float("nan"))

    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "precision": precision, "recall": recall, "f1": f1, "f2": f2}


# ── 배포 게이트 (diagnose_down_class.py/seed_stability_check.py의 확정 기준을 함수로 분리, 2026-09-06) ──

DEFAULT_PRECISION_MARGIN = 0.02  # 하락 비율(baseline) 대비 +2%p — 2026-08-17 사람 승인
DEFAULT_RECALL_FLOOR = 0.5


def passes_deployment_gate(preds, actuals, threshold=0.0,
                            precision_margin=DEFAULT_PRECISION_MARGIN, recall_floor=DEFAULT_RECALL_FLOOR):
    """diagnose_down_class.py/seed_stability_check.py(2026-08-17)가 확정한 배포 게이트 기준을
    재사용 가능한 함수로 분리했다(2026-09-06, Task T 자동화 착수) — 일회성 진단 스크립트가
    아니라 여기(train_common.py, 운영 파이프라인이 참조하는 공용 인프라)에 둔다.

    하락(actual<0)을 양성 클래스로 두고: precision이 '하락 비율(=이 actuals 안에서의 실제
    하락 비율)' 대비 +precision_margin 이상, recall이 recall_floor 이상이면 게이트 통과.
    seed_stability_check.py의 개별 시드별 판정 로직과 수치가 동일하게 나오는지 이관 시 확인했다.

    반환: confusion_at_threshold() 결과 dict + {down_rate, precision_gate, recall_floor,
    precision_pass, recall_pass, passed}."""
    m = confusion_at_threshold(preds, actuals, threshold=threshold)
    down_rate = float((actuals < 0).mean())
    precision_gate = down_rate + precision_margin

    precision_pass = (not np.isnan(m["precision"])) and m["precision"] >= precision_gate
    recall_pass = (not np.isnan(m["recall"])) and m["recall"] >= recall_floor

    return {
        **m,
        "down_rate": down_rate,
        "precision_gate": precision_gate,
        "recall_floor": recall_floor,
        "precision_pass": precision_pass,
        "recall_pass": recall_pass,
        "passed": precision_pass and recall_pass,
    }


# ── 변동성 예측 배포 게이트 (신규, 2026-09-06 — Task T 방향예측 폐기 + 변동성 예측 전환) ──
#
# 방향 예측용 passes_deployment_gate()는 값을 바꾸지 않고 그대로 둔다(기록 보존 — 결과_TaskT_
# 방향예측_폐기.md 참고, 더 이상 운영 파이프라인이 호출하지는 않지만 과거 실험 재현/감사용으로
# 남겨둔다). 변동성 예측은 판정 기준 자체가 다르다(precision/recall이 아니라 RMSE 비교이므로
# 별도 함수로 분리했다 — 하나의 함수에 두 판정 로직을 욱여넣으면 오히려 어느 쪽도 이해하기
# 어려워진다고 판단).

def passes_deployment_gate_volatility(hybrid_rmse, garch_rmse, sma_rmse):
    """하이브리드(GARCH+Transformer) 변동성 예측 배포 게이트 — 하이브리드 RMSE가 GARCH(1,1)와
    SMA20 baseline 둘 다보다 낮아야(개선돼야) 통과한다. 결과_변동성_조기경보_검증.md/
    결과_TaskT_방향예측_폐기.md에서 확인된 실증 결과(하이브리드가 GARCH·SMA20 둘 다를
    이김, 42~43% 개선)를 운영 게이트로 고정한 것 — 매일 재학습 후 이 기준을 넘는지 다시
    확인해 모델이 퇴화하지 않았는지 감시한다.

    세 RMSE는 호출부가 동일한 held-out(val) 구간·동일한 실현 변동성 정의(|로그수익률x100|)로
    미리 계산해서 넘겨야 한다 — 이 함수 자체는 비교만 한다."""
    beats_garch = hybrid_rmse < garch_rmse
    beats_sma = hybrid_rmse < sma_rmse
    return {
        "hybrid_rmse": hybrid_rmse, "garch_rmse": garch_rmse, "sma_rmse": sma_rmse,
        "beats_garch": beats_garch, "beats_sma": beats_sma,
        "passed": beats_garch and beats_sma,
    }


# ── 모델 체크포인트 저장/로드 (신규, 2026-09-06 — Task T 자동화 착수) ──────────────
#
# 이전에는 어떤 스크립트도 torch.save를 호출하지 않았다(best_state를 메모리에만 들고
# 있다가 평가 후 버림 — 가격예측/checkpoints/에 진단용 png 3개만 있고 .pt 파일이 전혀
# 없었음을 실측 확인). 버전 디렉터리(날짜+시각 기반)를 매번 새로 만들어 이전 버전을
# 덮어쓰지 않는다(롤백 가능) — latest.txt 텍스트 파일로 최신 버전을 가리킨다
# (심볼릭 링크는 Windows 권한 문제가 있을 수 있어 피함 — huggingface_hub 캐시가 이
# 프로젝트 환경에서 심링크 미지원 경고를 냈던 선례 참고).

def save_checkpoint(model, scaler, feature_cols, model_kwargs, checkpoint_dir, version=None,
                     extra_meta=None, model_class_name=None):
    """model.state_dict() + FeatureScaler + 피처 목록 + 모델 하이퍼파라미터 + 메타데이터를
    버전 디렉터리(checkpoint_dir/<version>/)에 저장한다. version 생략 시 UTC 날짜+시각
    (YYYYMMDD_HHMMSS)으로 자동 태깅 — 같은 날 여러 번 실행해도 초 단위까지 있어 겹치지 않는다.
    model_class_name 생략 시 model의 실제 클래스 이름을 자동으로 기록한다(2026-09-06 확장 —
    PooledTransformerRegressor 지원, 기존 호출부는 그대로 TransformerRegressor로 기록됨).
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


def load_checkpoint(checkpoint_dir, version=None, device="cpu", model_class=None):
    """version 생략 시 latest.txt가 가리키는 최신 버전을 로드한다. model_class 생략 시
    meta.json에 저장된 "model_class" 문자열로 클래스를 해석한다(2026-09-06 확장, 없으면
    TransformerRegressor로 간주 — 필드 추가 전 체크포인트와의 하위 호환).
    반환: (model, scaler, meta) — model은 eval() 상태로, device로 이동돼 있다."""
    if version is None:
        latest_path = os.path.join(checkpoint_dir, "latest.txt")
        if not os.path.exists(latest_path):
            raise FileNotFoundError(f"{latest_path} 없음 — 저장된 체크포인트가 없습니다")
        with open(latest_path, "r", encoding="utf-8") as f:
            version = f.read().strip()

    version_dir = os.path.join(checkpoint_dir, version)
    meta_path = os.path.join(version_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} 없음 — version={version!r} 체크포인트가 없습니다")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    cls = model_class
    if cls is None:
        class_name = meta.get("model_class", "TransformerRegressor")
        if class_name not in _MODEL_CLASSES:
            raise ValueError(f"알 수 없는 model_class: {class_name!r} (지원: {list(_MODEL_CLASSES)})")
        cls = _MODEL_CLASSES[class_name]

    n_features = len(meta["feature_cols"])
    model = cls(n_features=n_features, **meta["model_kwargs"])
    model.load_state_dict(torch.load(os.path.join(version_dir, "model.pt"), map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    scaler = FeatureScaler.load(os.path.join(version_dir, "scaler.npz"))
    return model, scaler, meta


# ── 무인 실행 예외 격리 (신규, 2026-09-06) ────────────────────────────────────
#
# 다른 트랙(주가_일일수집.py 등)의 원칙과 동일: 예외를 print만 하고 삼키지 않는다.
# 종목 단위로 격리해 하나가 실패해도 나머지가 계속 진행되게 하되, 실패는 콘솔 + 로그
# 파일에 traceback까지 명확히 남긴다(그냥 죽지도, 조용히 삼키지도 않음).

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


# ── Pooled(다종목) 학습/평가 루프 (신규, 2026-09-06) ──────────────────────────
#
# PooledTransformerRegressor.forward(x, ticker_ids)는 기존 TransformerRegressor.forward(x)와
# 시그니처가 달라 위 run_epoch/evaluate_predictions/train_transformer를 그대로 쓸 수 없다.
# 별도 함수로 분리했다 — 로직은 "배치에 ticker_ids가 하나 더 있고 모델 호출 시 함께 넘긴다"는
# 점만 다르고 나머지(스모크 테스트, early stopping, history)는 완전히 동일하다.

def make_pooled_loader(X, y, ticker_ids, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(ticker_ids))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def run_pooled_epoch(model, loader, criterion, optimizer, device, train_mode):
    model.train(train_mode)
    total_loss, n = 0.0, 0
    with torch.set_grad_enabled(train_mode):
        for xb, yb, tid in loader:
            xb, yb, tid = xb.to(device), yb.to(device), tid.to(device)
            pred = model(xb, tid)
            loss = criterion(pred, yb)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(xb)
            n += len(xb)
    return total_loss / n


def evaluate_pooled_predictions(model, loader, device):
    model.eval()
    preds, actuals, tids = [], [], []
    with torch.no_grad():
        for xb, yb, tid in loader:
            xb, tid = xb.to(device), tid.to(device)
            preds.append(model(xb, tid).cpu().numpy())
            actuals.append(yb.numpy())
            tids.append(tid.cpu().numpy())
    return np.concatenate(preds), np.concatenate(actuals), np.concatenate(tids)


def train_pooled_transformer(X_train, y_train, tid_train, X_val, y_val, tid_val,
                              n_features, lookback, num_stocks, embedding_dim,
                              d_model, nhead, num_layers, dim_feedforward, dropout,
                              batch_size, lr, weight_decay, smoke_epochs, max_epochs, patience,
                              device, seed=42, verbose=True, label=""):
    """train_transformer()의 pooled 버전 — PooledTransformerRegressor(ticker_ids 입력)용.
    test는 절대 건드리지 않는다 — train/val 로더만 받아서 학습·검증한다."""
    set_seed(seed)
    model = PooledTransformerRegressor(n_features, lookback, num_stocks, embedding_dim,
                                        d_model, nhead, num_layers, dim_feedforward, dropout).to(device)

    train_loader = make_pooled_loader(X_train, y_train, tid_train, batch_size, shuffle=True)
    val_loader = make_pooled_loader(X_val, y_val, tid_val, batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    prefix = f"[{label}] " if label else ""

    if verbose:
        print(f"{prefix}스모크 테스트 ({smoke_epochs} epoch)")
    for epoch in range(1, smoke_epochs + 1):
        train_loss = run_pooled_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss = run_pooled_epoch(model, val_loader, criterion, optimizer, device, False)
        history.append((epoch, train_loss, val_loss))
        finite = np.isfinite(train_loss) and np.isfinite(val_loss)
        if verbose:
            print(f"{prefix}  epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} finite={finite}")
        if not finite:
            raise RuntimeError(f"{label}: 스모크 테스트 중 loss가 NaN/Inf")

    if verbose:
        print(f"{prefix}본 학습 (최대 {max_epochs} epoch, patience={patience})")
    best_val_loss = float("inf")
    best_state = None
    best_epoch = smoke_epochs
    patience_counter = 0

    for epoch in range(smoke_epochs + 1, smoke_epochs + 1 + max_epochs):
        train_loss = run_pooled_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss = run_pooled_epoch(model, val_loader, criterion, optimizer, device, False)
        history.append((epoch, train_loss, val_loss))
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
            marker = " *best*"
        else:
            patience_counter += 1
        if verbose:
            print(f"{prefix}  epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}{marker}")
        if patience_counter >= patience:
            if verbose:
                print(f"{prefix}  early stopping (patience={patience}, best epoch={best_epoch})")
            break

    model.load_state_dict(best_state)
    last_epoch, last_train_loss, last_val_loss = history[-1]

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "last_epoch": last_epoch,
        "last_train_loss": last_train_loss,
        "last_val_loss": last_val_loss,
        "val_loader": val_loader,
        "overfit_ratio_at_end": last_val_loss / last_train_loss if last_train_loss > 0 else float("inf"),
    }
