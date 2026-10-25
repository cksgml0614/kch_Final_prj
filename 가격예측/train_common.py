# train_common.py — 운영 pooled LSTM 변동성 모델의 학습 설정(TrainConfig)과 학습 루프(train_model, train_fixed_epochs).
# 파이프라인 단계: 학습. 주요 진입 함수: train_model(게이트용, val early stopping), train_fixed_epochs(전체 재학습용).
# 흐름: 스모크 테스트(loss 유한성 확인) -> 본 학습(Adam+MSE, early stopping) -> best_state 복원.
# 배경: 결정기록.md의 "게이트는 train→val 모델로 판정하고 서빙은 전체 재학습 모델로 한다"

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from 가격예측.model import PooledLSTMRegressor


@dataclass(frozen=True)
class TrainConfig:
    """
    설명: 운영 LSTM 학습 하이퍼파라미터 묶음. 기본값이 운영값이다.
    주의 사항: 주간 튜닝은 이 객체를 바꿔 train_model에 넘긴다(검증 구간 사용과 채택 규칙은 결정기록.md "주간 튜닝").
    입력 타입->반환 타입: (필드 키워드 인자) -> TrainConfig
    """
    hidden_size: int = 32
    num_layers: int = 2
    dropout: float = 0.3
    embedding_dim: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    smoke_epochs: int = 2
    max_epochs: int = 60
    patience: int = 8
    seed: int = 42


def set_seed(seed):
    """
    설명: random·numpy·torch(CPU·CUDA)의 시드를 같은 값으로 고정한다.
    주의 사항: 전역 난수 상태를 바꾼다.
    입력 타입->반환 타입: (int) -> None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Pooled(다종목) 배치: (x, y, ticker_ids), 모델 호출은 model(x, ticker_ids) ──

def make_pooled_loader(X, y, ticker_ids, batch_size, shuffle):
    """
    설명: 배열 3개를 (x, y, ticker_id) 배치를 내는 DataLoader로 묶는다.
    주의 사항: X·y는 float32, ticker_ids는 정수 배열이어야 한다.
    입력 타입->반환 타입: (np.ndarray, np.ndarray, np.ndarray, int, bool) -> DataLoader
    """
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(ticker_ids))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def run_pooled_epoch(model, loader, criterion, optimizer, device, train_mode):
    """
    설명: loader를 한 바퀴 돌며 손실 평균을 구하고, train_mode면 역전파로 가중치를 갱신한다.
    주의 사항: model의 train/eval 모드를 바꾼다. 평균은 표본 수 가중이다.
    입력 타입->반환 타입: (nn.Module, DataLoader, nn.Module, Optimizer, torch.device, bool) -> float
    """
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


# ── 학습 루프 ─────────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, tid_train, X_val, y_val, tid_val, num_stocks, lookback, config, device,
                verbose=True, label=""):
    """
    설명: PooledLSTMRegressor를 config로 학습하고 val 손실이 가장 낮았던 가중치로 되돌린 결과 dict를 돌려준다.
    주의 사항: train/val만 받는다(test 금지). 스모크 단계 손실이 NaN/Inf면 RuntimeError. 시드를 고정한다.
    입력 타입->반환 타입: (np.ndarray x6, int, int, TrainConfig, torch.device, bool, str) -> dict
    """
    smoke_epochs, max_epochs, patience = config.smoke_epochs, config.max_epochs, config.patience
    set_seed(config.seed)
    model = PooledLSTMRegressor(X_train.shape[2], lookback, num_stocks, config.embedding_dim,
                                hidden_size=config.hidden_size, num_layers=config.num_layers,
                                dropout=config.dropout).to(device)

    train_loader = make_pooled_loader(X_train, y_train, tid_train, config.batch_size, shuffle=True)
    val_loader = make_pooled_loader(X_val, y_val, tid_val, config.batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

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


def train_fixed_epochs(X, y, tid, num_stocks, lookback, config, device, n_epochs, verbose=True, label=""):
    """
    설명: 검증 구간 없이 PooledLSTMRegressor를 정확히 n_epochs epoch 학습하고 마지막 가중치를 돌려준다(전체 재학습용).
    주의 사항: 시드·모델 생성·DataLoader·optimizer는 train_model과 같은 순서다. epoch 손실이 NaN/Inf면 RuntimeError.
    입력 타입->반환 타입: (np.ndarray x3, int, int, TrainConfig, torch.device, int, bool, str) -> dict
    """
    if n_epochs < 1:
        raise ValueError(f"{label}: n_epochs는 1 이상이어야 한다({n_epochs})")
    set_seed(config.seed)
    model = PooledLSTMRegressor(X.shape[2], lookback, num_stocks, config.embedding_dim,
                                hidden_size=config.hidden_size, num_layers=config.num_layers,
                                dropout=config.dropout).to(device)
    loader = make_pooled_loader(X, y, tid, config.batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    prefix = f"[{label}] " if label else ""
    if verbose:
        print(f"{prefix}전체 재학습 ({n_epochs} epoch 고정, {len(X)}행)")
    history = []
    for epoch in range(1, n_epochs + 1):
        train_loss = run_pooled_epoch(model, loader, criterion, optimizer, device, True)
        history.append((epoch, train_loss))
        if verbose:
            print(f"{prefix}  epoch {epoch}: train_loss={train_loss:.6f}")
        if not np.isfinite(train_loss):
            raise RuntimeError(f"{label}: 전체 재학습 epoch {epoch}의 loss가 NaN/Inf")
    return {"model": model, "history": history, "epochs": n_epochs}
