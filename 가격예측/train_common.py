# train_common.py — 학습 단계: 운영 pooled LSTM 변동성 모델의 학습 설정(TrainConfig)과 학습 루프(train_model).
#
# 2026-10-17 정리: Transformer 계열(TransformerRegressor/PooledTransformerRegressor 학습), 1세대 단일 종목
# 방향 예측 유틸(train_transformer, 방향 게이트, confusion 등), 급변 구간 손실 실험(가중 MSE·pinball),
# load_checkpoint를 지웠다(git 태그 pre-cleanup 시점에 있음). 배포 게이트·val 예측 평가는 evaluate.py로,
# 체크포인트 저장·예외 격리(run_isolated)는 store.py로 옮겼다.

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from 가격예측.model import PooledLSTMRegressor


@dataclass(frozen=True)
class TrainConfig:
    """운영 LSTM 학습 하이퍼파라미터(2026-10-17, 모듈 상수에서 설정 객체로). 기본값 = 2026-09-28 이후 운영값
    (가격예측_변동성_공통의 옛 D_MODEL/NUM_LAYERS/DROPOUT/EMBEDDING_DIM/BATCH_SIZE/LR/WEIGHT_DECAY/
    SMOKE_EPOCHS/MAX_EPOCHS/PATIENCE/SEED와 같은 값). 튜닝은 이 객체를 바꿔 train_model에 넘기는 방식으로 끼어든다
    (튜닝 검증은 train 구간 안의 expanding-window 폴드로만 — CLAUDE.md "하이퍼파라미터 월간 튜닝")."""
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Pooled(다종목) 학습/평가 루프 (2026-09-06) — 배치마다 (x, y, ticker_ids), 모델 호출은 model(x, ticker_ids) ──

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


# ── 학습 루프 (2026-10-17 단일화: 옛 train_pooled_lstm과 같은 연산 순서) ──────────────────────
# 스모크 테스트 -> 본 학습(Adam+MSE, early stopping) -> best_state 복원.

def train_model(X_train, y_train, tid_train, X_val, y_val, tid_val, num_stocks, lookback, config, device,
                verbose=True, label=""):
    """PooledLSTMRegressor를 config(TrainConfig)로 학습한다. test는 절대 건드리지 않는다 — train/val만 받는다.
    반환: model, history, best_epoch, best_val_loss, last_epoch, last_train_loss, last_val_loss, val_loader,
    overfit_ratio_at_end."""
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
