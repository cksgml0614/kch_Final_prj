# train_common.py — Task T-1 학습 루프 공통 유틸.
# train.py(체크포인트4)와 train_v2.py(피처 재설계 이후)가 동일 로직을 공유한다.

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from 가격예측.model import TransformerRegressor


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
