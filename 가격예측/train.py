# train.py — Task T-1 체크포인트 4
# 베이스라인 Transformer 학습: 1) 스모크 테스트 -> 2) 본 학습(loss curve) -> 3) test 평가
# (RMSE/MAE/방향성 정확도) -> 4) 다수 클래스 baseline 비교.
# 스트레스 구간 홀드아웃 3자 비교는 체크포인트 5에서 별도 진행 (여기서는 하지 않음).

import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from constants import STRESS_PERIOD_START
from 가격예측.model import TransformerRegressor
from 가격예측.sequence_dataset import FeatureScaler, build_sequences, split_sequences_by_date
from 가격예측.split_dataset import build_merged_dataset, split_normal_regime

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TICKER = "005930"
START = "2020-01-02"
END = "2026-07-31"
LOOKBACK = 20
SEED = 42

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


if __name__ == "__main__":
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    merged, meta = build_merged_dataset(TICKER, START, END)
    train_df, val_df, test_df, normal, stress = split_normal_regime(merged, STRESS_PERIOD_START)
    feature_cols = [c for c in merged.columns if c != "target"]

    print("\n=== 데이터 ===")
    print(f"정상 레짐 {len(normal)}행, feature_cols({len(feature_cols)}개): {feature_cols}")
    print(f"train {len(train_df)} / val {len(val_df)} / test {len(test_df)} (날짜 기준, 시퀀스화 전)")

    X_all, y_all, dates_all = build_sequences(normal, feature_cols, LOOKBACK)
    splits = split_sequences_by_date(X_all, y_all, dates_all, train_df.index.max(), val_df.index.max())
    X_train, y_train, d_train = splits["train"]
    X_val, y_val, d_val = splits["val"]
    X_test, y_test, d_test = splits["test"]

    print(f"\n=== 시퀀스 shape (lookback={LOOKBACK}) ===")
    print(f"X_train {X_train.shape}  X_val {X_val.shape}  X_test {X_test.shape}")
    print(f"train 시퀀스 날짜 범위: {str(d_train.min())[:10]} ~ {str(d_train.max())[:10]}")
    print(f"val   시퀀스 날짜 범위: {str(d_val.min())[:10]} ~ {str(d_val.max())[:10]}")
    print(f"test  시퀀스 날짜 범위: {str(d_test.min())[:10]} ~ {str(d_test.max())[:10]}")
    lost = len(train_df) - len(X_train)
    print(f"train 쪽에서 lookback 부족으로 제외된 시퀀스: {lost}개 (정상 레짐 맨 앞 {LOOKBACK - 1}행)")

    scaler = FeatureScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    n_features = X_train.shape[-1]
    model = TransformerRegressor(n_features, LOOKBACK, D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT).to(device)
    total_p, trainable_p = model.count_params()
    print("\n=== 모델 ===")
    print(f"d_model={D_MODEL} nhead={NHEAD} num_layers={NUM_LAYERS} dim_feedforward={DIM_FEEDFORWARD} dropout={DROPOUT}")
    print(f"파라미터 수: 총 {total_p:,} / 학습 가능 {trainable_p:,} "
          f"(train 시퀀스 {len(X_train)}개 대비 {trainable_p/len(X_train):.1f}x)")

    train_loader = make_loader(X_train_s, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val_s, y_val, BATCH_SIZE, shuffle=False)
    test_loader = make_loader(X_test_s, y_test, BATCH_SIZE, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # --- 1단계: 스모크 테스트 ---
    print(f"\n=== 1단계: 스모크 테스트 ({SMOKE_EPOCHS} epoch) ===")
    for epoch in range(1, SMOKE_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train_mode=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train_mode=False)
        finite = np.isfinite(train_loss) and np.isfinite(val_loss)
        print(f"  epoch {epoch}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  finite={finite}")
        if not finite:
            raise RuntimeError("스모크 테스트 중 loss가 NaN/Inf — 학습 중단")
    print("스모크 테스트 통과: shape 오류 없음, loss 유한값, 학습 루프 정상 동작 확인")

    # --- 2단계: 본 학습 (스모크에 이어서 계속, early stopping) ---
    print(f"\n=== 2단계: 본 학습 (최대 {MAX_EPOCHS} epoch, patience={PATIENCE}) ===")
    best_val_loss = float("inf")
    best_state = None
    best_epoch = SMOKE_EPOCHS
    patience_counter = 0

    for epoch in range(SMOKE_EPOCHS + 1, SMOKE_EPOCHS + 1 + MAX_EPOCHS):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train_mode=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train_mode=False)
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
            marker = " *best*"
        else:
            patience_counter += 1
        print(f"  epoch {epoch}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}{marker}")
        if patience_counter >= PATIENCE:
            print(f"  early stopping (patience={PATIENCE} 소진, best epoch={best_epoch})")
            break

    model.load_state_dict(best_state)
    print(f"best epoch: {best_epoch}, best val_loss(MSE): {best_val_loss:.6f}")

    # --- 3단계: test 평가 ---
    print("\n=== 3단계: test셋 평가 (정상 레짐만, 스트레스 구간 아님) ===")
    preds, actuals = evaluate_predictions(model, test_loader, device)
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    mae = float(np.mean(np.abs(preds - actuals)))
    model_dir_acc = directional_accuracy(preds, actuals)

    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"방향성 정확도(모델): {model_dir_acc:.4f} ({model_dir_acc*100:.2f}%)")

    print("\n=== 진단: test 실제 vs 예측 분포 ===")
    print(f"실제: mean={actuals.mean():+.4%} std={actuals.std():.4%} min={actuals.min():+.4%} max={actuals.max():+.4%}")
    print(f"예측: mean={preds.mean():+.4%} std={preds.std():.4%} min={preds.min():+.4%} max={preds.max():+.4%}")

    # --- 4단계: 다수 클래스(방향) baseline과 비교 ---
    print("\n=== 4단계: 방향성 정확도 baseline 비교 ===")
    label, base_acc = majority_baseline_accuracy(y_test)
    print(f"다수 클래스 baseline: {label}  정확도={base_acc:.4f} ({base_acc*100:.2f}%)")
    print(f"모델 방향성 정확도:   {model_dir_acc:.4f} ({model_dir_acc*100:.2f}%)")
    diff = model_dir_acc - base_acc
    print(f"차이(모델 - baseline): {diff:+.4f} ({diff*100:+.2f}%p)")

    print("\n=== test셋 예측 샘플 (앞 10개) ===")
    for i in range(min(10, len(d_test))):
        print(f"  {str(d_test[i])[:10]}  실제={actuals[i]:+.4%}  예측={preds[i]:+.4%}  "
              f"부호일치={np.sign(actuals[i]) == np.sign(preds[i])}")
