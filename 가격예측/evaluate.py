# evaluate.py — 평가 단계: 학습된 모델의 val 구간 예측, 종목평균 RMSE, baseline 3종과 비교한 배포 게이트.
# 2026-10-17 정리에서 train_common.py(게이트·예측 평가)와 가격예측_변동성_공통.train_daily_pooled_model
# (종목평균 RMSE 집계)에서 옮겼다. 계산식은 그대로다.

import numpy as np
import torch

from 가격예측.baselines import rmse_mae


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


def tickermean_rmse(preds, actuals, tids):
    """종목별 RMSE의 단순 평균(2026-09-27 게이트 집계 버그 수정). baseline 3종이 "종목별 val RMSE의 단순
    평균"이므로 하이브리드도 같은 방식으로 맞춘다 — 전체 val 행을 한데 모은(pooled) RMSE와 비교하면
    종목별 RMSE 평균 <= pooled RMSE(젠센 부등식)라 baseline에 구조적으로 유리했다(2026-09-12 첫 운영
    실행부터 게이트가 계속 미통과한 원인, 경위는 CLAUDE.md)."""
    return float(np.mean([
        rmse_mae(preds[tids == tid], actuals[tids == tid])[0]
        for tid in np.unique(tids)
    ]))


def passes_deployment_gate_volatility(hybrid_rmse, garch_rmse, sma_rmse, parkinson_rmse):
    """하이브리드(GARCH σ 피처 + LSTM) 변동성 예측 배포 게이트 — 하이브리드 RMSE가 GARCH(1,1)·
    SMA20·Parkinson-SMA20 baseline **셋 다**보다 낮아야(개선돼야) 통과한다(2026-09-07 확정).

    네 RMSE는 호출부가 동일한 held-out(val) 구간·동일한 실현 변동성 정의(|로그수익률x100|)로
    미리 계산해서 넘겨야 한다 — 이 함수 자체는 비교만 한다."""
    beats_garch = hybrid_rmse < garch_rmse
    beats_sma = hybrid_rmse < sma_rmse
    beats_parkinson = hybrid_rmse < parkinson_rmse
    return {
        "hybrid_rmse": hybrid_rmse, "garch_rmse": garch_rmse, "sma_rmse": sma_rmse,
        "parkinson_rmse": parkinson_rmse,
        "beats_garch": beats_garch, "beats_sma": beats_sma, "beats_parkinson": beats_parkinson,
        "passed": beats_garch and beats_sma and beats_parkinson,
    }


def evaluate_gate(model, val_loader, device, baselines_by_ticker):
    """val 구간 하이브리드 종목평균 RMSE와 baseline 3종(종목별 val RMSE 평균)을 비교해 게이트 dict를 돌려준다.
    baselines_by_ticker: 가격예측_변동성_공통.compute_ticker_baselines 결과(종목 순서 그대로)."""
    val_preds, val_actuals, val_tids = evaluate_pooled_predictions(model, val_loader, device)
    hybrid_val_rmse = tickermean_rmse(val_preds, val_actuals, val_tids)
    avg_garch_rmse = float(np.mean([b["val_rmse"]["garch"] for b in baselines_by_ticker.values()]))
    avg_sma_rmse = float(np.mean([b["val_rmse"]["sma20"] for b in baselines_by_ticker.values()]))
    avg_park_rmse = float(np.mean([b["val_rmse"]["parkinson"] for b in baselines_by_ticker.values()]))
    return passes_deployment_gate_volatility(hybrid_val_rmse, avg_garch_rmse, avg_sma_rmse, avg_park_rmse)
