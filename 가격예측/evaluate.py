# evaluate.py — 학습된 모델의 val 구간 예측, 종목평균 RMSE, baseline 3종과 비교한 배포 게이트를 계산한다.
# 파이프라인 단계: 평가. 주요 진입 함수: evaluate_gate.
# 배경: 결정기록.md의 "배포 게이트는 종목평균 RMSE로 baseline 3종을 모두 이겨야 통과"

import numpy as np
import torch

from 가격예측.baselines import rmse_mae


def evaluate_pooled_predictions(model, loader, device):
    """
    설명: loader 전체에 대해 모델 예측값·실제값·종목 id를 모아 돌려준다.
    주의 사항: model.eval()로 바꾼다(호출 뒤 학습을 이어가려면 train()으로 되돌려야 한다).
    입력 타입->반환 타입: (nn.Module, DataLoader, torch.device) -> (np.ndarray, np.ndarray, np.ndarray)
    """
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
    """
    설명: 종목별 RMSE를 구해 단순 평균한다.
    주의 사항: baseline 3종과 같은 집계 방식이어야 한다(전체 행을 모은 RMSE와 섞어 비교하지 말 것).
    입력 타입->반환 타입: (np.ndarray, np.ndarray, np.ndarray) -> float
    """
    return float(np.mean([
        rmse_mae(preds[tids == tid], actuals[tids == tid])[0]
        for tid in np.unique(tids)
    ]))


def passes_deployment_gate_volatility(hybrid_rmse, garch_rmse, sma_rmse, parkinson_rmse):
    """
    설명: 하이브리드 RMSE가 GARCH·SMA20·Parkinson-SMA20 RMSE 셋 모두보다 낮은지 판정해 게이트 dict를 만든다.
    주의 사항: 네 값은 같은 val 구간·같은 실현 변동성 정의로 미리 계산해 넘겨야 한다(이 함수는 비교만).
    입력 타입->반환 타입: (float, float, float, float) -> dict
    """
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
    """
    설명: val 구간 하이브리드 종목평균 RMSE와 baseline 3종의 종목별 val RMSE 평균을 비교해 게이트 dict를 돌려준다.
    주의 사항: baselines_by_ticker는 baselines.compute_ticker_baselines 결과(종목별 dict)여야 한다.
    입력 타입->반환 타입: (nn.Module, DataLoader, torch.device, dict) -> dict
    """
    val_preds, val_actuals, val_tids = evaluate_pooled_predictions(model, val_loader, device)
    hybrid_val_rmse = tickermean_rmse(val_preds, val_actuals, val_tids)
    avg_garch_rmse = float(np.mean([b["val_rmse"]["garch"] for b in baselines_by_ticker.values()]))
    avg_sma_rmse = float(np.mean([b["val_rmse"]["sma20"] for b in baselines_by_ticker.values()]))
    avg_park_rmse = float(np.mean([b["val_rmse"]["parkinson"] for b in baselines_by_ticker.values()]))
    return passes_deployment_gate_volatility(hybrid_val_rmse, avg_garch_rmse, avg_sma_rmse, avg_park_rmse)
