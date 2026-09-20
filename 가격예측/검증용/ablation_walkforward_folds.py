# ablation_walkforward_folds.py — Variant A(macro-light) walk-forward 재검증 (2026-09-20)
#
# ablation_feature_pruning.py의 70/15/15 단일 split(fold3, train_end=2024-10-07,
# val_end=2025-09-25) 결과에서 variant A(거시 3종 제거)가 baseline 대비 개선폭이 작았다
# (val RMSE -0.22%, val MAE -0.65%) — 이게 이 특정 split의 우연인지 확인하기 위해 train+val
# 구간(2020-01-01~2025-09-25) 안에서만 expanding window로 fold를 2개 더 추가한다.
#
# ⚠️ 데이터 유출 방지: 세 fold 전부 test(2025-09-26~오늘)는 전혀 사용하지 않는다 — fold1/2는
# train+val 구간 안에서만 경계를 나눈 것이라 test가 어느 fold의 train/val에도 섞이지 않는다.
# fold3은 재학습하지 않고 ablation_feature_pruning.py 실행 결과 값을 그대로 재사용한다.
#
# fold 경계(100종목 공유 캘린더 기준 가장 가까운 거래일, 굳이 12/31에 정확히 맞추지 않음):
#   fold1: train=2020-01-01~2022-12-29(대표종목 기준 실제 마지막 거래일), val=2022-12-30~2023-12-28
#   fold2: train=2020-01-01~2023-12-28,                                  val=2023-12-29~2024-09-30
#   fold3(재사용): train=2020-01-01~2024-10-07, val=2024-10-08~2025-09-25

import sys
from datetime import date

import numpy as np
import pandas as pd
import torch

from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
from 가격예측.가격예측_변동성_공통 import (
    BATCH_SIZE, D_MODEL, DIM_FEEDFORWARD, DROPOUT, EMBEDDING_DIM, LOOKBACK, LR,
    MAX_EPOCHS, NHEAD, NUM_LAYERS, PATIENCE, SEED, SMOKE_EPOCHS, WEIGHT_DECAY,
)
from 가격예측.garch_baseline import rmse_mae
from 가격예측.검증용.garch_baseline_check import compute_full_period_sigma
from 가격예측.pooled_dataset import build_pooled_sequences_with_split, compute_global_split_dates
from 가격예측.sequence_dataset import FeatureScaler
from 가격예측.split_dataset import build_merged_dataset_v2, build_merged_dataset_v2_volatility_hybrid
from 가격예측.train_common import evaluate_pooled_predictions, make_pooled_loader, train_pooled_transformer

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EXCLUDE_A = ["ECOS_722Y001_0101000", "bond_spread_10y_3y", "KOSDAQ"]

# fold3 — ablation_feature_pruning.py 실행 결과(2026-09-20) 그대로 재사용, 재학습 안 함.
FOLD3_RESULT = {
    "baseline": {"val_rmse": 2.077531, "val_mae": 1.396861},
    "variant_A": {"val_rmse": 2.073014, "val_mae": 1.387815},
}


def precompute_garch(tickers, start_date, end_date, train_end):
    """fold마다 train_end가 다르므로 GARCH도 fold별로 다시 적합해야 한다(leak-free 원칙 —
    이 fold의 train_end까지 데이터로만 파라미터를 추정). 벤치마크 결과(2026-09-20)
    100종목 직렬 적합 자체가 수 초대라 병렬화 이득이 없어 직렬로 둔다(garch_baseline_check.
    precompute_garch_parallel은 프로세스 spawn 오버헤드가 더 커서 이 문제 크기에는 역효과)."""
    cache = {}
    skipped = []
    for i, ticker in enumerate(tickers, 1):
        try:
            sigma, params = compute_full_period_sigma(ticker, start_date, end_date, train_end)
            cache[ticker] = (sigma, params)
        except ValueError as e:
            # 상장일이 이 fold의 train_end보다 늦은 종목(예: 443060/278470, 각각 2024-05-08/
            # 2024-02-27 상장 — fold1/2의 train_end보다 나중) — train_returns가 비어
            # arch_model.fit()이 "first_obs and last_obs produce an empty array"로 실패한다.
            # 이 fold에서만 제외한다(다른 fold/원본 100종목 검증에는 영향 없음).
            skipped.append(ticker)
            print(f"    ⚠️ {ticker}: GARCH 적합 실패(이 fold의 train 구간에 데이터 없음, "
                  f"상장일이 fold train_end 이후로 추정) — 이 fold에서 제외: {e}", flush=True)
        if i % 20 == 0 or i == len(tickers):
            print(f"    GARCH sigma 사전계산 {i}/{len(tickers)}", flush=True)
    if skipped:
        print(f"    이 fold에서 제외된 종목 {len(skipped)}개: {skipped}", flush=True)
    return cache


def make_build_fn(train_end, garch_cache, exclude_cols):
    def build_fn(ticker, s, e, precomputed_indicators=None, true_kospi=None):
        # precomputed_indicators/true_kospi(2026-09-20): pooled_dataset.build_pooled_sequences_
        # with_split이 이 시그니처를 보고 거시지표 캐시를 종목 루프 시작 전에 한 번만 만들어
        # 넘긴다 — 성능 최적화일 뿐 조인/계산 로직은 동일(값 변경 없음 검증 완료).
        sigma, params = garch_cache[ticker]
        merged, meta = build_merged_dataset_v2_volatility_hybrid(
            ticker, s, e, train_end, precomputed_sigma=sigma, garch_params=params,
            precomputed_indicators=precomputed_indicators, true_kospi=true_kospi,
        )
        if exclude_cols:
            merged = merged.drop(columns=exclude_cols)
        return merged, meta
    return build_fn


def run_one(name, exclude_cols, tickers, start_date, end_date, train_end, val_end, garch_cache, device):
    build_fn = make_build_fn(train_end, garch_cache, exclude_cols)
    splits, feature_cols, ticker_to_id = build_pooled_sequences_with_split(
        tickers, start_date, end_date, train_end, val_end, lookback=LOOKBACK, build_fn=build_fn,
    )
    X_train, y_train, tid_train, _, _ = splits["train"]
    X_val, y_val, tid_val, _, _ = splits["val"]
    # ⚠️ splits["test"](val_end 이후, 즉 다음 fold + 진짜 test 전부 포함)는 여기서 끝 —
    # 어떤 변수에도 담지 않는다.

    print(f"  [{name}] feature_cols({len(feature_cols)}개)  X_train {X_train.shape}  X_val {X_val.shape}")

    scaler = FeatureScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    result = train_pooled_transformer(
        X_train_s, y_train, tid_train, X_val_s, y_val, tid_val,
        len(feature_cols), LOOKBACK, len(tickers), EMBEDDING_DIM,
        D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT,
        BATCH_SIZE, LR, WEIGHT_DECAY, SMOKE_EPOCHS, MAX_EPOCHS, PATIENCE,
        device, seed=SEED, verbose=False, label=name,
    )
    val_loader = make_pooled_loader(X_val_s, y_val, tid_val, BATCH_SIZE, shuffle=False)
    preds, actuals, _ = evaluate_pooled_predictions(result["model"], val_loader, device)
    val_rmse, val_mae = rmse_mae(preds, actuals)
    print(f"  [{name}] val RMSE={val_rmse:.6f}  val MAE={val_mae:.6f}  best_epoch={result['best_epoch']}")
    return {"val_rmse": val_rmse, "val_mae": val_mae, "best_epoch": result["best_epoch"]}


def run_fold(fold_name, tickers, start_date, end_date, train_end, val_end, device):
    print("\n" + "=" * 100)
    print(f"=== {fold_name}: train=[{start_date}~{train_end.date()}]  val=({train_end.date()}~{val_end.date()}] ===")
    print("=" * 100)
    print("  GARCH sigma 사전계산 중(이 fold의 train_end 기준)...")
    garch_cache = precompute_garch(tickers, start_date, end_date, train_end)
    fold_tickers = [t for t in tickers if t in garch_cache]
    if len(fold_tickers) != len(tickers):
        print(f"  이 fold는 {len(fold_tickers)}/{len(tickers)}종목으로 진행 "
          f"(제외 {len(tickers) - len(fold_tickers)}개 — baseline/variant A 둘 다 동일 모집단)", flush=True)

    baseline = run_one(f"{fold_name}-baseline", [], fold_tickers, start_date, end_date,
                        train_end, val_end, garch_cache, device)
    variant_a = run_one(f"{fold_name}-variantA", EXCLUDE_A, fold_tickers, start_date, end_date,
                         train_end, val_end, garch_cache, device)
    return baseline, variant_a


def main():
    tickers = ACTIVE_TICKERS
    end_date = date.today().isoformat()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ref, _ = build_merged_dataset_v2(tickers[0], STOCK_INITIAL_LOAD_START, end_date)
    idx = ref.index
    fold3_train_end, fold3_val_end = compute_global_split_dates(idx)

    fold1_train_end = idx[idx <= pd.Timestamp("2022-12-31")].max()
    fold1_val_end = idx[idx <= pd.Timestamp("2023-12-31")].max()
    fold2_train_end = fold1_val_end
    fold2_val_end = idx[idx <= pd.Timestamp("2024-09-30")].max()

    print(f"fold1: train_end={fold1_train_end.date()}  val_end={fold1_val_end.date()}")
    print(f"fold2: train_end={fold2_train_end.date()}  val_end={fold2_val_end.date()}")
    print(f"fold3(재사용, 재학습 안 함): train_end={fold3_train_end.date()}  val_end={fold3_val_end.date()}")
    assert fold3_train_end.date().isoformat() == "2024-10-07", "fold3 경계가 이전 실행과 달라짐 — 재사용 전제 위반"

    results = {}
    results["fold1"] = run_fold("fold1", tickers, STOCK_INITIAL_LOAD_START, end_date,
                                 fold1_train_end, fold1_val_end, device)
    results["fold2"] = run_fold("fold2", tickers, STOCK_INITIAL_LOAD_START, end_date,
                                 fold2_train_end, fold2_val_end, device)
    results["fold3"] = (FOLD3_RESULT["baseline"], FOLD3_RESULT["variant_A"])

    print("\n" + "=" * 100)
    print("=== walk-forward 3-fold 요약 (val만 — test는 이번 실행에서 전혀 평가하지 않았음) ===")
    print("=" * 100)
    print(f"{'fold':<8}{'baseline RMSE':>15}{'variantA RMSE':>15}{'baseline MAE':>14}{'variantA MAE':>14}{'A<=baseline':>13}")
    directions = []
    for fname, (b, a) in results.items():
        win = a["val_rmse"] <= b["val_rmse"]
        directions.append(win)
        print(f"{fname:<8}{b['val_rmse']:>15.6f}{a['val_rmse']:>15.6f}{b['val_mae']:>14.6f}{a['val_mae']:>14.6f}{str(win):>13}")

    a_wins = sum(directions)
    print(f"\nVariant A가 baseline과 같거나 나은 fold: {a_wins}/3")

    if a_wins >= 2:
        verdict = "채택 후보 유지"
    else:
        verdict = "기각"
    print(f"1차 판정(다수결, >=2/3): {verdict}")

    if len(set(directions)) > 1:
        print("⚠️ fold마다 우세 방향이 일치하지 않음(3-0 완승이 아님) — 단순 우연으로 치부하지 "
              "않고 '국면 의존적 효과'로 함께 기록한다. 어느 쪽이 항상 맞다고 단정하지 않음.")
    else:
        print("fold 3개 전부 방향이 일치함 — 국면 의존적 신호 없음(일관된 결과).")

    print("\n(주: test(2025-09-26~오늘)는 이번 실행에서 어떤 fold/variant도 평가하지 않았다. "
          "최종 판단 후 test 확인은 별도 승인을 받아 진행할 것.)")


if __name__ == "__main__":
    main()
