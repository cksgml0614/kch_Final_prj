# ablation_feature_pruning.py — 상관관계 진단(train-only) 기반 피처 제거 ablation (2026-09-20)
#
# 상관관계_진단_외국인순매수.py의 train-only 진단(다중공선성 + 피처-target 상관)에서 확인된
# 거시지표를 실제로 제거했을 때 pooled 하이브리드 모델의 val 성능이 개선되는지 확인한다.
# baseline(현재 운영 중인 15피처)과 두 variant(A: 거시 3종 제거, B: CPI/M2 중복 하나 제거)를
# 동일 train/val 분할 경계·동일 하이퍼파라미터로 재학습해 val RMSE/MAE만으로 채택 여부를
# 판단한다. 하이퍼파라미터는 가격예측_변동성_공통.py(운영 중인 모듈)에서 그대로 가져온다 —
# 셋 다 절대 다르게 건드리지 않는다.
#
# ⚠️ 데이터 유출 방지: build_pooled_sequences()는 항상 train/val/test 3-way를 반환하지만,
# 이 스크립트는 splits["test"]를 어떤 변수에도 대입하지 않고 절대 평가하지 않는다. 채택
# 여부는 val 성능만으로 결정한다 — test 확인은 채택안이 확정된 뒤 별도 승인을 받아 진행한다.

import sys
from datetime import date

import numpy as np
import torch

from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
from 가격예측.가격예측_변동성_공통 import (
    BATCH_SIZE, D_MODEL, DIM_FEEDFORWARD, DROPOUT, EMBEDDING_DIM, LOOKBACK, LR,
    MAX_EPOCHS, NHEAD, NUM_LAYERS, PATIENCE, SEED, SMOKE_EPOCHS, WEIGHT_DECAY,
)
from 가격예측.garch_baseline import rmse_mae
from 가격예측.검증용.garch_baseline_check import compute_full_period_sigma
from 가격예측.pooled_dataset import build_pooled_sequences, compute_global_split_dates
from 가격예측.sequence_dataset import FeatureScaler
from 가격예측.split_dataset import build_merged_dataset_v2, build_merged_dataset_v2_volatility_hybrid
from 가격예측.train_common import evaluate_pooled_predictions, make_pooled_loader, train_pooled_transformer

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# variant_A: 사람이 지시한 3개(722Y001/기준금리, bond_spread_10y_3y, KOSDAQ)를 그대로 쓴다.
# ⚠️ train-only로 다시 계산해보니 이 3개가 실제 "가장 약한 3개"는 아니었다 — train-only
# |target상관| 오름차순: ECOS_901Y009_0(0.0091) < ECOS_722Y001_0101000(0.0101) <
# bond_spread_10y_3y(0.0143) < USD_KRW(0.0157) < ECOS_161Y005_BBHS00(0.0295) <
# ECOS_901Y067_I16E(0.0355) < KOSPI(0.0504) < KOSDAQ(0.0619, 8개 중 가장 강함). 즉 KOSDAQ은
# train-only 기준으로는 거시 8종 중 target과 가장 강한 축에 속한다(그래도 절대값 자체는 여전히
# 약함). 사람이 명시적으로 지정한 조합이라 그대로 실행하되, 이 불일치는 보고 시 반드시 알린다.
#
# variant_B: CPI(ECOS_901Y009_0) vs M2(ECOS_161Y005_BBHS00) 중복(r=0.96, train-only 재확인 —
# 8월/전체기간 0.98과 같은 방향) 하나만 남기는 것 — 어느 쪽을 남길지는 train-only |target상관|이
# 더 큰 쪽을 남기기로 위임받았다. train-only: CPI |r|=0.0091 < M2 |r|=0.0295 → M2를 남기고
# CPI를 제거한다. (전체기간 수치로는 반대로 CPI가 더 강했다(+0.1201 vs +0.1076) — train-only로
# 다시 계산하며 순위가 뒤집힌 사례. 근거 수치는 실행 시 다시 출력해 재확인한다.)
VARIANTS = {
    "baseline": [],
    "variant_A_macro_light": ["ECOS_722Y001_0101000", "bond_spread_10y_3y", "KOSDAQ"],
    "variant_B_cpi_m2_dedup": ["ECOS_901Y009_0"],
}


def precompute_garch(tickers, start_date, end_date, train_end):
    """세 variant가 전부 garch_sigma를 그대로 쓰므로(제거 대상 아님), 종목당 한 번만 GARCH를
    적합해 캐시로 공유한다 — 안 하면 variant 3개 x 100종목 = 300회 적합(3배 낭비)."""
    cache = {}
    for i, ticker in enumerate(tickers, 1):
        sigma, params = compute_full_period_sigma(ticker, start_date, end_date, train_end)
        cache[ticker] = (sigma, params)
        if i % 20 == 0 or i == len(tickers):
            print(f"  GARCH sigma 사전계산 {i}/{len(tickers)}", flush=True)
    return cache


def make_build_fn(train_end, garch_cache, exclude_cols):
    def build_fn(ticker, s, e, precomputed_indicators=None, true_kospi=None):
        # precomputed_indicators/true_kospi(2026-09-20): pooled_dataset.build_pooled_sequences가
        # 이 시그니처를 보고 거시지표 캐시를 종목 루프 시작 전에 한 번만 만들어 넘긴다 —
        # 성능 최적화일 뿐 조인/계산 로직은 동일(값 변경 없음 검증 완료, CLAUDE.md 참고).
        sigma, params = garch_cache[ticker]
        merged, meta = build_merged_dataset_v2_volatility_hybrid(
            ticker, s, e, train_end, precomputed_sigma=sigma, garch_params=params,
            precomputed_indicators=precomputed_indicators, true_kospi=true_kospi,
        )
        if exclude_cols:
            merged = merged.drop(columns=exclude_cols)
        return merged, meta
    return build_fn


def run_variant(name, exclude_cols, tickers, start_date, end_date, train_end, garch_cache, device):
    print("\n" + "=" * 100)
    print(f"=== variant: {name}  (제거 피처: {exclude_cols or '없음(baseline)'}) ===")
    print("=" * 100)

    build_fn = make_build_fn(train_end, garch_cache, exclude_cols)
    splits, feature_cols, ticker_to_id, (train_end2, val_end2) = build_pooled_sequences(
        tickers, start_date, end_date, lookback=LOOKBACK, build_fn=build_fn,
    )
    assert train_end2 == train_end, "train_end가 variant 간 어긋남 — 공정 비교 조건 위반"

    X_train, y_train, tid_train, _, _ = splits["train"]
    X_val, y_val, tid_val, _, _ = splits["val"]
    # ⚠️ splits["test"]는 여기서 끝 — 어떤 변수에도 담지 않는다(위 파일 헤더 주석 참고).

    print(f"feature_cols({len(feature_cols)}개): {feature_cols}")
    print(f"시퀀스: X_train {X_train.shape}  X_val {X_val.shape}")

    scaler = FeatureScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    result = train_pooled_transformer(
        X_train_s, y_train, tid_train, X_val_s, y_val, tid_val,
        len(feature_cols), LOOKBACK, len(tickers), EMBEDDING_DIM,
        D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT,
        BATCH_SIZE, LR, WEIGHT_DECAY, SMOKE_EPOCHS, MAX_EPOCHS, PATIENCE,
        device, seed=SEED, verbose=True, label=name,
    )

    val_loader = make_pooled_loader(X_val_s, y_val, tid_val, BATCH_SIZE, shuffle=False)
    preds, actuals, _ = evaluate_pooled_predictions(result["model"], val_loader, device)
    val_rmse, val_mae = rmse_mae(preds, actuals)

    print(f"\n[{name}] val RMSE={val_rmse:.6f}  val MAE={val_mae:.6f}  "
          f"best_epoch={result['best_epoch']}  best_val_loss(MSE)={result['best_val_loss']:.6f}")

    return {"name": name, "n_features": len(feature_cols), "excluded": exclude_cols,
            "val_rmse": val_rmse, "val_mae": val_mae, "best_epoch": result["best_epoch"]}


def main():
    tickers = ACTIVE_TICKERS
    end_date = date.today().isoformat()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"하이퍼파라미터(가격예측_변동성_공통.py에서 그대로 가져옴, 세 variant 공통): "
          f"LOOKBACK={LOOKBACK} D_MODEL={D_MODEL} NHEAD={NHEAD} NUM_LAYERS={NUM_LAYERS} "
          f"DIM_FEEDFORWARD={DIM_FEEDFORWARD} DROPOUT={DROPOUT} EMBEDDING_DIM={EMBEDDING_DIM} "
          f"BATCH_SIZE={BATCH_SIZE} LR={LR} WEIGHT_DECAY={WEIGHT_DECAY} MAX_EPOCHS={MAX_EPOCHS} "
          f"PATIENCE={PATIENCE} SEED={SEED}")

    ref_merged, _ = build_merged_dataset_v2(tickers[0], STOCK_INITIAL_LOAD_START, end_date)
    train_end, val_end = compute_global_split_dates(ref_merged.index)
    print(f"전역 분할 경계: train_end={train_end.date()}  val_end={val_end.date()} "
          f"(그 이후 test는 이번 실험에서 다루지 않음)")

    print(f"\nGARCH sigma 사전계산({len(tickers)}종목, 세 variant가 공유)...")
    garch_cache = precompute_garch(tickers, STOCK_INITIAL_LOAD_START, end_date, train_end)

    results = []
    for name, exclude_cols in VARIANTS.items():
        r = run_variant(name, exclude_cols, tickers, STOCK_INITIAL_LOAD_START, end_date,
                         train_end, garch_cache, device)
        results.append(r)

    print("\n" + "=" * 100)
    print("=== ablation 결과 요약 (val만 — test는 이번 실행에서 전혀 평가하지 않았음) ===")
    print("=" * 100)
    print(f"{'variant':<28}{'#features':>10}{'val RMSE':>12}{'val MAE':>10}")
    for r in results:
        print(f"{r['name']:<28}{r['n_features']:>10}{r['val_rmse']:>12.6f}{r['val_mae']:>10.6f}")

    base = results[0]
    print(f"\nbaseline: val RMSE={base['val_rmse']:.6f}  val MAE={base['val_mae']:.6f}")
    for r in results[1:]:
        delta_rmse = r["val_rmse"] - base["val_rmse"]
        delta_mae = r["val_mae"] - base["val_mae"]
        verdict = "채택 후보(RMSE 개선/동일)" if r["val_rmse"] <= base["val_rmse"] else "폐기(RMSE 악화)"
        mae_note = "" if (delta_mae <= 0) == (delta_rmse <= 0) else "  ⚠️ RMSE/MAE 방향 불일치 — 판정 재검토 필요"
        print(f"{r['name']}: ΔRMSE={delta_rmse:+.6f}  ΔMAE={delta_mae:+.6f} -> {verdict}{mae_note}")

    print("\n(주: test는 이번 실행에서 어떤 variant도 평가하지 않았다 — 최종 채택안 확정 후 "
          "별도 승인을 받아 딱 한 번만 확인할 것.)")


if __name__ == "__main__":
    main()
