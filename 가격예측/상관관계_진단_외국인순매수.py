# 상관관계_진단_외국인순매수.py
# 100종목 pooled 기준 피처-피처/피처-target 상관관계 진단(다중공선성 재점검) — 순수 진단
# 스크립트. 본 모델/파이프라인에는 편입하지 않는다.
#
# 2026-09-20: 원래 이 파일에 있던 외국인 순매수(foreign_net_intensity/foreign_streak/
# foreign_minus_individual) 관련 수집·조인·GARCH 잔차 상관 코드는 전부 제거했다 — 외국인
# 순매수 트랙 자체가 폐기됐기 때문(KRX Data Marketplace 로그인 필수화, CLAUDE.md 참고).
# 파일명은 그대로 유지하고 순수 상관관계 진단 스크립트로만 남긴다.
#
# 2026-09-20(같은 날, 두 번째 수정): 전체 기간(train+val+test) 기준으로 상관계수를 계산하고
# 있던 것을 train만 대상으로 바꿨다 — 이 진단 결과를 근거로 ablation_feature_pruning.py에서
# 실제로 피처를 제거해 재학습하는데, val/test 구간 정보가 상관계수 계산에 섞여 들어가면
# "val/test를 보고 피처를 고른" 것과 다를 바 없는 정보 누수가 된다. 전체 기간판 히트맵/수치
# (feature_correlation_heatmap_pooled100*.png)는 별도 파일로 이미 저장돼 있어 그대로 두고,
# 이번 결과는 `_train_only` 접미사로 구분한다.

import sys
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
from 가격예측.momentum_feature import load_true_kospi
from 가격예측.pooled_dataset import compute_global_split_dates
from 가격예측.split_dataset import build_merged_dataset_v2, build_merged_dataset_v2_volatility_hybrid
from 시장지표.feature_loader import load_indicator_cache

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

OUT_DIR = "가격예측/checkpoints"


def collect_pooled_diagnostic_frame(tickers, start_date, end_date):
    """tickers 각각에 build_merged_dataset_v2_volatility_hybrid(가격 정상성 피처 + 거시지표
    8종 + garch_sigma + excess_return_z_lag1)를 적용해 concat한 pooled DataFrame을 반환한다.
    train_end는 대표 종목(tickers[0]) 기준 전역 분할 경계 하나로 통일한다(pooled_dataset.py와
    동일 기준 — 종목별로 따로 계산하면 경계가 어긋난다, 위 build_merged_dataset_v2_volatility_
    hybrid의 docstring 참고)."""
    ref_merged, _ = build_merged_dataset_v2(tickers[0], start_date, end_date)
    train_end, _ = compute_global_split_dates(ref_merged.index)

    # 거시지표 9종 + KOSPI 원자료는 종목과 무관하므로 한 번만 조회해 100종목이 공유한다
    # (2026-09-20, 성능 최적화 — 조인/계산 로직은 동일, 값 변경 없음 검증 완료).
    precomputed_indicators = load_indicator_cache(end_date)
    true_kospi = load_true_kospi(start_date, end_date)

    rows = []
    for ticker in tickers:
        try:
            merged, _ = build_merged_dataset_v2_volatility_hybrid(
                ticker, start_date, end_date, train_end,
                precomputed_indicators=precomputed_indicators, true_kospi=true_kospi,
            )
        except Exception as e:
            print(f"{ticker}: 피처 생성 실패 — {e}")
            continue
        merged = merged.copy()
        merged["ticker"] = ticker
        rows.append(merged)

    if not rows:
        raise RuntimeError("모든 종목 실패 — 피처 생성 로직/데이터 커버리지부터 확인")
    return pd.concat(rows), train_end


if __name__ == "__main__":
    pooled_full, train_end = collect_pooled_diagnostic_frame(
        ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START, date.today().isoformat()
    )
    print(f"\n결합 완료(전체 기간): {len(pooled_full)}행, {pooled_full['ticker'].nunique()}/{len(ACTIVE_TICKERS)}종목")

    # ⚠️ 데이터 유출 방지: 상관계수는 train 구간만으로 계산한다(val/test를 섞으면 그 정보를
    # 보고 피처를 골라 재학습하는 것과 같은 누수가 된다).
    pooled = pooled_full[pooled_full.index <= train_end]
    print(f"train만 대상: train_end={train_end.date()}, n={len(pooled)}행 "
          f"(전체 {len(pooled_full)}행 중 {len(pooled)/len(pooled_full)*100:.1f}%)")

    feature_cols = [c for c in pooled.columns if c not in ("target", "ticker")]

    # 텍스트 출력((a)/(b))과 히트맵이 전부 이 하나의 (feature+target) 상관행렬에서 파생된다 —
    # 별도로 다시 계산하지 않는다.
    full_corr = pooled[feature_cols + ["target"]].corr(numeric_only=True)

    # ============================================================
    # (a) 피처간 상관관계 (다중공선성 재점검, train만 — 8월 단일종목 히트맵과 비교용)
    # ============================================================
    print("\n=== (a) 피처간 상관관계 (다중공선성, train만) ===")
    print(full_corr.loc[feature_cols, feature_cols].round(2))

    pairs = []
    for i, a in enumerate(feature_cols):
        for b in feature_cols[i + 1:]:
            c = full_corr.loc[a, b]
            if abs(c) > 0.8:
                pairs.append((a, b, c))
    if pairs:
        print("\n|corr| > 0.8인 피처 쌍:")
        for a, b, c in sorted(pairs, key=lambda x: -abs(x[2])):
            print(f"  {a} - {b}: {c:+.4f}")
    else:
        print("\n|corr| > 0.8인 피처 쌍: 없음")

    # ============================================================
    # (b) 피처-target 상관계수 (train만)
    # ============================================================
    print("\n=== (b) 피처-target 상관계수 (train만) ===")
    corr_with_target = full_corr["target"].drop("target")
    corr_sorted = corr_with_target.reindex(corr_with_target.abs().sort_values(ascending=False).index)
    print(f"{'피처':<28}{'상관계수':>10}")
    for name, c in corr_sorted.items():
        print(f"  {name:<28}{c:+.4f}")

    # ============================================================
    # 히트맵 — target까지 포함한 (n+1)x(n+1) 전체 행렬, train만 대상
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(full_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                square=True, ax=ax, cbar_kws={"label": "Pearson correlation"})
    ax.set_title(f"Feature+target correlation heatmap (pooled {pooled['ticker'].nunique()} tickers, "
                 f"train만 n={len(pooled)}, train_end={train_end.date()})")
    plt.tight_layout()
    heatmap_path = f"{OUT_DIR}/feature_correlation_heatmap_pooled100_train_only.png"
    fig.savefig(heatmap_path, dpi=150)
    print(f"\n히트맵 저장: {heatmap_path} (전체 기간판: "
          f"{OUT_DIR}/feature_correlation_heatmap_pooled100_with_target.png, "
          f"8월 단일종목판: {OUT_DIR}/feature_correlation_heatmap.png)")
