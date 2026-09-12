# split_dataset.py — Task T-1 체크포인트 3
# base 피처(체크포인트 1) + momentum 피처(체크포인트 2)를 merge한 실제 유효 행 기준으로
# 정상 레짐/스트레스 구간을 분리하고, 정상 레짐 내에서 70/15/15 시계열 분할한다.
#
# 2026-09-06(Task T 자동화 착수): split_full_period() 추가 — 스트레스 구간 분리 없이 전체
# 기간을 그대로 70/15/15 분할한다. split_normal_regime()은 Task T-1 진단 스크립트들이 계속
# 참조하므로 그대로 남겨뒀다(값 변경 없음) — 새 자동화 파이프라인(가격예측_공통.py)은
# split_full_period()만 사용한다.

# 2026-09-06(실제 pooled 학습 최초 실행 중 발견·수정): build_next_day_merged_window() 추가.
# dataset_builder.build_next_day_feature_window()는 base V2 피처(12개)만 만들고 momentum
# 피처(excess_return_z_lag1)가 빠져있어, 실제 학습에 쓰는 feature_cols(13개, build_merged_
# dataset_v2 기준)와 어긋나는 버그였다 — 서빙(다음 거래일 예측) 경로가 한 번도 실제로
# 끝까지 실행된 적이 없어(가격예측_공통.py도 설계만 되고 실행은 이번이 처음) 발견되지 않고
# 있었다. 단일 종목/pooled 두 파이프라인 모두 이 버그의 영향을 받으므로 여기 한 곳에서 고친다.

import sys

import pandas as pd

from constants import STRESS_PERIOD_END, STRESS_PERIOD_START
from 가격예측.dataset_builder import (
    build_base_dataset,
    build_base_dataset_v2,
    build_base_dataset_v2_volatility,
    build_next_day_feature_window,
    build_next_day_feature_window_volatility,
)
from 가격예측.momentum_feature import build_momentum_feature

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO는 나머지 전부 (반올림 오차를 test 쪽에 흡수)


def build_merged_dataset(ticker, start_date, end_date):
    """base(14피처+target) + momentum(1피처)을 merge하고, 둘 중 하나라도 NaN인 행은 제거한다."""
    base, raw = build_base_dataset(ticker, start_date, end_date)
    momentum, diag = build_momentum_feature(ticker, start_date, end_date)

    merged = base.copy()
    merged["excess_return_z_lag1"] = momentum
    before = len(merged)
    merged = merged.dropna(how="any")
    after = len(merged)

    return merged, {"base": base, "raw": raw, "momentum": momentum, "diag": diag,
                     "rows_before_momentum_dropna": before, "rows_after_momentum_dropna": after}


def build_merged_dataset_v2(ticker, start_date, end_date):
    """build_merged_dataset과 동일하나 build_base_dataset_v2(정상성 재설계 피처)를 쓴다.
    momentum(excess_return_z_lag1)은 이미 정상성 있는 z-score라 변경 없이 그대로 재사용."""
    base, raw = build_base_dataset_v2(ticker, start_date, end_date)
    momentum, diag = build_momentum_feature(ticker, start_date, end_date)

    merged = base.copy()
    merged["excess_return_z_lag1"] = momentum
    before = len(merged)
    merged = merged.dropna(how="any")
    after = len(merged)

    return merged, {"base": base, "raw": raw, "momentum": momentum, "diag": diag,
                     "rows_before_momentum_dropna": before, "rows_after_momentum_dropna": after}


def build_merged_dataset_v2_volatility(ticker, start_date, end_date):
    """build_merged_dataset_v2의 변동성 버전(2026-09-06, Task T 변동성 예측). target이
    |로그수익률x100|이고 피처에 recent_vol_ma20이 추가된 build_base_dataset_v2_volatility를
    쓴다는 점만 다르다 — momentum(excess_return_z_lag1)은 방향/변동성 어느 쪽이든 "종목의
    시장 대비 초과수익률 흐름"이라는 같은 의미의 입력 피처라 그대로 재사용한다(target 정의와
    무관, 재계산 불필요)."""
    base, raw = build_base_dataset_v2_volatility(ticker, start_date, end_date)
    momentum, diag = build_momentum_feature(ticker, start_date, end_date)

    merged = base.copy()
    merged["excess_return_z_lag1"] = momentum
    before = len(merged)
    merged = merged.dropna(how="any")
    after = len(merged)

    return merged, {"base": base, "raw": raw, "momentum": momentum, "diag": diag,
                     "rows_before_momentum_dropna": before, "rows_after_momentum_dropna": after}


def build_merged_dataset_v2_volatility_hybrid(ticker, start_date, end_date, train_end,
                                               precomputed_sigma=None, garch_params=None):
    """build_merged_dataset_v2_volatility(14피처) + GARCH(1,1) 조건부 σ를 15번째 피처로
    추가한 하이브리드 버전(2026-09-06) — GARCH가 이미 잡아낸 "어제 변동성→오늘 변동성"
    persistence 신호를 Transformer가 처음부터 재학습하지 않고 그대로 입력받게 하기 위함.

    train_end가 필요한 이유: GARCH 파라미터는 train 구간만으로 추정해야 하는데, 이 함수는
    종목별로 독립 호출되므로 전역 분할 경계(pooled_dataset.compute_global_split_dates로 미리
    계산된 값)를 호출부가 넘겨줘야 한다 — 이 함수 내부에서 새로 계산하면 대표 종목 기준과
    어긋날 위험이 있다(2026-09-06 GARCH 실험 때 raw 인덱스 기준으로 잘못 계산해 경계가
    한 달 가까이 어긋났던 사고 재발 방지).

    precomputed_sigma/garch_params(둘 다 2026-09-06 추가, 선택): 이미 계산해둔 전체 구간
    σ 시리즈·파라미터가 있으면 그대로 쓰고 내부에서 다시 계산하지 않는다 — 가격예측_변동성_
    일일수집.py가 GARCH 파라미터를 월 1회만 재추정(캐싱)하도록 만들면서, 매일 실행 시
    이 함수가 매번 재추정하면 캐싱이 무의미해지는 문제를 막기 위해 추가했다. 생략하면
    기존과 동일하게(garch_baseline.compute_full_period_sigma()로 매번 계산) 동작한다 —
    기존 호출부(test_evaluation_pooled_volatility_hybrid.py 등)는 값 변경 없음.

    garch_baseline 함수들을 그대로 재사용한다 — 여기서 지연 임포트하는 이유는 garch_baseline.py가
    이미 split_dataset.build_merged_dataset_v2를 임포트하고 있어(분할 경계 계산용), 모듈
    최상단에서 서로를 임포트하면 순환 임포트가 되기 때문이다."""
    merged, meta = build_merged_dataset_v2_volatility(ticker, start_date, end_date)

    if precomputed_sigma is not None:
        sigma_full, params = precomputed_sigma, garch_params
    else:
        from 가격예측.검증용.garch_baseline_check import compute_full_period_sigma  # 2026-09-12: 함수가 검증용/으로 이동(경로 갱신, 순환 임포트 방지 목적은 동일)
        sigma_full, params = compute_full_period_sigma(ticker, start_date, end_date, train_end)

    merged = merged.copy()
    merged["garch_sigma"] = sigma_full.reindex(merged.index)
    before = len(merged)
    merged = merged.dropna(how="any")
    after = len(merged)

    meta = dict(meta)
    meta["garch_params"] = params
    meta["rows_before_garch_dropna"] = before
    meta["rows_after_garch_dropna"] = after
    return merged, meta


def build_next_day_merged_window(ticker, start_date, end_date, lookback):
    """서빙(추론) 전용 — build_next_day_feature_window()(base V2 피처)에 momentum 피처
    (excess_return_z_lag1)를 합쳐, 실제 학습에 쓰는 feature_cols(build_merged_dataset_v2
    기준 13개)와 정확히 일치하는 lookback 시퀀스를 만든다.

    momentum 쪽도 동일한 원리를 적용한다: build_momentum_feature()가 반환하는 momentum
    시리즈(momentum[t] = z_(t-1), 이미 실현된 과거 target들의 피처)의 마지막 (lookback-1)개는
    그대로 쓰고, 마지막 행(=아직 실현 안 된 T+1행의 momentum)은 diag["z"]의 마지막 값(오늘 T
    시점에 계산된 z_T)을 그대로 가져다 쓴다 — build_next_day_feature_window()가 base 피처에
    적용한 것과 완전히 동일한 "오늘 원값 = 다음 행 피처" 관계의 momentum 버전이다.

    반환: (window, last_confirmed_date) — window는 build_next_day_feature_window()와 같은
    컬럼(12개) + 'excess_return_z_lag1' 1개 = 13개, feature_cols와 순서만 다를 수 있으니
    호출부는 반드시 컬럼명으로 인덱싱해서 쓸 것(위치 의존 금지)."""
    base_window, last_confirmed_date = build_next_day_feature_window(ticker, start_date, end_date, lookback)

    momentum, diag = build_momentum_feature(ticker, start_date, end_date)
    momentum_next = diag["z"].iloc[-1]
    if pd.isna(momentum_next):
        raise RuntimeError(f"{ticker}: 최신 momentum 값이 NaN — 가격 이력이 window보다 짧을 수 있음")

    past_momentum = momentum.tail(lookback - 1).values
    momentum_col = list(past_momentum) + [float(momentum_next)]
    if len(momentum_col) != lookback:
        raise RuntimeError(
            f"{ticker}: momentum 시퀀스 길이 불일치 — 기대 {lookback}, 실제 {len(momentum_col)}"
        )

    window = base_window.copy()
    window["excess_return_z_lag1"] = momentum_col
    return window, last_confirmed_date


def build_next_day_merged_window_volatility_hybrid(ticker, start_date, end_date, lookback, garch_params):
    """서빙(추론) 전용 — 변동성 하이브리드 모델(15피처)의 다음 거래일 예측 시퀀스를 만든다
    (2026-09-06, 가격예측_변동성_일일수집.py용). build_next_day_feature_window_volatility()
    (13개: base 12 + recent_vol_ma20) + momentum(1개, build_next_day_merged_window()와 동일
    원리) + garch_sigma(1개, GARCH 다음 거래일 예측)를 합친다.

    garch_params: 월 1회 캐싱된(또는 방금 재추정한) 고정 GARCH 파라미터 — 호출부가 넘겨준다.
    이 함수 내부에서 재추정하지 않는다(캐싱 취지 유지 — 매일 예측 때마다 재추정하면 캐싱이
    무의미해짐).

    ⚠️ 정렬 주의: returns(로그수익률)는 raw(get_features 원본)보다 첫 행이 하나 적다(pct_change
    특성상 첫 행이 NaN이라 dropna로 빠짐) — 그래서 위치 기반이 아니라 base_window의 실제
    날짜 라벨(base_window.index)로 GARCH 조회 시작 위치를 찾는다(위치 산술로 하면 하루
    어긋난다).

    반환: (window, last_confirmed_date) — window 컬럼 15개(순서는 feature_cols와 다를 수
    있으니 호출부는 반드시 컬럼명으로 인덱싱할 것)."""
    from 가격예측.garch_baseline import (  # 순환 임포트 방지(지연 임포트)
        compute_log_returns_pct,
        forecast_next_day_sigma,
        forecast_with_fixed_params,
        load_close_prices,
    )

    base_window, last_confirmed_date = build_next_day_feature_window_volatility(ticker, start_date, end_date, lookback)

    momentum, diag = build_momentum_feature(ticker, start_date, end_date)
    momentum_next = diag["z"].iloc[-1]
    if pd.isna(momentum_next):
        raise RuntimeError(f"{ticker}: 최신 momentum 값이 NaN — 가격 이력이 window보다 짧을 수 있음")
    past_momentum = momentum.tail(lookback - 1).values
    momentum_col = list(past_momentum) + [float(momentum_next)]

    close = load_close_prices(ticker, start_date, end_date)
    returns = compute_log_returns_pct(close)

    past_dates = base_window.index[:-1]  # window 마지막 행(오늘=T+1용) 제외 (lookback-1)개 날짜
    # ⚠️ 2026-09-06: forecast_with_fixed_params()가 "start_pos+1부터 시작"하도록 수정됨(라벨링
    # 버그 수정, garch_baseline.py 참고) — 출력 첫 날짜가 past_dates[0]이 되려면 그 하루 전
    # 위치를 start_pos로 넘겨야 한다.
    target_pos = returns.index.get_loc(past_dates[0])
    past_sigma_series = forecast_with_fixed_params(returns, garch_params, target_pos - 1)
    if not past_sigma_series.index.equals(pd.DatetimeIndex(past_dates)):
        raise RuntimeError(f"{ticker}: GARCH 과거 시퀀스 날짜가 base_window와 어긋남")
    garch_next = forecast_next_day_sigma(returns, garch_params)
    garch_col = list(past_sigma_series.values) + [garch_next]

    if len(momentum_col) != lookback or len(garch_col) != lookback:
        raise RuntimeError(
            f"{ticker}: 시퀀스 길이 불일치 — momentum={len(momentum_col)}, garch={len(garch_col)}, 기대={lookback}"
        )

    window = base_window.copy()
    window["excess_return_z_lag1"] = momentum_col
    window["garch_sigma"] = garch_col
    return window, last_confirmed_date


def split_full_period(merged, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """merged 전체 기간을 정상/스트레스 구분 없이 시계열 70/15/15로 분할한다(2026-09-06,
    Task T 자동화 착수 시 설계 확정 — 특정 구간을 하드코딩으로 예외 처리하는 방식은 다종목
    확장과 맞지 않다는 판단, 뉴스 트랙 D-3와 동일 논리). split_normal_regime()과 달리
    STRESS_PERIOD_START/END를 전혀 참조하지 않는다 — 그 상수와 analysis/detect_stress_period.py는
    Task T-1 진단 기록으로만 보존하고 새 파이프라인은 참조하지 않는다(CLAUDE.md 명시)."""
    n = len(merged)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)

    train = merged.iloc[:n_train]
    val = merged.iloc[n_train:n_train + n_val]
    test = merged.iloc[n_train + n_val:]

    return train, val, test


def split_normal_regime(merged, stress_start):
    """merged를 stress_start 기준으로 정상/스트레스로 나누고, 정상 구간을 70/15/15로 분할한다."""
    stress_start_ts = pd.Timestamp(stress_start)

    normal = merged[merged.index < stress_start_ts]
    stress = merged[merged.index >= stress_start_ts]

    n = len(normal)
    n_train = round(n * TRAIN_RATIO)
    n_val = round(n * VAL_RATIO)

    train = normal.iloc[:n_train]
    val = normal.iloc[n_train:n_train + n_val]
    test = normal.iloc[n_train + n_val:]

    return train, val, test, normal, stress


if __name__ == "__main__":
    TICKER = "005930"
    START = "2020-01-02"
    END = "2026-07-31"  # 전체 구간(정상+스트레스) 통째로 merge 후 상수 기준 분리

    merged, meta = build_merged_dataset(TICKER, START, END)

    print(f"=== build_merged_dataset({TICKER}, {START}, {END}) ===")
    print(f"base(체크포인트1) 행 수: {len(meta['base'])}")
    print(f"momentum(체크포인트2) NaN 제외 유효 행 수: {meta['momentum'].notna().sum()}")
    print(f"merge 직후(dropna 전) 행 수: {meta['rows_before_momentum_dropna']}")
    print(f"merge 후 실제 유효 행 수(dropna 후, 최종 학습 가능 행 수): {meta['rows_after_momentum_dropna']}")
    print(f"merge로 추가 소실된 행 수: {meta['rows_before_momentum_dropna'] - meta['rows_after_momentum_dropna']}")
    print(f"컬럼: {list(merged.columns)}")

    train, val, test, normal, stress = split_normal_regime(merged, STRESS_PERIOD_START)

    print(f"\n=== 정상 레짐 / 스트레스 분리 (기준: STRESS_PERIOD_START={STRESS_PERIOD_START}) ===")
    print(f"정상 레짐: {len(normal)}행 ({normal.index.min().date()} ~ {normal.index.max().date()})")
    print(f"스트레스: {len(stress)}행 ({stress.index.min().date()} ~ {stress.index.max().date()})")
    print(f"합계 = merge 후 유효 행 수와 일치: {len(normal) + len(stress) == len(merged)}")

    print(f"\n=== 정상 레짐 내 70/15/15 분할 (실제 merge 유효 행 {len(normal)}행 기준) ===")
    print(f"train: {len(train)}행 ({train.index.min().date()} ~ {train.index.max().date()})")
    print(f"val:   {len(val)}행 ({val.index.min().date()} ~ {val.index.max().date()})")
    print(f"test:  {len(test)}행 ({test.index.min().date()} ~ {test.index.max().date()})")
    print(f"비율: train {len(train)/len(normal)*100:.1f}% / val {len(val)/len(normal)*100:.1f}% / "
          f"test {len(test)/len(normal)*100:.1f}%")

    # --- 스트레스 구간 상수와 test셋 경계 겹침 최종 확인 ---
    print("\n=== 스트레스 구간(2026-02-02~2026-07-31) vs test셋 경계 최종 확인 ===")
    stress_start_ts = pd.Timestamp(STRESS_PERIOD_START)
    stress_end_ts = pd.Timestamp(STRESS_PERIOD_END)

    test_max = test.index.max()
    no_overlap_test = test_max < stress_start_ts
    print(f"test셋 최종일 = {test_max.date()}  <  스트레스 시작일 = {stress_start_ts.date()}  →  겹침 없음: {no_overlap_test}")

    all_normal_before_stress = bool((normal.index < stress_start_ts).all())
    print(f"정상 레짐(train/val/test 전체) 모든 행이 스트레스 시작일 이전인가: {all_normal_before_stress}")

    all_stress_in_range = bool(((stress.index >= stress_start_ts) & (stress.index <= stress_end_ts)).all())
    print(f"스트레스 구간 모든 행이 [{STRESS_PERIOD_START}, {STRESS_PERIOD_END}] 안에 있는가: {all_stress_in_range}")

    gap_days = (stress.index.min() - test_max).days
    print(f"test 마지막날 → 스트레스 첫날 간격: {gap_days}일 (음수/0이면 겹침, CLAUDE.md 정의상 거래일 갭이라 "
          f"실제 거래일 기준으로는 test 다음 거래일이 스트레스 첫 거래일이어야 정상)")

    assert no_overlap_test and all_normal_before_stress and all_stress_in_range, "스트레스 구간과 정상 레짐이 겹친다 — 확인 필요"
    print("\n✅ 최종 확인: merge 후에도 스트레스 구간과 test셋(정상 레짐 전체) 겹침 없음.")
