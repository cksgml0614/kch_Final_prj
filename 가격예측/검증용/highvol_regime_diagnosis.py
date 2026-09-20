# highvol_regime_diagnosis.py — 급변 구간(상위20%) 오차의 성격 진단 (2026-09-20)
#
# 목적: fold3 val의 상위20% 오차가 "국면 진입 첫날"(사전 신호 없음)에 집중돼 있는지,
# "국면 지속 중"(GARCH σ가 이미 반영된 이후)에도 남아있는지 구분한다 — W 스윕(가중 MSE)을
# 계속할지, 비대칭 손실(2순위)로 넘어갈지 판단하는 근거.
#
# ⚠️ 라벨링 로직 출처: 결과_변동성_조기경보_검증.md의 "국면 진입 첫날" 판정을 그대로
# 재사용한다(급변일 = |수익률| >= 2x 직전 60일 평균 |수익률|, 첫날 = 직전 N거래일 이내 다른
# 급변일 없음, N=5 주 기준 + N=10/20 민감도). 이 라벨링 자체는 실측 가격만으로 정의되므로
# 그 문서가 폐기된 사유(GARCH forecast(start=k) σ 계산 버그로 recall/precision 수치가
# 무효화됨)와 무관하다 — 무효화된 것은 "GARCH 예측이 이 라벨을 얼마나 맞혔는가"였지, 라벨
# 정의 자체가 아니다. 원본 스크립트(volatility_early_warning_*_check.py)는 결과 문서화 후
# 삭제됐으므로 문서의 서술을 그대로 재구현한다.
#
# ⚠️ 재학습 관련: baseline 모델은 이전 실행(loss_weighting_high_vol.py)에서 이미 학습했지만
# 예측값을 디스크에 저장하지 않아 프로세스 종료와 함께 사라졌다. 예측값 없이는 이 진단
# 자체가 불가능하므로, 동일 시드/동일 데이터로 baseline을 1회 "재현"한다(새 실험이 아니라
# 기존 결과를 다시 만들어내는 것 — 아래에서 이전 실행의 집계 수치와 완전히 일치하는지
# 재현성 자체를 검증한다). W=1.5/2.0/3.0은 이번 진단과 무관하므로 재실행하지 않는다.
#
# test는 이번에도 전혀 사용하지 않는다.

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
from 가격예측.garch_baseline import compute_log_returns_pct, load_close_prices, rmse_mae
from 가격예측.검증용.garch_baseline_check import compute_full_period_sigma
from 가격예측.pooled_dataset import build_pooled_sequences, compute_global_split_dates
from 가격예측.sequence_dataset import FeatureScaler
from 가격예측.split_dataset import build_merged_dataset_v2, build_merged_dataset_v2_volatility_hybrid
from 가격예측.train_common import evaluate_pooled_predictions, make_pooled_loader, train_pooled_transformer

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TOP_QUANTILE = 0.80
BIG_MOVE_MULTIPLIER = 2.0
BIG_MOVE_WINDOW = 60
ISOLATION_N_PRIMARY = 5
ISOLATION_N_SENSITIVITY = [5, 10, 20]

# 이전 실행(loss_weighting_high_vol.py) baseline 결과 — 재현성 확인용 참고치.
PREV_BASELINE = {"top_rmse": 3.9636, "top_mae": 3.0028, "bottom_rmse": 1.2115, "bottom_mae": 0.9954}


def precompute_garch(tickers, start_date, end_date, train_end):
    cache = {}
    for i, ticker in enumerate(tickers, 1):
        sigma, params = compute_full_period_sigma(ticker, start_date, end_date, train_end)
        cache[ticker] = (sigma, params)
        if i % 20 == 0 or i == len(tickers):
            print(f"  GARCH sigma 사전계산 {i}/{len(tickers)}", flush=True)
    return cache


def make_build_fn(train_end, garch_cache):
    def build_fn(ticker, s, e, precomputed_indicators=None, true_kospi=None):
        sigma, params = garch_cache[ticker]
        return build_merged_dataset_v2_volatility_hybrid(
            ticker, s, e, train_end, precomputed_sigma=sigma, garch_params=params,
            precomputed_indicators=precomputed_indicators, true_kospi=true_kospi,
        )
    return build_fn


def compute_big_move_labels(returns, window=BIG_MOVE_WINDOW, multiplier=BIG_MOVE_MULTIPLIER):
    """급변일 = |수익률_t| >= multiplier * 직전 window일 평균|수익률| (t 미포함, leak-free
    관용구는 이 프로젝트의 SMA baseline과 동일: shift(1).rolling(window).mean()).
    반환: bool Series, index=returns.index."""
    trailing_avg = returns.abs().shift(1).rolling(window, min_periods=window).mean()
    return (returns.abs() >= multiplier * trailing_avg).fillna(False)


def compute_first_day_labels(big_move, n):
    """big_move(bool Series) 중 "고립된 첫날" — 직전 n거래일 이내에 다른 급변일이 없는 날.
    결과_변동성_조기경보_검증.md [3]/[4]의 정의 그대로."""
    dates = big_move.index
    big_move_dates = set(dates[big_move.values])
    is_first_day = pd.Series(False, index=dates)
    date_list = list(dates)
    pos = {d: i for i, d in enumerate(date_list)}
    for d in big_move_dates:
        i = pos[d]
        window_dates = date_list[max(0, i - n):i]
        if not any(wd in big_move_dates for wd in window_dates):
            is_first_day[d] = True
    return is_first_day


def main():
    tickers = ACTIVE_TICKERS
    end_date = date.today().isoformat()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ref, _ = build_merged_dataset_v2(tickers[0], STOCK_INITIAL_LOAD_START, end_date)
    train_end, val_end = compute_global_split_dates(ref.index)
    print(f"fold3 분할 경계: train_end={train_end.date()}  val_end={val_end.date()}")
    assert train_end.date().isoformat() == "2024-10-07", "fold3 경계가 이전 실행과 달라짐"

    print(f"\nGARCH sigma 사전계산({len(tickers)}종목)...")
    garch_cache = precompute_garch(tickers, STOCK_INITIAL_LOAD_START, end_date, train_end)

    build_fn = make_build_fn(train_end, garch_cache)
    splits, feature_cols, ticker_to_id, _ = build_pooled_sequences(
        tickers, STOCK_INITIAL_LOAD_START, end_date, lookback=LOOKBACK, build_fn=build_fn,
    )
    X_train, y_train, tid_train, d_train, _ = splits["train"]
    X_val, y_val, tid_val, d_val, _ = splits["val"]
    # ⚠️ splits["test"]는 여기서 끝 — 어떤 변수에도 담지 않는다.

    scaler = FeatureScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    print("\n=== baseline 재현(재학습이 아니라 이전 결과 재생성 — 동일 시드/데이터) ===")
    result = train_pooled_transformer(
        X_train_s, y_train, tid_train, X_val_s, y_val, tid_val,
        len(feature_cols), LOOKBACK, len(tickers), EMBEDDING_DIM,
        D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT,
        BATCH_SIZE, LR, WEIGHT_DECAY, SMOKE_EPOCHS, MAX_EPOCHS, PATIENCE,
        device, seed=SEED, verbose=True, label="baseline-repro",
    )
    val_loader = make_pooled_loader(X_val_s, y_val, tid_val, BATCH_SIZE, shuffle=False)
    preds, actuals, tids_out = evaluate_pooled_predictions(result["model"], val_loader, device)

    val_threshold = np.quantile(y_val, TOP_QUANTILE)
    top_mask = y_val > val_threshold
    bottom_mask = ~top_mask

    def seg(mask):
        r, m = rmse_mae(preds[mask], actuals[mask])
        return r, m, int(mask.sum())

    top_rmse, top_mae, n_top = seg(top_mask)
    bot_rmse, bot_mae, n_bot = seg(bottom_mask)
    print(f"\n재현 결과: 상위20% RMSE={top_rmse:.4f} MAE={top_mae:.4f} (n={n_top}) | "
          f"하위80% RMSE={bot_rmse:.4f} MAE={bot_mae:.4f} (n={n_bot})")
    print(f"이전 실행 참고치: 상위20% RMSE={PREV_BASELINE['top_rmse']} MAE={PREV_BASELINE['top_mae']} | "
          f"하위80% RMSE={PREV_BASELINE['bottom_rmse']} MAE={PREV_BASELINE['bottom_mae']}")
    reproduced = (abs(top_rmse - PREV_BASELINE["top_rmse"]) < 1e-3 and
                  abs(bot_rmse - PREV_BASELINE["bottom_rmse"]) < 1e-3)
    print(f"재현 여부(이전 실행과 사실상 동일한가, |차이|<1e-3): {reproduced}")
    if not reproduced:
        print("⚠️ 재현 실패 — 이 스크립트의 baseline이 이전 실행과 다르다. 원인 파악 전에는 "
              "아래 분석 결과를 이전 W 실험과 직접 비교하지 말 것.")

    # ── 급변일/국면 진입 첫날 라벨링 (종목별 전체 기간, 실측 가격 기준) ──
    print(f"\n=== 급변일/국면 진입 첫날 라벨링 (급변일 = |수익률| >= {BIG_MOVE_MULTIPLIER}x "
          f"직전 {BIG_MOVE_WINDOW}일 평균) ===")
    id_to_ticker = {v: k for k, v in ticker_to_id.items()}
    first_day_label = {n: np.zeros(len(y_val), dtype=bool) for n in ISOLATION_N_SENSITIVITY}
    big_move_label = np.zeros(len(y_val), dtype=bool)

    for ticker, tid in ticker_to_id.items():
        mask = tid_val == tid
        if mask.sum() == 0:
            continue
        close = load_close_prices(ticker, STOCK_INITIAL_LOAD_START, end_date)
        returns = compute_log_returns_pct(close)
        big_move = compute_big_move_labels(returns)

        val_dates = pd.DatetimeIndex(d_val[mask])
        idx_in_arr = np.where(mask)[0]

        bm_aligned = big_move.reindex(val_dates).fillna(False)
        big_move_label[idx_in_arr] = bm_aligned.values

        for n in ISOLATION_N_SENSITIVITY:
            first_day = compute_first_day_labels(big_move, n)
            fd_aligned = first_day.reindex(val_dates).fillna(False)
            first_day_label[n][idx_in_arr] = fd_aligned.values

    print(f"전체 val 표본 중 급변일: {int(big_move_label.sum())}/{len(y_val)}")
    for n in ISOLATION_N_SENSITIVITY:
        print(f"  N={n} 첫날: {int(first_day_label[n].sum())}건 / "
              f"지속중: {int((big_move_label & ~first_day_label[n]).sum())}건")

    # ── 상위20% 구간을 급변일 여부 + 첫날/지속중으로 분해 ──
    print(f"\n=== 상위20% 구간(n={n_top})을 급변일 라벨로 분해 (N={ISOLATION_N_PRIMARY} 주 기준) ===")
    top_not_bigmove = top_mask & ~big_move_label
    top_first_day = top_mask & big_move_label & first_day_label[ISOLATION_N_PRIMARY]
    top_continuing = top_mask & big_move_label & ~first_day_label[ISOLATION_N_PRIMARY]

    rows = []
    for label, m in [("첫날(국면 진입)", top_first_day), ("지속중(둘째 날 이후)", top_continuing),
                      ("급변일 기준 미달(상위20%지만 2x기준 미달)", top_not_bigmove)]:
        if m.sum() == 0:
            rows.append((label, float("nan"), float("nan"), 0))
            continue
        r, mae = rmse_mae(preds[m], actuals[m])
        rows.append((label, r, mae, int(m.sum())))

    print(f"{'그룹':<38}{'RMSE':>10}{'MAE':>10}{'n':>8}")
    for label, r, mae, n in rows:
        print(f"{label:<38}{r:>10.4f}{mae:>10.4f}{n:>8}")

    # ── N 민감도 (10/20) ──
    print(f"\n=== N 민감도 (상위20% 내 첫날 vs 지속중, N={ISOLATION_N_SENSITIVITY}) ===")
    for n in ISOLATION_N_SENSITIVITY:
        fd = top_mask & big_move_label & first_day_label[n]
        cont = top_mask & big_move_label & ~first_day_label[n]
        fd_r, fd_m = rmse_mae(preds[fd], actuals[fd]) if fd.sum() else (float("nan"), float("nan"))
        cont_r, cont_m = rmse_mae(preds[cont], actuals[cont]) if cont.sum() else (float("nan"), float("nan"))
        print(f"  N={n:>2}: 첫날 RMSE={fd_r:.4f} MAE={fd_m:.4f} (n={int(fd.sum())})  |  "
              f"지속중 RMSE={cont_r:.4f} MAE={cont_m:.4f} (n={int(cont.sum())})")

    # ── 해석 ──
    print("\n" + "=" * 100)
    print("=== 해석 ===")
    print("=" * 100)
    fd_rmse = rows[0][1]
    cont_rmse = rows[1][1]
    if np.isnan(fd_rmse) or np.isnan(cont_rmse):
        print("⚠️ 한쪽 그룹 표본이 0이라 비교 불가 — N 값이나 급변일 정의 재검토 필요.")
    elif cont_rmse < fd_rmse * 0.7:
        print(f"'국면 지속 중' RMSE({cont_rmse:.4f})가 '국면 진입 첫날' RMSE({fd_rmse:.4f})보다 "
              "뚜렷하게 낮음 → 오차가 첫날에 집중돼 있음 → '손실함수로 못 푸는 정보 한계'로 결론.")
        print("제안: W 스윕/비대칭 손실 모두 중단하고 이 결과를 한계로 문서화.")
    else:
        print(f"'국면 지속 중' RMSE({cont_rmse:.4f})가 '국면 진입 첫날' RMSE({fd_rmse:.4f})와 "
              "비슷하거나 오히려 큼 → 지속 구간에서도 개선 여지가 있다는 뜻.")
        print("제안: 비대칭 손실(2순위) 진행 여부를 사람과 다시 논의.")

    print("\n(주: test(2025-09-26~오늘)는 이번 실행에서 전혀 평가하지 않았다.)")


if __name__ == "__main__":
    main()
