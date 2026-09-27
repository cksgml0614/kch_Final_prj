import numpy as np, pandas as pd
from datetime import date
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
import 가격예측.가격예측_변동성_공통 as c
from 가격예측.momentum_feature import load_true_kospi
from 가격예측.pooled_dataset import compute_global_split_dates
from 가격예측.split_dataset import build_merged_dataset_v2, build_merged_dataset_v2_volatility_hybrid
from 시장지표.feature_loader import load_indicator_cache
from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
start, end = STOCK_INITIAL_LOAD_START, date.today().isoformat()
ref, _ = build_merged_dataset_v2(ACTIVE_TICKERS[0], start, end); te, ve = compute_global_split_dates(ref.index)
ind = load_indicator_cache(end); tk = load_true_kospi(start, end); fr = []
for t in ACTIVE_TICKERS:
    b = c.compute_ticker_baselines(t, start, end, te, ve)
    m, _ = build_merged_dataset_v2_volatility_hybrid(t, start, end, te, precomputed_sigma=b["sigma_full"], garch_params=b["params"], precomputed_indicators=ind, true_kospi=tk)
    fr.append(m[m.index <= te])
tr = pd.concat(fr); rng = np.random.default_rng(0); idx = rng.choice(len(tr), 30000, replace=False); s = tr.iloc[idx]
y = s.target.values
for f in ("close_return", "open_gap_ratio", "garch_sigma", "hl_range_ratio"):
    x = s[f].values; ax_ = np.abs(x)
    uniq = len(np.unique(x)) / len(x); zero = (x == 0).mean()
    mi = mutual_info_regression(x[:, None], y, random_state=0)[0]
    mi_null = np.mean([mutual_info_regression(x[:, None], rng.permutation(y), random_state=0)[0] for _ in range(3)])
    xj = x + rng.normal(0, x.std() * 1e-3, len(x))
    mi_j = mutual_info_regression(xj[:, None], y, random_state=0)[0]
    print(f"{f:<16} 고유값비율={uniq:.3f} 0비율={zero:.3f} | MI={mi:.4f} 셔플MI={mi_null:.4f} 지터MI={mi_j:.4f} | "
          f"|x| pearson={pearsonr(ax_, y)[0]:.3f} spearman={spearmanr(ax_, y)[0]:.3f}")

# ── 결정 검증: 단일 피처 GBM 회귀의 val R² (비선형 신호가 실제 예측에 쓸모 있는가) ──
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
fv = []
for t in ACTIVE_TICKERS:
    b = c.compute_ticker_baselines(t, start, end, te, ve)
    m, _ = build_merged_dataset_v2_volatility_hybrid(t, start, end, te, precomputed_sigma=b["sigma_full"], garch_params=b["params"], precomputed_indicators=ind, true_kospi=tk)
    fv.append(m[(m.index > te) & (m.index <= ve)])
va = pd.concat(fv)
feats = [x for x in tr.columns if x != "target"]
for name, cols in [("close_return", ["close_return"]), ("|close_return|", None), ("open_gap_ratio", ["open_gap_ratio"]),
                   ("garch_sigma", ["garch_sigma"]), ("hl_range_ratio", ["hl_range_ratio"]), ("15피처 전체", feats)]:
    if cols is None:
        Xtr, Xva = np.abs(tr[["close_return"]].values), np.abs(va[["close_return"]].values)
    else:
        Xtr, Xva = tr[cols].values, va[cols].values
    g = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, early_stopping=True, random_state=0).fit(Xtr, tr.target.values)
    print(f"R2 {name:<16} val R²={r2_score(va.target.values, g.predict(Xva)):.4f}", flush=True)
