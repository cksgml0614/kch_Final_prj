# 다음날 급변 여부 전용 분류기(가벼운 GBM) — train 학습, val PR-AUC. test 미사용, 저장 없음.
# 레이블: |r_t| >= 2 x 직전60일 평균|r| (highvol_regime_diagnosis.compute_big_move_labels, 기존 정의 그대로)
# 입력: 운영과 같은 15피처(행 t = t-1까지의 정보, garch_sigma 포함)
import os, sys, json, time
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\kch_Final_prj"); os.chdir(r"C:\kch_Final_prj")
from datetime import date
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
import 가격예측.가격예측_변동성_공통 as c
from 가격예측.garch_baseline import compute_log_returns_pct, load_close_prices
from 가격예측.momentum_feature import load_true_kospi
from 가격예측.pooled_dataset import compute_global_split_dates
from 가격예측.split_dataset import build_merged_dataset_v2, build_merged_dataset_v2_volatility_hybrid
from 가격예측.검증용.highvol_regime_diagnosis import compute_big_move_labels
from 시장지표.feature_loader import load_indicator_cache
from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START

tickers, start, end = ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START, date.today().isoformat()
ref, _ = build_merged_dataset_v2(tickers[0], start, end)
train_end, val_end = compute_global_split_dates(ref.index)
ind = load_indicator_cache(end); tk = load_true_kospi(start, end)
frames = []
for t in tickers:
    b = c.compute_ticker_baselines(t, start, end, train_end, val_end)
    m, _ = build_merged_dataset_v2_volatility_hybrid(t, start, end, train_end, precomputed_sigma=b["sigma_full"],
        garch_params=b["params"], precomputed_indicators=ind, true_kospi=tk)
    ret = compute_log_returns_pct(load_close_prices(t, start, end))
    m = m.copy(); m["label"] = compute_big_move_labels(ret).reindex(m.index)
    m["trail60"] = ret.abs().shift(1).rolling(60, min_periods=60).mean().reindex(m.index)
    m["ticker"] = t; frames.append(m)
df = pd.concat(frames)
feats = [x for x in df.columns if x not in ("target", "label", "ticker", "trail60")]
df = df[df.trail60.notna()]                      # 60일 평균이 정의된 행만(레이블이 성립하는 행)
tr = df[df.index <= train_end]; va = df[(df.index > train_end) & (df.index <= val_end)]
ytr = tr.label.astype(int).values; yva = va.label.astype(int).values
print(f"feats({len(feats)})={feats}")
print(f"train n={len(tr)} pos={ytr.mean():.4f} | val n={len(va)} pos={yva.mean():.4f} (무작위 PR-AUC 기준선)", flush=True)
base = yva.mean()

def fit_eval(y_train, seed):
    clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, early_stopping=True,
                                         validation_fraction=0.15, n_iter_no_change=30, random_state=seed)
    clf.fit(tr[feats].values, y_train)
    p = clf.predict_proba(va[feats].values)[:, 1]
    return average_precision_score(yva, p), roc_auc_score(yva, p), clf.n_iter_, p

res = {"base_rate": base}
print("\n=== 단일 피처 점수(참고, 학습 없음) ===")
for name, score in (("garch_sigma", va.garch_sigma), ("recent_vol_ma20", va.recent_vol_ma20),
                    ("garch_sigma/trail60", va.garch_sigma / va.trail60), ("hl_range_ratio", va.hl_range_ratio),
                    ("|close_return|", va.close_return.abs())):
    ap = average_precision_score(yva, score.values); res[f"single_{name}"] = ap
    print(f"  {name:<22} PR-AUC={ap:.4f} ({ap/base:.2f}x)  ROC-AUC={roc_auc_score(yva, score.values):.4f}")

print("\n=== GBM (15피처) seed 3개 ===")
aps = []
for s in (0, 1, 2):
    ap, roc, it, p = fit_eval(ytr, s); aps.append(ap)
    print(f"  seed={s} PR-AUC={ap:.4f} ({ap/base:.2f}x)  ROC-AUC={roc:.4f}  iters={it}", flush=True)
    if s == 0: np.save(os.path.join(OUT, "bigmove_clf_val_pred.npy"), p)
res["gbm_ap"] = aps

print("\n=== 레이블 셔플(train 레이블만 무작위로 섞어 재학습, N=7) ===")
rng = np.random.default_rng(12345); sh = []
for i in range(7):
    ap, roc, it, _ = fit_eval(rng.permutation(ytr), 0); sh.append(ap)
    print(f"  shuffle {i+1}: PR-AUC={ap:.4f} ({ap/base:.2f}x) iters={it}", flush=True)
sh = np.array(sh); real = float(np.mean(aps))
res["shuffle_ap"] = sh.tolist()
z = (real - sh.mean()) / sh.std() if sh.std() > 0 else float("inf")
print(f"\n실제 평균 PR-AUC={real:.4f} ({real/base:.2f}x) vs 셔플 평균 {sh.mean():.4f}±{sh.std():.4f} "
      f"(max {sh.max():.4f}) | 격차 {z:.1f}σ | 실제>셔플 전부: {bool(real > sh.max())}")
json.dump(res, open(os.path.join(OUT, "bigmove_clf_results.json"), "w"), ensure_ascii=False, indent=1)
print("DONE")
