# 초과수익률 z 5클래스 GBM 분류 — train 학습, val 평가. test 미사용, 저장 없음.
# 레이블: D-2 정의 그대로(momentum_feature._excess_return_z, 직전 60일 표준편차, t 미포함) — 행 t의 z_t.
import os, sys, json
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\kch_Final_prj"); os.chdir(r"C:\kch_Final_prj")
from datetime import date
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import 가격예측.가격예측_변동성_공통 as c
from 가격예측.momentum_feature import build_momentum_feature, load_true_kospi
from 가격예측.pooled_dataset import compute_global_split_dates
from 가격예측.split_dataset import build_merged_dataset_v2, build_merged_dataset_v2_volatility_hybrid
from 시장지표.feature_loader import load_indicator_cache
from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
pd.set_option("display.width", 200)

tickers, start, end = ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START, date.today().isoformat()
ref, _ = build_merged_dataset_v2(tickers[0], start, end)
train_end, val_end = compute_global_split_dates(ref.index)
ind = load_indicator_cache(end); tk = load_true_kospi(start, end)
frames = []
for t in tickers:
    b = c.compute_ticker_baselines(t, start, end, train_end, val_end)
    m, _ = build_merged_dataset_v2_volatility_hybrid(t, start, end, train_end, precomputed_sigma=b["sigma_full"],
        garch_params=b["params"], precomputed_indicators=ind, true_kospi=tk)
    _, diag = build_momentum_feature(t, start, end, true_kospi=tk)
    m = m.copy(); m["z"] = diag["z"].reindex(m.index); m["ticker"] = t
    # 정합성: 피처 excess_return_z_lag1(행 t)은 z_(t-1)이어야 한다
    zl = diag["z"].shift(1).reindex(m.index)
    ok = np.allclose(m.excess_return_z_lag1.values, zl.values, equal_nan=True)
    if not ok: print(f"⚠️ {t}: excess_return_z_lag1 != z.shift(1)")
    frames.append(m)
df = pd.concat(frames)
feats = [x for x in df.columns if x not in ("target", "z", "ticker")]
df = df[np.isfinite(df.z)]
bins = [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf]
df["y"] = pd.cut(df.z, bins=bins, labels=False, right=False).astype(int)   # 0:z<-1.5 ... 4:z>=1.5
tr = df[df.index <= train_end]; va = df[(df.index > train_end) & (df.index <= val_end)]
ytr, yva = tr.y.values, va.y.values
names = ["z<-1.5", "-1.5~-0.5", "-0.5~0.5", "0.5~1.5", "z>1.5"]
print(f"feats({len(feats)}) | train n={len(tr)} val n={len(va)}")
print("train 분포:", np.round(np.bincount(ytr, minlength=5) / len(ytr), 4).tolist())
print("val   분포:", np.round(np.bincount(yva, minlength=5) / len(yva), 4).tolist())

maj = np.bincount(ytr, minlength=5).argmax()
maj_acc = float((yva == maj).mean()); maj_f1 = f1_score(yva, np.full_like(yva, maj), average="macro", zero_division=0)
prior = np.bincount(ytr, minlength=5) / len(ytr); rng = np.random.default_rng(0)
rand_f1 = np.array([f1_score(yva, rng.choice(5, size=len(yva), p=prior), average="macro", zero_division=0) for _ in range(200)])
print(f"\n[baseline] 최빈 클래스({names[maj]}) 비율/정확도={maj_acc:.4f}, 최빈 예측 macro F1={maj_f1:.4f}")
print(f"[baseline] 클래스 사전분포 무작위 예측 macro F1={rand_f1.mean():.4f}±{rand_f1.std():.4f}", flush=True)

def fit(y, seed, cw="balanced"):
    clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, early_stopping=True, validation_fraction=0.15,
                                         n_iter_no_change=30, random_state=seed, class_weight=cw)
    clf.fit(tr[feats].values, y); p = clf.predict(va[feats].values)
    return f1_score(yva, p, average="macro", zero_division=0), accuracy_score(yva, p), clf.n_iter_, p

print("\n=== GBM 5클래스(class_weight=balanced) seed 3개 ===")
f1s = []
for s in (0, 1, 2):
    f1, acc, it, p = fit(ytr, s); f1s.append(f1)
    print(f"  seed={s} macro F1={f1:.4f} (최빈 대비 {f1/maj_f1:.2f}x, 무작위 대비 {f1/rand_f1.mean():.2f}x) acc={acc:.4f} iters={it}", flush=True)
    if s == 0: p0 = p
f1u, accu, itu, _ = fit(ytr, 0, cw=None)
print(f"  (참고) 가중치 없음 seed=0 macro F1={f1u:.4f} acc={accu:.4f} iters={itu}")

print("\n=== train 레이블 셔플 N=7 (같은 설정) ===")
rs = np.random.default_rng(12345); sh = []
for i in range(7):
    f1, acc, it, _ = fit(rs.permutation(ytr), 0); sh.append(f1)
    print(f"  shuffle {i+1}: macro F1={f1:.4f} acc={acc:.4f} iters={it}", flush=True)
sh = np.array(sh); real = float(np.mean(f1s))
print(f"\n실제 평균 macro F1={real:.4f} vs 셔플 {sh.mean():.4f}±{sh.std():.4f} (max {sh.max():.4f}) | "
      f"격차 {(real - sh.mean()) / sh.std():.1f}σ | 실제>셔플 전부: {bool(real > sh.max())}")

print("\n=== confusion matrix (seed 0, 행=실제, 열=예측, 행 정규화) ===")
cm = confusion_matrix(yva, p0, labels=range(5))
print(pd.DataFrame(np.round(cm / cm.sum(1, keepdims=True), 3), index=names, columns=names))
print("건수:"); print(pd.DataFrame(cm, index=names, columns=names))
print("\n=== 부분 신호 점검 ===")
ext_a = np.isin(yva, [0, 4]); ext_p = np.isin(p0, [0, 4])
prec = ext_a[ext_p].mean() if ext_p.any() else float("nan")
print(f"극단(|z|>=1.5) 여부: 실제 비율={ext_a.mean():.4f} | 예측 극단 중 실제 극단 비율(precision)={prec:.4f} "
      f"({prec/ext_a.mean():.2f}x) | recall={ext_p[ext_a].mean():.4f}")
f1_ext = f1_score(ext_a, ext_p); f1_ext_rand = np.mean([f1_score(ext_a, rng.random(len(ext_a)) < ext_p.mean()) for _ in range(100)])
print(f"  극단 vs 비극단 F1={f1_ext:.4f} vs 같은 예측 비율 무작위 {f1_ext_rand:.4f}")
both = ext_a & ext_p
sign_ok = ((yva[both] == 0) & (p0[both] == 0)) | ((yva[both] == 4) & (p0[both] == 4))
print(f"  실제·예측 모두 극단인 {int(both.sum())}건 중 방향(+/-)까지 맞은 비율={sign_ok.mean():.4f} (무작위 ~0.5)")
up_a = yva >= 3; dn_a = yva <= 1; up_p = p0 >= 3; dn_p = p0 <= 1; dirm = (up_a | dn_a) & (up_p | dn_p)
print(f"  방향(z>=0.5 vs z<-0.5) — 실제·예측 모두 비중립 {int(dirm.sum())}건 중 부호 일치={((up_a & up_p) | (dn_a & dn_p))[dirm].mean():.4f}")
json.dump({"f1": f1s, "shuffle": sh.tolist(), "maj_f1": maj_f1, "rand_f1": float(rand_f1.mean())}, open(os.path.join(OUT, "excess5_results.json"), "w"))
print("DONE")
