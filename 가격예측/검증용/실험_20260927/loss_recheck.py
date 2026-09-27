# 급변구간 손실함수 3종 + LSTM 재검증 — 대칭 집계(같은 행, 같은 방식)로 재판정. val만, 저장 없음.
import os, sys, json, time
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\kch_Final_prj"); os.chdir(r"C:\kch_Final_prj")
from datetime import date
import numpy as np, pandas as pd, torch
from sklearn.isotonic import IsotonicRegression
import 가격예측.가격예측_변동성_공통 as c
from 가격예측.garch_baseline import compute_log_returns_pct, compute_sma_baseline, load_close_prices
from 가격예측.parkinson_baseline import compute_parkinson_vol_pct, load_high_low, PARK_WINDOW
from 가격예측.pooled_dataset import build_pooled_sequences, compute_global_split_dates
from 가격예측.sequence_dataset import FeatureScaler
from 가격예측.split_dataset import build_merged_dataset_v2, build_merged_dataset_v2_volatility_hybrid
from 가격예측.train_common import (evaluate_pooled_predictions, train_pooled_lstm, train_pooled_transformer,
                                  train_pooled_transformer_pinball, train_pooled_transformer_weighted)
from 가격예측.검증용.highvol_regime_diagnosis import compute_big_move_labels, compute_first_day_labels
from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START

tickers, start, end = ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START, date.today().isoformat()
ref, _ = build_merged_dataset_v2(tickers[0], start, end)
train_end, val_end = compute_global_split_dates(ref.index)
B = {t: c.compute_ticker_baselines(t, start, end, train_end, val_end) for t in tickers}
rows = []
for t in tickers:
    close = load_close_prices(t, start, end); ret = compute_log_returns_pct(close)
    hl = load_high_low(t, start, end)
    park = compute_parkinson_vol_pct(hl["high"], hl["low"]).rolling(PARK_WINDOW, min_periods=PARK_WINDOW).mean().shift(1)
    bm = compute_big_move_labels(ret); fd = compute_first_day_labels(bm, 5)
    rows.append(pd.DataFrame({"ticker": t, "realized": ret.abs(), "garch": B[t]["sigma_full"].reindex(ret.index),
        "sma20": compute_sma_baseline(ret), "parkinson": park.reindex(ret.index), "bigmove": bm, "firstday": fd}))
base = pd.concat(rows).rename_axis("date").reset_index()

def build_fn(ticker, s, e, precomputed_indicators=None, true_kospi=None):
    b = B[ticker]
    return build_merged_dataset_v2_volatility_hybrid(ticker, s, e, train_end, precomputed_sigma=b["sigma_full"],
        garch_params=b["params"], precomputed_indicators=precomputed_indicators, true_kospi=true_kospi)
splits, fc, t2id, _ = build_pooled_sequences(tickers, start, end, lookback=c.LOOKBACK, build_fn=build_fn)
Xtr, ytr, tidtr, dtr, tntr = splits["train"]; Xva, yva, tidva, dva, tnva = splits["val"]
sc = FeatureScaler().fit(Xtr); Xtr_s = sc.transform(Xtr).astype(np.float32); Xva_s = sc.transform(Xva).astype(np.float32)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"split train_end={train_end.date()} val_end={val_end.date()} n_train={len(Xtr)} n_val={len(Xva)}", flush=True)

key = lambda tn, d: pd.MultiIndex.from_arrays([tn, pd.to_datetime(d)])
bi = base.set_index(["ticker", "date"])
trf = bi.reindex(key(tntr, dtr)); vaf = bi.reindex(key(tnva, dva))
assert np.allclose(vaf.realized.values, yva, atol=1e-4)
valid = vaf[["garch", "sma20", "parkinson"]].notna().all(axis=1).values
print(f"baseline 유효 행 {int(valid.sum())}/{len(valid)} (나머지는 모든 비교에서 제외)", flush=True)
bm_tr = trf.bigmove.fillna(False).values.astype(bool)
q80 = np.quantile(yva, 0.8)
top = yva > q80; bmv = vaf.bigmove.fillna(False).values.astype(bool); fdv = vaf.firstday.fillna(False).values.astype(bool)
G = {"①급변일": valid & top & bmv, "③하위80%": valid & ~top, "④지속중": valid & top & bmv & ~fdv, "전체": valid.copy()}
BL = {m: vaf[m].values.astype(float) for m in ("garch", "sma20", "parkinson")}

def rm(p, a): return float(np.sqrt(np.mean((p - a) ** 2)))
def agg(p, mask, how):
    if how == "pooled": return rm(p[mask], yva[mask])
    return float(np.mean([rm(p[mask & (tidva == t)], yva[mask & (tidva == t)]) for t in np.unique(tidva[mask])]))
BASE_ROWS = {}
for g, m in G.items():
    for how in ("pooled", "per_ticker"):
        BASE_ROWS[(g, how)] = {k: agg(v, m, how) for k, v in BL.items()}
print("baselines:", json.dumps({f"{g}|{h}": {k: round(x, 4) for k, x in d.items()} for (g, h), d in BASE_ROWS.items()}, ensure_ascii=False), flush=True)

common = (len(fc), c.LOOKBACK, len(tickers), c.EMBEDDING_DIM, c.D_MODEL, c.NHEAD, c.NUM_LAYERS, c.DIM_FEEDFORWARD,
          c.DROPOUT, c.BATCH_SIZE, c.LR, c.WEIGHT_DECAY, c.SMOKE_EPOCHS, c.MAX_EPOCHS, c.PATIENCE, dev)
results = {}
def record(name, p, extra=None):
    rec = {"name": name}
    for g, m in G.items():
        for how in ("pooled", "per_ticker"):
            rec[f"{g}|{how}"] = agg(p, m, how)
    rec.update(extra or {})
    results[name] = rec; np.save(os.path.join(OUT, f"lossrecheck_{name}.npy"), p)
    print("RESULT", json.dumps(rec, ensure_ascii=False), flush=True)
def evalm(r): return evaluate_pooled_predictions(r["model"], r["val_loader"], dev)[0]

t0 = time.time()
r = train_pooled_transformer(Xtr_s, ytr, tidtr, Xva_s, yva, tidva, *common, seed=42, verbose=False)
p_base = evalm(r); record("baseline_MSE", p_base, {"best_epoch": r["best_epoch"]})
iso = IsotonicRegression(out_of_bounds="clip").fit(p_base, yva); record("isotonic(val-fit)", iso.predict(p_base))
for label, wmask in (("w1_top20q", ytr > np.quantile(ytr, 0.8)), ("w2_bigmove", bm_tr)):
    for W in (1.5, 2.0, 3.0):
        sw = np.where(wmask, W, 1.0).astype(np.float32)
        r = train_pooled_transformer_weighted(Xtr_s, ytr, tidtr, sw, Xva_s, yva, tidva, *common, seed=42, verbose=False)
        record(f"{label}_W{W}", evalm(r), {"best_epoch": r["best_epoch"]})
for tau in (0.6, 0.7, 0.8):
    r = train_pooled_transformer_pinball(Xtr_s, ytr, tidtr, Xva_s, yva, tidva, tau, *common, seed=42, verbose=False)
    record(f"pinball_tau{tau}", evalm(r), {"best_epoch": r["best_epoch"]})
for s in (42, 43, 44):
    r = train_pooled_lstm(Xtr_s, ytr, tidtr, Xva_s, yva, tidva, *common, seed=s, verbose=False)
    record(f"lstm_s{s}", evalm(r), {"best_epoch": r["best_epoch"]})
print(f"total {time.time()-t0:.0f}s")
pd.DataFrame(results).T.to_csv(os.path.join(OUT, "loss_recheck_results.csv"))
json.dump({f"{g}|{h}": d for (g, h), d in BASE_ROWS.items()}, open(os.path.join(OUT, "loss_recheck_baselines.json"), "w"), ensure_ascii=False)
print("DONE")
