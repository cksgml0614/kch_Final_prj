# LSTM + market_bigmove_ratio(16번째 피처, 전날 값) — (0.3,1e-4), seed 42, 운영과 같은 split. val만, 저장은 scratchpad.
import os, sys, json
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\kch_Final_prj"); os.chdir(r"C:\kch_Final_prj")
from datetime import date
import numpy as np, pandas as pd, torch
import 가격예측.가격예측_변동성_공통 as c
from 가격예측.garch_baseline import compute_log_returns_pct, load_close_prices, rmse_mae
from 가격예측.pooled_dataset import build_pooled_sequences, compute_global_split_dates
from 가격예측.sequence_dataset import FeatureScaler
from 가격예측.split_dataset import build_merged_dataset_v2, build_merged_dataset_v2_volatility_hybrid
from 가격예측.train_common import evaluate_pooled_predictions, train_pooled_lstm
from 가격예측.검증용.highvol_regime_diagnosis import compute_big_move_labels
from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
tickers, start, end = ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START, date.today().isoformat()
labs = {}
for t in tickers:
    ret = compute_log_returns_pct(load_close_prices(t, start, end))
    labs[t] = compute_big_move_labels(ret).where(ret.abs().shift(1).rolling(60, min_periods=60).mean().notna())
ratio = pd.DataFrame(labs).astype(float).mean(axis=1, skipna=True).dropna()
ratio_prev = ratio.shift(1)                       # 행 t = 전날(t-1) 비율 — 누수 방지
ref, _ = build_merged_dataset_v2(tickers[0], start, end); te, ve = compute_global_split_dates(ref.index)
B = {t: c.compute_ticker_baselines(t, start, end, te, ve) for t in tickers}
dropped = {}
def bf(ticker, s, e, precomputed_indicators=None, true_kospi=None):
    b = B[ticker]
    m, meta = build_merged_dataset_v2_volatility_hybrid(ticker, s, e, te, precomputed_sigma=b["sigma_full"], garch_params=b["params"],
                                                        precomputed_indicators=precomputed_indicators, true_kospi=true_kospi)
    m = m.copy(); tgt = m.pop("target")
    m["market_bigmove_ratio"] = ratio_prev.reindex(m.index).values; m["target"] = tgt
    n0 = len(m); m = m.dropna(); dropped[ticker] = n0 - len(m)
    return m, meta
sp, fc, t2id, (te2, ve2) = build_pooled_sequences(tickers, start, end, lookback=c.LOOKBACK, build_fn=bf)
assert (te2, ve2) == (te, ve) and fc[-1] == "market_bigmove_ratio" and len(fc) == 16
Xtr, ytr, tidtr, _, _ = sp["train"]; Xva, yva, tidva, dva, tnva = sp["val"]
print(f"feats={len(fc)} n_train={len(Xtr)} (15피처 때 104,229) n_val={len(Xva)} | 초기 NaN으로 빠진 행 합계={sum(dropped.values())}", flush=True)
sc = FeatureScaler().fit(Xtr); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
r = train_pooled_lstm(sc.transform(Xtr).astype(np.float32), ytr, tidtr, sc.transform(Xva).astype(np.float32), yva, tidva,
                      len(fc), c.LOOKBACK, len(tickers), c.EMBEDDING_DIM, c.D_MODEL, c.NHEAD, c.NUM_LAYERS, c.DIM_FEEDFORWARD,
                      c.DROPOUT, c.BATCH_SIZE, c.LR, c.WEIGHT_DECAY, c.SMOKE_EPOCHS, c.MAX_EPOCHS, c.PATIENCE, dev, seed=42, verbose=False)
p, a, tids = evaluate_pooled_predictions(r["model"], r["val_loader"], dev)
h = float(np.mean([rmse_mae(p[tids == t], a[tids == t])[0] for t in np.unique(tids)]))
base = {k: float(np.mean([B[t]["val_rmse"][k] for t in tickers])) for k in ("garch", "sma20", "parkinson")}
np.save(os.path.join(OUT, "lstm_mbr_pred.npy"), p)
pd.DataFrame({"ticker": tnva, "date": pd.to_datetime(dva)}).to_parquet(os.path.join(OUT, "lstm_mbr_keys.parquet"))
rec = dict(gate_rmse=h, passed=all(h < v for v in base.values()), base=base, best_epoch=r["best_epoch"],
           std_ratio=float(p.std() / a.std()), pred_max=float(p.max()), overfit_ratio=float(r["overfit_ratio_at_end"]))
json.dump(rec, open(os.path.join(OUT, "lstm_mbr_results.json"), "w")); print("RESULT", json.dumps(rec)); print("DONE")
