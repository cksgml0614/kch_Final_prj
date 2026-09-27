# LSTM 규제 완화 스윕 — (dropout, weight_decay) 4조합, seed 42, 운영과 동일 데이터·split. val만, 저장은 scratchpad.
import os, sys, json
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\kch_Final_prj"); os.chdir(r"C:\kch_Final_prj")
from datetime import date
import numpy as np, pandas as pd, torch
import 가격예측.가격예측_변동성_공통 as c
from 가격예측.garch_baseline import rmse_mae
from 가격예측.pooled_dataset import build_pooled_sequences, compute_global_split_dates
from 가격예측.sequence_dataset import FeatureScaler
from 가격예측.split_dataset import build_merged_dataset_v2, build_merged_dataset_v2_volatility_hybrid
from 가격예측.train_common import evaluate_pooled_predictions, train_pooled_lstm
from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
tickers, start, end = ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START, date.today().isoformat()
ref, _ = build_merged_dataset_v2(tickers[0], start, end); te, ve = compute_global_split_dates(ref.index)
B = {t: c.compute_ticker_baselines(t, start, end, te, ve) for t in tickers}
def bf(ticker, s, e, precomputed_indicators=None, true_kospi=None):
    b = B[ticker]
    return build_merged_dataset_v2_volatility_hybrid(ticker, s, e, te, precomputed_sigma=b["sigma_full"], garch_params=b["params"],
                                                     precomputed_indicators=precomputed_indicators, true_kospi=true_kospi)
sp, fc, t2id, _ = build_pooled_sequences(tickers, start, end, lookback=c.LOOKBACK, build_fn=bf)
Xtr, ytr, tidtr, _, _ = sp["train"]; Xva, yva, tidva, dva, tnva = sp["val"]
sc = FeatureScaler().fit(Xtr); Xtr_s = sc.transform(Xtr).astype(np.float32); Xva_s = sc.transform(Xva).astype(np.float32)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base = {k: float(np.mean([B[t]["val_rmse"][k] for t in tickers])) for k in ("garch", "sma20", "parkinson")}
print("baselines(gate):", base, flush=True)
keys = pd.DataFrame({"ticker": tnva, "date": pd.to_datetime(dva)}); keys.to_parquet(os.path.join(OUT, "lstm_reg_keys.parquet"))
res = []
for do, wd in ((0.3, 1e-4), (0.15, 1e-4), (0.15, 1e-5), (0.1, 0.0)):
    r = train_pooled_lstm(Xtr_s, ytr, tidtr, Xva_s, yva, tidva, len(fc), c.LOOKBACK, len(tickers), c.EMBEDDING_DIM,
                          c.D_MODEL, c.NHEAD, c.NUM_LAYERS, c.DIM_FEEDFORWARD, do, c.BATCH_SIZE, c.LR, wd,
                          c.SMOKE_EPOCHS, c.MAX_EPOCHS, c.PATIENCE, dev, seed=42, verbose=False)
    p, a, tids = evaluate_pooled_predictions(r["model"], r["val_loader"], dev)
    h = float(np.mean([rmse_mae(p[tids == t], a[tids == t])[0] for t in np.unique(tids)]))
    name = f"do{do}_wd{wd:g}"; np.save(os.path.join(OUT, f"lstm_reg_{name}.npy"), p)
    rec = dict(name=name, dropout=do, weight_decay=wd, gate_rmse=h, passed=all(h < v for v in base.values()),
               best_epoch=r["best_epoch"], std_ratio=float(p.std() / a.std()), pred_max=float(p.max()),
               overfit_ratio=float(r["overfit_ratio_at_end"]))
    print("RESULT", json.dumps(rec), flush=True); res.append(rec)
json.dump({"base": base, "res": res}, open(os.path.join(OUT, "lstm_reg_results.json"), "w"))
print("DONE")
