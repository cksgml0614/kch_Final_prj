# 운영 LSTM(15피처, (0.3,1e-4), seed 42) 재학습 → train/val 예측 저장(후처리 임계값을 train 예측 분위수로 정하기 위함)
import os, sys
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\kch_Final_prj"); os.chdir(r"C:\kch_Final_prj")
from datetime import date
import numpy as np, pandas as pd, torch
import 가격예측.가격예측_변동성_공통 as c
from 가격예측.pooled_dataset import build_pooled_sequences, compute_global_split_dates
from 가격예측.sequence_dataset import FeatureScaler
from 가격예측.split_dataset import build_merged_dataset_v2, build_merged_dataset_v2_volatility_hybrid
from 가격예측.train_common import evaluate_pooled_predictions, make_pooled_loader, train_pooled_lstm
from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START
tickers, start, end = ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START, date.today().isoformat()
ref, _ = build_merged_dataset_v2(tickers[0], start, end); te, ve = compute_global_split_dates(ref.index)
B = {t: c.compute_ticker_baselines(t, start, end, te, ve) for t in tickers}
def bf(ticker, s, e, precomputed_indicators=None, true_kospi=None):
    b = B[ticker]
    return build_merged_dataset_v2_volatility_hybrid(ticker, s, e, te, precomputed_sigma=b["sigma_full"], garch_params=b["params"],
                                                     precomputed_indicators=precomputed_indicators, true_kospi=true_kospi)
sp, fc, _, _ = build_pooled_sequences(tickers, start, end, lookback=c.LOOKBACK, build_fn=bf)
Xtr, ytr, tidtr, dtr, tntr = sp["train"]; Xva, yva, tidva, dva, tnva = sp["val"]
sc = FeatureScaler().fit(Xtr); Xtr_s = sc.transform(Xtr).astype(np.float32); Xva_s = sc.transform(Xva).astype(np.float32)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
r = train_pooled_lstm(Xtr_s, ytr, tidtr, Xva_s, yva, tidva, len(fc), c.LOOKBACK, len(tickers), c.EMBEDDING_DIM, c.D_MODEL, c.NHEAD,
                      c.NUM_LAYERS, c.DIM_FEEDFORWARD, c.DROPOUT, c.BATCH_SIZE, c.LR, c.WEIGHT_DECAY, c.SMOKE_EPOCHS, c.MAX_EPOCHS,
                      c.PATIENCE, dev, seed=42, verbose=False)
pva = evaluate_pooled_predictions(r["model"], r["val_loader"], dev)[0]
ptr = evaluate_pooled_predictions(r["model"], make_pooled_loader(Xtr_s, ytr, tidtr, c.BATCH_SIZE, shuffle=False), dev)[0]
ref_va = np.load(os.path.join(OUT, "lstm_reg_do0.3_wd0.0001.npy"))
print(f"val 예측 재현: max|차이|={np.abs(pva - ref_va).max():.2e} | best_epoch={r['best_epoch']}")
np.save(os.path.join(OUT, "lstm_prod_train_pred.npy"), ptr); np.save(os.path.join(OUT, "lstm_prod_val_pred.npy"), pva)
pd.DataFrame({"ticker": tntr, "date": pd.to_datetime(dtr), "y": ytr}).to_parquet(os.path.join(OUT, "lstm_prod_train_keys.parquet"))
print("DONE")
