# 상위 P% 예측에만 배율 M 곱하는 후처리 — 임계값은 train 예측 분위수. val 평가(게이트·①/③ 가격 %오차·std 비율).
import os, sys
sys.path.insert(0, r"C:\kch_Final_prj"); os.chdir(r"C:\kch_Final_prj")
OUT = os.path.dirname(os.path.abspath(__file__))
import numpy as np, pandas as pd
from constants import ACTIVE_TICKERS
ptr = np.load(os.path.join(OUT, "lstm_prod_train_pred.npy"))
k = pd.read_parquet(os.path.join(OUT, "lstm_reg_keys.parquet")); k["lstm"] = np.load(os.path.join(OUT, "lstm_prod_val_pred.npy"))
v = pd.read_parquet(os.path.join(OUT, "val_diag_rows.parquet"))[["ticker", "date", "y", "garch", "sma20", "parkinson", "spike"]]
b = np.load(os.path.join(OUT, "price_level_preds_B_15feats+close_prev.npz"), allow_pickle=True)
Bp = pd.DataFrame({"ticker": np.array(ACTIVE_TICKERS)[b["tid"]], "date": pd.to_datetime(b["dates"]), "actual": b["actual"], "close_prev": b["close_prev"]})
full = k.merge(v, on=["ticker", "date"], validate="one_to_one").merge(Bp, on=["ticker", "date"], validate="one_to_one")
# 운영 게이트: 하이브리드는 val 전체 행 종목평균 RMSE vs baseline 종목평균(운영 값 그대로)
BASE = {"garch": 2.1450157077290175, "sma20": 2.017747666579522, "parkinson": 2.00130669608661}
def gate(col): return float(np.mean([np.sqrt(np.mean((g[col] - g.y) ** 2)) for _, g in full.groupby("ticker")]))
d = full.dropna(subset=["garch", "sma20", "parkinson"]).reset_index(drop=True)
top = d.y > np.quantile(d.y, 0.8); sp = d.spike.fillna(False).astype(bool); G1, G3 = (top & sp).values, (~top).values
sgn = np.sign(np.log(d.actual / d.close_prev))
def pct(col, m):
    e = d.close_prev * np.exp(sgn * d[col] / 100); err = (e - d.actual)[m]; return float(np.sqrt(np.mean((err / d.actual[m] * 100) ** 2)))
def ptk(col, m): return float(np.mean([np.sqrt(np.mean((g[col] - g.y) ** 2)) for _, g in d[m].groupby("ticker")]))
bl3 = {c: ptk(c, G3) for c in ("garch", "sma20", "parkinson")}
rows = []
def add(name, col, P=None, M=None, th=None):
    full_share = float((full[col] != full.lstm).mean()) if col != "lstm" else 0.0
    g = gate(col)
    rows.append({"조합": name, "P%": P, "M": M, "임계값": th, "전환비율(val)": round(full_share, 3),
                 "①전환": round(float((d[col] != d.lstm)[G1].mean()), 3), "③전환": round(float((d[col] != d.lstm)[G3].mean()), 3),
                 "게이트RMSE": round(g, 4), "게이트통과": all(g < x for x in BASE.values()),
                 "①가격%오차": round(pct(col, G1), 4), "③가격%오차": round(pct(col, G3), 4),
                 "③게이트": all(ptk(col, G3) < x for x in bl3.values()), "std비율": round(float(full[col].std() / full.y.std()), 3)})
add("LSTM 원본", "lstm")
for P in (10, 20, 30):
    th = float(np.quantile(ptr, 1 - P / 100))
    for M in (1.1, 1.2, 1.5, 2.0, 3.0):
        col = f"P{P}_M{M}"
        for frame in (full, d):
            frame[col] = np.where(frame.lstm >= th, frame.lstm * M, frame.lstm)
        add(col, col, P, M, round(th, 3))
T = pd.DataFrame(rows); pd.set_option("display.width", 250)
print(f"참고: ① GARCH+오라클 4.3664% | ③ baseline 종목평균 RMSE {({c: round(x, 4) for c, x in bl3.items()})}")
print(T.to_string(index=False))
T.to_csv(r"C:/kch_Final_prj/발표자료/postproc_topP_scaling.csv", index=False, float_format="%.4f")
