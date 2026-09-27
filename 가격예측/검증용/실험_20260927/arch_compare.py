# LSTM vs Transformer 아키텍처 비교(val만, test 미사용). DB/MLflow/체크포인트 저장 없음.
# train_daily_pooled_model의 데이터 준비(split·15피처·scaler)를 그대로 재사용하기 위해 lstm
# train_fn을 여러 시드를 도는 래퍼로 잠시 바꿔 끼운다.
import json, os, sys, time
os.environ["MODEL_ARCHITECTURE"] = "lstm"
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
sys.path.insert(0, r"C:\kch_Final_prj"); os.chdir(r"C:\kch_Final_prj")
from datetime import date
import 가격예측.가격예측_변동성_공통 as c
from 가격예측.train_common import evaluate_pooled_predictions, train_pooled_lstm, train_pooled_transformer
from 가격예측.garch_baseline import rmse_mae
from constants import ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START

RUNS = [("transformer", 42, train_pooled_transformer)] + [("lstm", s, train_pooled_lstm) for s in (42, 43, 44)]
results = []

def multi_seed_fn(*args, seed=42, **kw):
    device = args[-1]
    keep = None
    for arch, s, fn in RUNS:
        t0 = time.time()
        r = fn(*args, seed=s, **{**kw, "label": f"{arch}-s{s}"})
        p, a, _ = evaluate_pooled_predictions(r["model"], r["val_loader"], device)
        rm, _ = rmse_mae(p, a)
        n_params = sum(x.numel() for x in r["model"].parameters())
        rec = dict(arch=arch, seed=s, val_rmse=float(rm), best_epoch=r["best_epoch"],
                   best_val_loss=float(r["best_val_loss"]), pred_std=float(p.std()), target_std=float(a.std()),
                   n_params=n_params, sec=round(time.time() - t0))
        print("RESULT", json.dumps(rec), flush=True)
        results.append(rec)
        if arch == "lstm" and s == 42:
            keep = r
    return keep

c._ARCHITECTURES["lstm"]["train_fn"] = multi_seed_fn
out = c.train_daily_pooled_model(ACTIVE_TICKERS, STOCK_INITIAL_LOAD_START, date.today().isoformat())
g = out["gate"]
print(f"\nsplit: train_end={out['train_end'].date()} val_end={out['val_end'].date()} n_train={out['n_train']} n_val={out['n_val']}")
print(f"baselines: GARCH={g['garch_rmse']:.4f} SMA20={g['sma_rmse']:.4f} Parkinson={g['parkinson_rmse']:.4f}")
for r in results:
    rm = r["val_rmse"]
    print(f"{r['arch']:<12} seed={r['seed']} val_rmse={rm:.4f} best_ep={r['best_epoch']} "
          f"vsGARCH={rm < g['garch_rmse']} vsSMA20={rm < g['sma_rmse']} vsPark={rm < g['parkinson_rmse']} "
          f"std_ratio={r['pred_std']/r['target_std']:.3f} params={r['n_params']} {r['sec']}s")
