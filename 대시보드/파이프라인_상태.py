# 대시보드/파이프라인_상태.py — 패널 A(파이프라인 상태)의 출처 조회와 14일 × 단계 격자 복원(2026-10-18)
# 출처: GitHub Actions(인증 없음, 새로고침당 1요청), 로컬 로그(PIPELINE_LOG_DIR), MLflow(sqlite 읽기 전용, MLFLOW_DB_PATH),
#       운영 DB(model_predictions_realized·model_registry·daily_stock_prices·market_indicators, 읽기 전용).
# 열(사이클 날짜) 규칙: 시각에서 12시간을 뺀 KST 날짜. 저녁 수집(Actions)과 다음 날 05:00 예측이 같은 열에 모인다
#   (예: 10/17 22:58 Actions, 10/18 05:02 예측 → 10/17 열). 모델 버전(UTC 저장 시각)도 KST로 바꿔 같은 규칙을 쓴다.
# 기록이 없는 칸은 새 상태를 만들지 않고 waiting + "기록 없음", 출처 조회가 실패한 칸은 skipped + "조회 실패".
# 로그의 [collect]·[PRICE-ALIGN] 줄은 2026-10-17부터만 있다(그 전 날짜의 해당 단계는 "기록 없음").

import glob
import json
import os
import re
import sqlite3
import traceback
import urllib.request
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

import 데이터

KST = 데이터.KST
CYCLE_SHIFT = timedelta(hours=12)
LOG_DIR = os.environ.get("PIPELINE_LOG_DIR", os.path.join(데이터.ROOT, "logs"))
MLFLOW_DB = os.environ.get("MLFLOW_DB_PATH", os.path.join(데이터.ROOT, "mlflow.db"))
GITHUB_RUNS_URL = ("https://api.github.com/repos/cksgml0614/kch_Final_prj/actions/workflows/"
                   "daily_data_collection.yml/runs?per_page=20")
EXP_DAILY = "일일_자동화"
EXP_OLD = "old_daily_until_20261011"  # 2026-10-11까지의 운영 run(학습·MLflow 칸 복원용)
EXP_TUNING = "주간_튜닝"
N_TICKERS = 100
STAGES = ["① 수집", "② 데이터 반영", "③ 학습", "④ 게이트", "⑤ 예측 저장", "⑥ MLflow", "⑦ registry·실측", "⑧ 튜닝(주 1회)"]
NO_RECORD = ("waiting", "기록 없음")
LOOKUP_FAILED = ("skipped", "조회 실패")


def cycle_day(dt):
    """
    설명: KST 시각을 사이클 날짜(시각 - 12시간의 날짜)로 바꾼다.
    주의 사항: dt는 tz-aware KST datetime이어야 한다.
    입력 타입->반환 타입: (datetime) -> date
    """
    return (dt - CYCLE_SHIFT).date()


def hm(dt):
    """
    설명: 시각을 'HH:MM'으로 만든다.
    주의 사항: 없음.
    입력 타입->반환 타입: (datetime) -> str
    """
    return dt.strftime("%H:%M")


# ── 출처 조회(각각 실패하면 예외를 던지고, 호출부가 "조회 실패"로 바꾼다) ──────────────────────

@st.cache_data(ttl=데이터.TTL, show_spinner=False)
def github_runs():
    """
    설명: 일일 수집 워크플로의 최근 run 20개를 인증 없이 한 번 요청해 DataFrame으로 돌려준다.
    주의 사항: 외부 요청 1회(인증 없는 한도 시간당 60회 — ttl 300 캐시로 새로고침당 1회). 실패하면 예외.
    입력 타입->반환 타입: () -> pd.DataFrame
    """
    req = urllib.request.Request(GITHUB_RUNS_URL, headers={"Accept": "application/vnd.github+json",
                                                           "User-Agent": "kch-dashboard"})
    with urllib.request.urlopen(req, timeout=10) as r:
        runs = json.load(r).get("workflow_runs", [])
    rows = []
    for x in runs:
        started = pd.Timestamp(x.get("run_started_at") or x["created_at"]).tz_convert(KST).to_pydatetime()
        rows.append({"started": started, "status": x["status"], "conclusion": x.get("conclusion"),
                     "event": x["event"], "attempt": x.get("run_attempt")})
    return pd.DataFrame(rows, columns=["started", "status", "conclusion", "event", "attempt"])


@st.cache_data(ttl=데이터.TTL, show_spinner=False)
def mlflow_runs():
    """
    설명: mlflow.db(sqlite, 읽기 전용)에서 운영·옛 운영·튜닝 실험의 run과 태그·지표를 읽어 (runs, 있는 실험 이름 목록)을 돌려준다.
    주의 사항: 파일 읽기만(mode=ro). MLflow 내부 테이블 구조에 기대므로 MLflow 버전이 바뀌면 조회 실패가 될 수 있다.
    입력 타입->반환 타입: () -> (pd.DataFrame, list[str])
    """
    if not os.path.exists(MLFLOW_DB):
        raise FileNotFoundError("mlflow.db 없음")
    con = sqlite3.connect(f"file:{MLFLOW_DB}?mode=ro", uri=True, timeout=5)
    try:
        exps = dict(con.execute("SELECT name, experiment_id FROM experiments WHERE lifecycle_stage = 'active' "
                                "AND name IN (?, ?, ?)", (EXP_DAILY, EXP_OLD, EXP_TUNING)).fetchall())
        names = {v: k for k, v in exps.items()}
        rows = []
        if exps:
            marks = ",".join("?" * len(exps))
            for run_uuid, eid, status, start in con.execute(
                    f"SELECT run_uuid, experiment_id, status, start_time FROM runs WHERE lifecycle_stage = 'active' "
                    f"AND experiment_id IN ({marks})", list(exps.values())).fetchall():
                tags = dict(con.execute("SELECT key, value FROM tags WHERE run_uuid = ?", (run_uuid,)).fetchall())
                mets = dict(con.execute("SELECT key, value FROM latest_metrics WHERE run_uuid = ?", (run_uuid,)).fetchall())
                rows.append({"experiment": names[eid], "run_uuid": run_uuid, "status": status,
                             "start": datetime.fromtimestamp(start / 1000, KST),
                             "version": tags.get("version"), "prediction_date": tags.get("prediction_date"),
                             "gate_passed": mets.get("gate_passed"), "hybrid": mets.get("hybrid_rmse_tickermean"),
                             "garch": mets.get("garch_rmse"), "sma": mets.get("sma_rmse"),
                             "parkinson": mets.get("parkinson_rmse"), "n_success": mets.get("n_predictions_success")})
    finally:
        con.close()
    return pd.DataFrame(rows), sorted(exps)


LOG_NAME = re.compile(r"daily_(\d{8})_(\d{4})\.log$")


@st.cache_data(ttl=데이터.TTL, show_spinner=False)
def log_records():
    """
    설명: 로컬 일일 로그(daily_YYYYMMDD_HHMM.log)에서 단계별 마커를 읽어 파일마다 dict 하나로 돌려준다.
    주의 사항: 파일 읽기만. 로그 폴더가 없으면 예외.
    입력 타입->반환 타입: () -> list[dict]
    """
    if not os.path.isdir(LOG_DIR):
        raise FileNotFoundError("로그 폴더 없음")
    out = []
    for path in sorted(glob.glob(os.path.join(LOG_DIR, "daily_*.log"))):
        m = LOG_NAME.search(path)
        if not m:
            continue
        start = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M").replace(tzinfo=KST)
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        rec = {"file": os.path.basename(path), "start": start, "stock_rc": None, "indicator_rc": None,
               "price_align": None, "train_version": None, "gate": None, "n_pred": None,
               "mlflow_skip": "[MLFLOW-SKIP]" in text, "mlflow_fail": "MLflow 로깅 실패" in text,
               "predict": "failed" if "volatility predict FAILED" in text else
                          "done" if "volatility predict done" in text else None}
        if (x := re.search(r"\[collect\] stock collect \w+ rc=(\d+)", text)):
            rec["stock_rc"] = int(x.group(1))
        if (x := re.search(r"\[collect\] indicator collect \w+ rc=(\d+)", text)):
            rec["indicator_rc"] = int(x.group(1))
        if (x := re.search(r"^\[PRICE-ALIGN\][^\n]*", text, re.M)):
            rec["price_align"] = x.group(0)
        if (x := re.search(r"학습 완료\(버전 (\d{8}_\d{6})[^\n]*배포 게이트 (통과|미통과)", text)):
            rec["train_version"], rec["gate"] = x.group(1), x.group(2)
        if (x := re.search(r"종목별 예측 결과 \((\d+)/(\d+) 성공", text)):
            rec["n_pred"] = int(x.group(1))
        out.append(rec)
    return out


@st.cache_data(ttl=데이터.TTL, show_spinner=False)
def db_state(since):
    """
    설명: since 이후 사이클 복원에 필요한 DB 사실(데이터가 있는 날짜, 최신일, 버전별 예측 행·게이트·실측, registry)을 읽는다.
    주의 사항: DB 읽기(읽기 전용 연결). 버전별 n·게이트·실측·target_date는 그 버전의 마지막 prediction_date 기준, n_total은 버전 전체 행.
    입력 타입->반환 타입: (date) -> dict
    """
    stock = 데이터.query("SELECT date FROM daily_stock_prices WHERE date >= %s GROUP BY date", (since,))
    ind = 데이터.query("SELECT date FROM market_indicators WHERE date >= %s GROUP BY date", (since,))
    latest = 데이터.query("""
        SELECT 'stock' AS k, max(date) AS d FROM daily_stock_prices
        UNION ALL SELECT indicator_code, max(date) FROM market_indicators WHERE indicator_code IN ('KOSPI', 'KOSDAQ')
        GROUP BY indicator_code""")
    since_v = (since - timedelta(days=1)).strftime("%Y%m%d")
    preds = 데이터.query("""
        WITH v AS (SELECT model_version, max(prediction_date) AS pd, count(*) AS n_total FROM model_predictions
                   WHERE model_version >= %s GROUP BY model_version)
        SELECT p.model_version, p.prediction_date, v.n_total, count(*) AS n, sum(p.gate_passed::int) AS n_gate,
               count(*) FILTER (WHERE p.realized_status = 'confirmed') AS n_conf, max(p.target_date) AS target_date
        FROM model_predictions_realized p JOIN v ON v.model_version = p.model_version AND v.pd = p.prediction_date
        GROUP BY 1, 2, 3""", (since_v,))
    reg = 데이터.query("SELECT model_version, source FROM model_registry WHERE model_version >= %s", (since_v,))
    return {"stock_dates": set(stock["date"]), "indicator_dates": set(ind["date"]),
            "latest": dict(zip(latest["k"], latest["d"])),
            "preds": {r["model_version"]: r for r in preds.to_dict("records")},
            "registry": dict(zip(reg["model_version"], reg["source"]))}


def load_sources(since):
    """
    설명: 네 출처를 각각 조회하고, 실패한 출처는 None으로 두어 {이름: 값}을 돌려준다(앱은 멈추지 않는다).
    주의 사항: 실패 원인은 콘솔에만 출력한다.
    입력 타입->반환 타입: (date) -> dict
    """
    src = {}
    for name, fn in (("github", github_runs), ("mlflow", mlflow_runs), ("logs", log_records), ("db", lambda: db_state(since))):
        try:
            src[name] = fn()
        except Exception:
            traceback.print_exc()
            src[name] = None
    return src


# ── 격자 복원 ──────────────────────────────────────────────────────────────────────

def _version_day(v):
    """모델 버전(UTC 저장 시각)의 사이클 날짜. 형식이 다르면 None."""
    t = 데이터.version_kst(v)
    return cycle_day(t) if t else None


def _cell_collect(d, src):
    """① 수집: Actions 결과 우선, 없으면 로컬 [collect] rc. Actions 실패여도 그날 데이터가 DB에 있으면 경고."""
    gh, logs, db = src["github"], src["logs"], src["db"]
    present = db is not None and (d in db["stock_dates"] or d in db["indicator_dates"])
    local = [r for r in (logs or []) if cycle_day(r["start"]) == d and r["stock_rc"] is not None]
    loc_txt = f"로컬 rc 주가 {local[-1]['stock_rc']}·지표 {local[-1]['indicator_rc']}" if local else "로컬 수집 기록 없음"
    acts = gh[gh["started"].map(cycle_day) == d].sort_values("started") if gh is not None and len(gh) else None
    if acts is not None and len(acts):
        a = acts.iloc[-1]
        tip = f"Actions {a['event']} {a['started']:%m/%d %H:%M} 시작 · {a['conclusion'] or a['status']} · {loc_txt}"
        if a["status"] != "completed":
            return "running", f"진행 {hm(a['started'])}", tip
        if a["conclusion"] == "success":
            return "success", f"성공 {hm(a['started'])}", tip
        if present:
            return "warning", "실패·데이터 있음", tip + " · 그날 데이터가 DB에 있음(원인 미확인)"
        return "failure", "실패", tip
    if local:
        ok = local[-1]["stock_rc"] == 0 and local[-1]["indicator_rc"] == 0
        if ok:
            return "success", "로컬 rc=0", loc_txt
        return ("warning", "로컬 rc≠0·데이터 있음", loc_txt) if present else ("failure", "로컬 수집 실패", loc_txt)
    if gh is None:
        return (*LOOKUP_FAILED, "GitHub Actions 조회 실패")
    return (*NO_RECORD, "이 날짜의 Actions run(최근 20개)·로컬 수집 기록 없음")


def _cell_align(d, src):
    """② 데이터 반영: 그 사이클 마지막 로그의 [PRICE-ALIGN] 줄."""
    if src["logs"] is None:
        return (*LOOKUP_FAILED, "로그 조회 실패")
    recs = [r for r in src["logs"] if cycle_day(r["start"]) == d and r["price_align"]]
    if not recs:
        return (*NO_RECORD, "[PRICE-ALIGN] 줄 없음(2026-10-17 이전 로그에는 없다)")
    line = recs[-1]["price_align"]
    m = re.search(r"(\d+)/(\d+)종목 일치", line)
    if "이탈" in line:
        return "warning", "이탈 제외", line
    if m and m.group(1) == m.group(2):
        return "success", f"{m.group(1)}/{m.group(2)} 일치", line
    return "failure", "정렬 실패", line


def _day_run(d, src):
    """그 사이클의 마지막 MLflow run(운영·옛 운영 실험)과 마지막 로그."""
    run = None
    if src["mlflow"] is not None:
        runs = src["mlflow"][0]
        if len(runs):
            r = runs[(runs["experiment"] != EXP_TUNING) & (runs["start"].map(cycle_day) == d)].sort_values("start")
            run = r.iloc[-1] if len(r) else None
    logs = [r for r in (src["logs"] or []) if cycle_day(r["start"]) == d]
    return run, (logs[-1] if logs else None)


def _day_version(d, src, run, log):
    """그 사이클에 만들어진 마지막 모델 버전(run 태그·로그·DB 예측 행·registry 중 가장 늦은 것)."""
    cands = []
    if run is not None and run["version"]:
        cands.append(run["version"])
    if log and log["train_version"]:
        cands.append(log["train_version"])
    if src["db"] is not None:
        cands += [v for v in list(src["db"]["preds"]) + list(src["db"]["registry"]) if _version_day(v) == d]
    return max(cands) if cands else None


def _cell_train(run, log, src):
    """③ 학습: MLflow run 상태, 없으면 로그의 학습 완료 줄."""
    if run is not None:
        st_ = {"FINISHED": "success", "FAILED": "failure", "RUNNING": "running"}.get(run["status"], "warning")
        return st_, f"{hm(run['start'])}" if st_ == "success" else run["status"], f"MLflow run {run['experiment']}"
    if log is not None and log["train_version"]:
        return "success", "완료(로그)", log["file"]
    if log is not None and log["predict"] == "failed":
        return "failure", "실패(로그)", log["file"]
    if src["mlflow"] is None and src["logs"] is None:
        return (*LOOKUP_FAILED, "MLflow·로그 조회 실패")
    return (*NO_RECORD, "MLflow run·로그 없음")


def _cell_gate(v, run, log, src):
    """④ 게이트: 그 버전 예측 행의 gate_passed, 없으면 MLflow 지표·로그."""
    db = src["db"]
    if v and db is not None and v in db["preds"]:
        p = db["preds"][v]
        return ("success", "통과", f"{v} · gate_passed {p['n_gate']}/{p['n']}") if p["n_gate"] == p["n"] else \
            ("failure", "미통과", f"{v} · gate_passed {p['n_gate']}/{p['n']}")
    if run is not None and run["gate_passed"] is not None and not pd.isna(run["gate_passed"]):
        return ("success", "통과", "MLflow gate_passed") if run["gate_passed"] >= 1 else ("failure", "미통과", "MLflow gate_passed")
    if log is not None and log["gate"]:
        return ("success" if log["gate"] == "통과" else "failure"), f"{log['gate']}(로그)", log["file"]
    if db is None:
        return (*LOOKUP_FAILED, "DB 조회 실패")
    return (*NO_RECORD, "게이트 기록 없음")


def _count_status(n):
    """예측 저장 행 수 판정: 100이면 성공, 90 이상이면 경고, 미만이면 실패."""
    return "success" if n >= N_TICKERS else "warning" if n >= 90 else "failure"


def _cell_save(v, run, log, src):
    """⑤ 예측 저장: 그 버전 마지막 prediction_date의 행 수. 나중 실행이 같은 키를 덮어써 행이 줄었으면 실행 기록의 저장 수."""
    recorded = None  # 그 버전 실행이 기록한 저장 성공 수(MLflow 지표 또는 로그)
    if run is not None and run["version"] == v and run["n_success"] is not None and not pd.isna(run["n_success"]):
        recorded = int(run["n_success"])
    elif log is not None and log["train_version"] == v and log["n_pred"] is not None:
        recorded = log["n_pred"]
    db = src["db"]
    p = db["preds"].get(v) if v and db is not None else None
    if p is not None and (recorded is None or p["n_total"] >= recorded):
        return (_count_status(p["n"]), f"{p['n']}행",
                f"{v} · 기준일 {p['prediction_date']} 행 {p['n']} (버전 전체 {p['n_total']}행)")
    if recorded is not None and v:
        left = p["n_total"] if p is not None else 0
        return (_count_status(recorded), f"{recorded}(덮어씀 {recorded - left})",
                f"{v} 실행 기록 저장 {recorded}행 중 {recorded - left}행을 이후 같은 키 재실행이 덮어썼다(남은 {left}행)")
    if db is None:
        return (*LOOKUP_FAILED, "DB 조회 실패")
    return (*NO_RECORD, "예측 저장 기록 없음")


def _cell_mlflow(run, log, src):
    """⑥ MLflow: run 유무·상태, 없으면 로그의 건너뜀·실패 줄."""
    if run is not None:
        return ("success", f"{hm(run['start'])}", run["experiment"]) if run["status"] == "FINISHED" else \
            ("failure", run["status"], run["experiment"])
    if log is not None and log["mlflow_fail"]:
        return "failure", "기록 실패(로그)", log["file"]
    if log is not None and log["mlflow_skip"]:
        return "warning", "건너뜀(로그)", log["file"]
    if src["mlflow"] is None:
        return (*LOOKUP_FAILED, "MLflow 조회 실패")
    return (*NO_RECORD, "MLflow run 없음")


def _cell_registry(v, src):
    """⑦ registry·실측: model_registry 등록과 실측 확정 여부."""
    db = src["db"]
    if db is None:
        return (*LOOKUP_FAILED, "DB 조회 실패")
    if not v:
        return (*NO_RECORD, "모델 버전 없음")
    if v not in db["registry"]:
        return "warning", "미등록", v
    p = db["preds"].get(v)
    if p is None:
        return "success", "등록·덮어씀", f"{v} · 예측 행은 이후 실행이 덮어썼다"
    if p["n_conf"] == p["n"]:
        return "success", "등록·실측", f"{v} · 실측 {p['n_conf']}/{p['n']}"
    # 대기 이유: 그 버전 마지막 기준일 예측의 대상일(target_date) 종가가 아직 없다. 대상일이 휴장이면 실측은 다음 거래일.
    return ("waiting", "실측 대기", f"평가일 {데이터.fmt_day(p['target_date'])} 종가 확정 후 반영(휴장이면 다음 거래일) · "
            f"{v} · 실측 {p['n_conf']}/{p['n']}")


def _cell_tuning(d, src):
    """⑧ 튜닝: MLflow 실험 주간_튜닝의 run."""
    if src["mlflow"] is None:
        return (*LOOKUP_FAILED, "MLflow 조회 실패")
    runs, exps = src["mlflow"]
    if EXP_TUNING not in exps:
        return (*NO_RECORD, f"MLflow 실험 '{EXP_TUNING}' 없음")
    r = runs[(runs["experiment"] == EXP_TUNING) & (runs["start"].map(cycle_day) == d)] if len(runs) else runs
    if not len(r):
        return (*NO_RECORD, "튜닝 run 없음")
    last = r.sort_values("start").iloc[-1]
    return ("success" if last["status"] == "FINISHED" else "failure"), f"{last['status']} {hm(last['start'])}", EXP_TUNING


def build_grid(days, src):
    """
    설명: 사이클 날짜 목록 × 8단계의 (상태, 글자, 설명) 격자를 {단계: [칸...]}으로 만든다.
    주의 사항: 기록 없는 칸은 waiting "기록 없음", 출처 조회 실패 칸은 skipped "조회 실패".
    입력 타입->반환 타입: (list[date], dict) -> dict
    """
    grid = {s: [] for s in STAGES}
    for d in days:
        run, log = _day_run(d, src)
        v = _day_version(d, src, run, log)
        cells = [_cell_collect(d, src), _cell_align(d, src), _cell_train(run, log, src), _cell_gate(v, run, log, src),
                 _cell_save(v, run, log, src), _cell_mlflow(run, log, src), _cell_registry(v, src), _cell_tuning(d, src)]
        for s, c in zip(STAGES, cells):
            grid[s].append(c)
    return grid


def today_cycle(now=None):
    """
    설명: 지금 시각의 사이클 날짜와 다음 05:00 실행 시각을 돌려준다.
    주의 사항: 없음.
    입력 타입->반환 타입: (datetime|None) -> (date, datetime)
    """
    now = now or datetime.now(KST)
    nxt = now.replace(hour=5, minute=0, second=0, microsecond=0)
    if nxt <= now:
        nxt += timedelta(days=1)
    return cycle_day(now), nxt


def last_finished_daily_run(src):
    """
    설명: 운영 실험(일일_자동화)의 마지막 FINISHED run을 돌려준다. 없으면 None, MLflow 조회 실패면 예외.
    주의 사항: 없음.
    입력 타입->반환 타입: (dict) -> pd.Series|None
    """
    if src["mlflow"] is None:
        raise RuntimeError("MLflow 조회 실패")
    runs = src["mlflow"][0]
    if not len(runs):
        return None
    r = runs[(runs["experiment"] == EXP_DAILY) & (runs["status"] == "FINISHED")].sort_values("start")
    return r.iloc[-1] if len(r) else None


def days_back(end, n):
    """
    설명: end를 마지막으로 하는 n일의 날짜 목록(오름차순).
    주의 사항: 없음.
    입력 타입->반환 타입: (date, int) -> list[date]
    """
    return [end - timedelta(days=i) for i in range(n - 1, -1, -1)]

