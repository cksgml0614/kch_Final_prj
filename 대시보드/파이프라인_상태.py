# 대시보드/파이프라인_상태.py — 패널 A(파이프라인 상태)의 출처 조회와 14일 × 단계 격자 복원(2026-10-18)
# 출처: GitHub Actions(인증 없음, 새로고침당 1요청), 로컬 로그(PIPELINE_LOG_DIR), MLflow(sqlite 읽기 전용, MLFLOW_DB_PATH),
#       운영 DB(model_predictions_realized·model_registry·daily_stock_prices·market_indicators, 읽기 전용).
# 묶음(사이클) 규칙: 시각에서 12시간을 뺀 KST 날짜(cycle_day). 저녁 수집(Actions)·저녁 수동 실행과 다음 날 05:00 예측이
#   한 사이클로 묶인다(경계 = 매일 12:00). 모델 버전(UTC 저장 시각)도 KST로 바꿔 같은 규칙을 쓴다.
# 열 날짜(2026-10-25 사람 결정): 사이클 날짜 = 그 사이클의 수집 대상일(거래일 후보)이다(예: 10/23(금) 저녁 수집, 10/24(토) 05:00
#   예측 → 10/23 열). 그래서 토·일 열이 휴장 열이다. 10/18~10/25에는 열 이름을 아침 05:00 실행 날짜(사이클 날짜 + 1일)로 썼는데,
#   일·월요일 열이 휴장으로 보여 헷갈려 되돌렸다. 실행 날짜는 열 머리·칸 툴팁에 적는다.
# 늦은 보충 실행(2026-10-25): 실행(MLflow run·로그·모델 버전)은 시각으로 사이클을 정하되, 실제로 처리한 예측 기준일이 그 사이클에서
#   기대되는 거래일(사이클 날짜 이하 최신 주가일)보다 이르면 그 기준일 열로 옮긴다(run_column). 예: 10/14 18:04 실행은 10/13분을
#   처리했으므로 10/13 열. 주말 재실행은 기준일 = 기대 거래일(금)이라 토·일 열에 남는다. ① 수집은 시각 그대로다.
# 예정 열(2026-10-25): 격자 마지막 열은 현재 사이클 다음 날이다. 기록 없는 칸은 waiting "예정"(휴장이면 휴장).
# 기록이 없는 칸은 새 상태를 만들지 않고 waiting + "기록 없음", 출처 조회가 실패한 칸은 skipped + "조회 실패".
# 휴장 열(2026-10-25 개정, 처음 규칙은 2026-10-18): 열 날짜(수집 대상일)가 휴장이면 그 열 전체를 휴장으로 본다.
#   휴장 = 토·일(요일로 바로) 또는 평일 공휴일(데이터로 증명: 주가·KOSPI 행 없음 + 뒤 거래일 주가 있음, holiday_reason).
#   ①② 성공·기록 없음 → skipped "휴장", 경고·실패(①) → warning "휴장·경고"(원래 판정은 툴팁). ② 정렬 실패는 그대로 실패.
#   ③~⑦ 실행 없음 → skipped "휴장". 실행이 있고 같은 예측 기준일을 앞 실행이 이미 계산했으면 성공·대기 칸은 skipped "재실행",
#   실패·경고·진행 칸은 그대로 둔다(덮어쓴 예측이 지금 서빙되므로 문제는 가리지 않는다). 앞 실행이 없으면(보충 실행) 원래 판정.
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


def column_day(dt):
    """
    설명: KST 시각을 격자 열 날짜로 바꾼다(열 날짜 = 사이클 날짜 = 수집 대상일).
    주의 사항: 묶음 경계는 cycle_day와 같다(매일 12:00).
    입력 타입->반환 타입: (datetime) -> date
    """
    return cycle_day(dt)


def cycle_of(column):
    """
    설명: 격자 열 날짜의 사이클 날짜(= 수집 대상일, 열 날짜와 같다).
    주의 사항: 열 날짜와 사이클 날짜가 다르던 시절(2026-10-18~25)의 호출부를 그대로 두려고 남긴 함수다.
    입력 타입->반환 타입: (date) -> date
    """
    return column


def run_day(column):
    """
    설명: 열(수집 대상일)의 아침 05:00 실행 날짜(다음 날).
    주의 사항: 없음.
    입력 타입->반환 타입: (date) -> date
    """
    return column + timedelta(days=1)


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
                             "run_kind": tags.get("run_kind"), "state": tags.get("state"),
                             "tuned_params_id": tags.get("tuned_params_id"),
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
    kospi = 데이터.query("SELECT date FROM market_indicators WHERE indicator_code = 'KOSPI' AND date >= %s", (since,))
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
    return {"stock_dates": set(stock["date"]), "indicator_dates": set(ind["date"]), "kospi_dates": set(kospi["date"]),
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

def _as_date(x):
    """기준일 값(date·문자열·Timestamp, 섞인 태그 '2026-10-08,2026-10-12'는 가장 늦은 날)을 date로. 없으면 None."""
    if x is None or (not isinstance(x, str) and pd.isna(x)) or x == "":
        return None
    if isinstance(x, str):
        return max(pd.Timestamp(p.strip()).date() for p in x.split(",") if p.strip())
    return pd.Timestamp(x).date()


def _version_pd(v, src):
    """모델 버전의 예측 기준일(DB 예측 행 → 같은 버전 MLflow 태그). 모르면 None."""
    if not v:
        return None
    db = src["db"]
    if db is not None and v in db["preds"]:
        return _as_date(db["preds"][v]["prediction_date"])
    if src["mlflow"] is not None and len(src["mlflow"][0]):
        runs = src["mlflow"][0]
        r = runs[runs["version"] == v]
        if len(r):
            return _as_date(r.iloc[-1]["prediction_date"])
    return None


def run_column(t, pdate, db):
    """
    설명: 실행 시각 t와 처리한 예측 기준일 pdate로 그 실행의 열(사이클 날짜)을 정한다.
    주의 사항: 기본은 시각 - 12시간의 날짜. 기준일이 그 사이클의 기대 거래일(사이클 날짜 이하 최신 주가일)보다 이르면 늦은 보충
      실행이므로 기준일 열로 옮긴다. 기준일이나 주가 날짜를 모르면 시각 그대로.
    입력 타입->반환 타입: (datetime, date|None, dict|None) -> date
    """
    c = cycle_day(t)
    if pdate is None or db is None:
        return c
    past = [x for x in db["stock_dates"] if x <= c]
    return pdate if past and pdate < max(past) else c


def _version_day(v, src):
    """모델 버전(UTC 저장 시각)의 열 — 늦은 보충 실행이면 기준일 열(run_column). 형식이 다르면 None."""
    t = 데이터.version_kst(v)
    return run_column(t, _version_pd(v, src), src["db"]) if t else None


def _log_day(r, src):
    """로그 파일의 열 — [PRICE-ALIGN] 공통 최신일(없으면 학습 버전의 기준일)로 run_column."""
    m = re.search(r"공통 최신일 (\d{4}-\d{2}-\d{2})", r["price_align"] or "")
    pdate = _as_date(m.group(1)) if m else _version_pd(r["train_version"], src)
    return run_column(r["start"], pdate, src["db"])


def _mlflow_day(row, src):
    """MLflow 운영 run의 열 — 태그 prediction_date(없으면 버전의 기준일)로 run_column."""
    pdate = _as_date(row["prediction_date"]) or _version_pd(row["version"], src)
    return run_column(row["start"], pdate, src["db"])


def non_trading_evidence(x, db):
    """
    설명: 날짜 x가 휴장일(거래일 아님)임을 데이터로 증명할 수 있으면 그 근거 문구를, 아니면 None을 돌려준다.
    주의 사항: 거래일 캘린더는 수집기와 같다 — daily_stock_prices에 있는 날짜(지표_공통.check_daily_indicator_lag).
      휴장 = x에 주가 행도 ECOS KOSPI 행도 없고, x보다 뒤 거래일의 주가가 이미 있음(증분 수집은 MAX(date)+1부터라
      그 뒤 날짜가 있으면 x는 소스에 없던 날이다). 뒤 거래일이 아직 없으면(가장 최근 주말 등) 증명할 수 없어 None.
      요일·공휴일 목록은 쓰지 않는다.
    입력 타입->반환 타입: (date, dict|None) -> str|None
    """
    if db is None or x in db["stock_dates"] or x in db["kospi_dates"]:
        return None
    later = [d for d in db["stock_dates"] if d > x]
    if not later:
        return None
    return f"{데이터.fmt_day(x)} 주가·KOSPI 행 없음, 다음 거래일 {데이터.fmt_day(min(later))} 주가 있음"


def holiday_reason(x, db):
    """
    설명: 날짜 x가 휴장이면 근거 문구를, 아니면(또는 알 수 없으면) None을 돌려준다.
    주의 사항: 토·일은 요일로 바로 휴장(KRX는 주말에 열리지 않는다). 평일 공휴일은 공휴일 목록 없이 데이터로만 증명하므로
      다음 거래일 주가가 들어오기 전(공휴일 다음 날 아침 등)에는 None이다. x에 주가·KOSPI 행이 있으면 거래일.
    입력 타입->반환 타입: (date, dict|None) -> str|None
    """
    if db is not None and (x in db["stock_dates"] or x in db["kospi_dates"]):
        return None
    if x.weekday() >= 5:
        return f"{데이터.fmt_day(x)} 주말"
    return non_trading_evidence(x, db)


def column_holiday(column, src):
    """
    설명: 격자 열(수집 대상일)이 휴장이면 근거 문구, 아니면 None.
    주의 사항: 열 머리 표시용. 판정은 holiday_reason과 같다.
    입력 타입->반환 타입: (date, dict) -> str|None
    """
    return holiday_reason(cycle_of(column), src["db"])


def _cell_collect(d, src):
    """
    ① 수집. 수집 대상일(d)이 휴장이면: 성공·기록 없음 → skipped "휴장", 경고·실패(로컬 rc≠0 포함) → warning "휴장·경고".
    진행 중·조회 실패는 그대로 둔다. 휴장 근거가 없으면 원래 판정.
    """
    status, text, tip = _collect_base(d, src)
    why = holiday_reason(d, src["db"])
    if why is None or status == "running" or (status, text) == LOOKUP_FAILED:
        return status, text, tip
    local = [r for r in (src["logs"] or []) if cycle_day(r["start"]) == d and r["stock_rc"] is not None]
    local_bad = bool(local) and not (local[-1]["stock_rc"] == 0 and local[-1]["indicator_rc"] == 0)
    if status in ("warning", "failure") or local_bad:
        return "warning", "휴장·경고", f"수집 대상일 휴장({why}) — 새 거래일은 없지만 수집이 경고·실패로 끝남 · 원래 판정 {text} · {tip}"
    return "skipped", "휴장", f"수집할 새 거래일 없음({why}) · 원래 판정 {text} · {tip}"


def _collect_base(d, src):
    """
    설명: ① 수집 기본 판정. 그 사이클의 마지막 Actions run 결과를 먼저 보고, 없으면 마지막 로컬 [collect] rc를 본다.
    주의 사항: 실패여도 그날 주가·지표 행이 DB에 있으면 경고. Actions는 최근 20개만 보므로 오래된 열은 로컬 기록에 기댄다.
    입력 타입->반환 타입: (date, dict) -> (str, str, str)
    """
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
    """
    ② 데이터 반영. 수집 대상일(d)이 휴장이면: 성공·기록 없음 → skipped "휴장", 이탈 경고 → warning "휴장·경고".
    정렬 실패·조회 실패는 그대로 둔다(정렬 실패면 그 실행은 예측 없이 멈췄다).
    """
    status, text, tip = _align_base(d, src)
    why = holiday_reason(d, src["db"])
    if why is None or status == "failure" or (status, text) == LOOKUP_FAILED:
        return status, text, tip
    if status == "warning":
        return "warning", "휴장·경고", f"수집 대상일 휴장({why}) · 원래 판정 {text} · {tip}"
    return "skipped", "휴장", f"반영할 새 거래일 없음({why}) · 원래 판정 {text} · {tip}"


def _align_base(d, src):
    """
    설명: ② 데이터 반영 기본 판정 — 그 사이클 마지막 로그의 [PRICE-ALIGN] 줄(전 종목 일치 성공, 이탈 경고, 그 밖은 실패).
    주의 사항: 이 줄은 2026-10-17 로그부터만 있어 그 전 열은 기록 없음.
    입력 타입->반환 타입: (date, dict) -> (str, str, str)
    """
    if src["logs"] is None:
        return (*LOOKUP_FAILED, "로그 조회 실패")
    recs = [r for r in src["logs"] if r["price_align"] and _log_day(r, src) == d]
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
    """
    설명: 그 사이클의 마지막 MLflow run(운영·옛 운영 실험)과 마지막 일일 로그 기록을 돌려준다.
    주의 사항: 튜닝 실험 run은 뺀다. 없거나 출처가 실패면 None.
    입력 타입->반환 타입: (date, dict) -> (pd.Series|None, dict|None)
    """
    run = None
    if src["mlflow"] is not None:
        runs = src["mlflow"][0]
        if len(runs):
            r = runs[runs["experiment"] != EXP_TUNING]
            r = r[[_mlflow_day(x, src) == d for _, x in r.iterrows()]].sort_values("start")
            run = r.iloc[-1] if len(r) else None
    logs = [r for r in (src["logs"] or []) if _log_day(r, src) == d]
    return run, (logs[-1] if logs else None)


def _day_version(d, src, run, log):
    """
    설명: 그 사이클에 만들어진 마지막 모델 버전(run 태그·로그·DB 예측 행·registry 중 가장 늦은 것)을 돌려준다.
    주의 사항: 버전은 문자열 비교로 고른다(YYYYMMDD_HHMMSS라 시간 순서와 같다). 없으면 None.
    입력 타입->반환 타입: (date, dict, pd.Series|None, dict|None) -> str|None
    """
    cands = []
    if run is not None and run["version"]:
        cands.append(run["version"])
    if log and log["train_version"]:
        cands.append(log["train_version"])
    if src["db"] is not None:
        cands += [v for v in list(src["db"]["preds"]) + list(src["db"]["registry"]) if _version_day(v, src) == d]
    return max(cands) if cands else None


def _cell_train(run, log, src):
    """
    설명: ③ 학습 칸 — MLflow run 상태, 없으면 로그의 학습 완료·예측 실패 줄.
    주의 사항: 성공 칸 글자는 run 시작 시각.
    입력 타입->반환 타입: (pd.Series|None, dict|None, dict) -> (str, str, str)
    """
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
    """
    설명: ④ 게이트 칸 — 그 버전 마지막 기준일 예측 행의 gate_passed, 없으면 MLflow gate_passed 지표, 그다음 로그.
    주의 사항: 예측 행은 전부 통과여야 통과.
    입력 타입->반환 타입: (str|None, pd.Series|None, dict|None, dict) -> (str, str, str)
    """
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
    """
    설명: 예측 저장 행 수를 상태로 바꾼다 — N_TICKERS(100) 이상 성공, 90 이상 경고, 미만 실패.
    주의 사항: 없음.
    입력 타입->반환 타입: (int) -> str
    """
    return "success" if n >= N_TICKERS else "warning" if n >= 90 else "failure"


def _cell_save(v, run, log, src):
    """
    설명: ⑤ 예측 저장 칸 — 그 버전 마지막 기준일의 예측 행 수로 판정한다.
    주의 사항: 이후 실행이 같은 키를 덮어써 버전 전체 행이 실행 기록(MLflow·로그의 저장 수)보다 적으면, 실행 기록 수로 판정하고
    "기록(덮어씀 N)"으로 쓴다(결정기록.md "⑤ 예측 저장의 덮어씀 판정", 사람 확인 대기).
    입력 타입->반환 타입: (str|None, pd.Series|None, dict|None, dict) -> (str, str, str)
    """
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
    """
    설명: ⑥ MLflow 칸 — run 유무·상태, 없으면 로그의 [MLFLOW-SKIP](경고)·로깅 실패(실패) 줄.
    주의 사항: 없음.
    입력 타입->반환 타입: (pd.Series|None, dict|None, dict) -> (str, str, str)
    """
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
    """
    설명: ⑦ registry·실측 칸 — model_registry 등록 여부(미등록은 경고)와 그 버전 예측의 실측 확정 여부(미확정은 대기).
    주의 사항: 예측 행이 이후 실행에 덮어써졌으면 "등록·덮어씀".
    입력 타입->반환 타입: (str|None, dict) -> (str, str, str)
    """
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


# 주간 튜닝 부모 run의 state 태그 → (상태, 칸 글자 앞 단어, 툴팁 설명). 대시보드는 앞 단어로 아이콘을 고른다.
TUNING_STATES = {
    "adopted": ("success", "채택", "새 값 채택 — 다음 일일 학습부터 적용"),
    "kept": ("success", "유지", "후보 기각 — 현재 적용 값 유지"),
    "reverted": ("warning", "복귀", "현재 적용 값이 기본값보다 나빠 기본값으로 되돌림 — 다음 일일 학습은 기본값"),
    "none": ("success", "기본", "후보 기각 — 채택값 없음, 기본값 유지"),
}


def _cell_tuning(d, src):
    """
    설명: ⑧ 튜닝 칸 — 실험 주간_튜닝의 그 사이클 마지막 부모 run(태그 run_kind=tuning_parent)의 status·state 태그로 판정한다.
    주의 사항: 시도별 중첩 run(tuning_trial)은 보지 않는다. 의존은 run_kind·state 태그와 run 시각·status뿐(지표 이름에 기대지 않음) —
    튜닝 코드가 이 태그 이름이나 state 값 4종을 바꾸면 TUNING_STATES도 함께 고친다. 부모 run이 없으면 기록 없음.
    입력 타입->반환 타입: (date, dict) -> (str, str, str)
    """
    if src["mlflow"] is None:
        return (*LOOKUP_FAILED, "MLflow 조회 실패")
    runs, exps = src["mlflow"]
    if EXP_TUNING not in exps:
        return (*NO_RECORD, f"MLflow 실험 '{EXP_TUNING}' 없음")
    r = runs[(runs["experiment"] == EXP_TUNING) & (runs["run_kind"] == "tuning_parent")
             & (runs["start"].map(cycle_day) == d)] if len(runs) else runs
    if not len(r):
        return (*NO_RECORD, "튜닝 부모 run 없음(시도별 중첩 run은 보지 않는다)")
    last = r.sort_values("start").iloc[-1]
    t = hm(last["start"])
    head = f"주간 튜닝 {last['start']:%m/%d} {t} 시작"
    if last["status"] == "RUNNING":
        return "running", f"진행 {t}", f"{head} · 진행 중"
    if last["status"] != "FINISHED":
        return "failure", f"실패 {t}", f"{head} · 부모 run {last['status']} — 결과 파일은 바뀌지 않았을 수 있다"
    tid = f" · {last['tuned_params_id']}" if last["tuned_params_id"] else ""
    if last["state"] not in TUNING_STATES:
        return "success", f"완료 {t}", f"{head} · 완료, state 태그 없음·알 수 없음({last['state']!r}){tid}"
    status, word, desc = TUNING_STATES[last["state"]]
    return status, f"{word} {t}", f"{head} · state={last['state']} · {desc}{tid}"


def _run_start(run, log, v):
    """그 사이클 실행의 시작 시각(MLflow run → 로그 파일 이름 → 모델 버전 순). 모르면 None."""
    if run is not None:
        return run["start"]
    if log is not None:
        return log["start"]
    return 데이터.version_kst(v) if v else None


def _run_prediction_date(src, run, v):
    """그 사이클 실행의 예측 기준일(DB 예측 행 → MLflow 태그). 모르면 None."""
    p = src["db"]["preds"].get(v) if v and src["db"] is not None else None
    if p is not None:
        return pd.Timestamp(p["prediction_date"]).date()
    if run is not None and pd.notna(run["prediction_date"]) and run["prediction_date"]:
        return pd.Timestamp(run["prediction_date"]).date()
    return None


def earlier_run_for(pdate, before, src):
    """
    설명: 예측 기준일 pdate를 시각 before보다 먼저 계산한 운영 실행이 있으면 그 시각(KST datetime), 없으면 None.
    주의 사항: MLflow 운영 run(태그 prediction_date)과 DB 예측 행(버전 시각)을 본다. 덮어써져 DB에 없는 실행은 MLflow로만 보인다.
    입력 타입->반환 타입: (date, datetime, dict) -> datetime|None
    """
    found = []
    if src["mlflow"] is not None and len(src["mlflow"][0]):
        runs = src["mlflow"][0]
        r = runs[(runs["experiment"] != EXP_TUNING) & (runs["prediction_date"] == pdate.isoformat())
                 & (runs["start"] < before)]
        found += list(r["start"])
    if src["db"] is not None:
        for ver, p in src["db"]["preds"].items():
            t = 데이터.version_kst(ver)
            if t and t < before and pd.Timestamp(p["prediction_date"]).date() == pdate:
                found.append(t)
    return min(found) if found else None


def _holiday_run_cells(d, src, run, log, v, cells):
    """
    ③~⑦ 휴장 열 덮어쓰기(칸 5개 목록을 돌려준다). 휴장이 아니면 그대로.
    실행 없음(기록 없음 칸) → skipped "휴장". 같은 기준일을 앞 실행이 이미 계산한 재실행 → 성공·대기 칸만 skipped "재실행".
    """
    why = holiday_reason(d, src["db"])
    if why is None:
        return cells
    start = _run_start(run, log, v)
    if start is None:
        return [("skipped", "휴장", f"수집 대상일 휴장({why}) — 새 거래일이 없어 학습할 이유가 없다 · {tip}")
                if (status, text) == NO_RECORD else (status, text, tip) for status, text, tip in cells]
    pdate = _run_prediction_date(src, run, v)
    prev = earlier_run_for(pdate, start, src) if pdate else None
    if prev is None:
        note = f"수집 대상일 휴장({why})이지만 " + (f"기준일 {데이터.fmt_day(pdate)}의 첫 계산(보충 실행)" if pdate
                                                  else "예측 기준일을 알 수 없음")
        return [(status, text, f"{note} · {tip}") for status, text, tip in cells]
    note = (f"휴장 재실행 — 수집 대상일 휴장({why}), 같은 기준일 {데이터.fmt_day(pdate)}을 {prev:%m/%d %H:%M} 실행이 이미 계산했고 "
            f"이 실행이 같은 키를 다시 썼다")
    return [("skipped", "재실행", f"{note} · 원래 판정 {text} · {tip}")
            if status in ("success", "waiting") and (status, text) != NO_RECORD
            else (status, text, f"{note} · {tip}") for status, text, tip in cells]


def build_grid(days, src, upcoming=None):
    """
    설명: 열 날짜(수집 대상일) 목록 × 8단계의 (상태, 글자, 설명) 격자를 {단계: [칸...]}으로 만든다.
    주의 사항: 기록 없는 칸은 waiting "기록 없음", 출처 조회 실패 칸은 skipped "조회 실패". 칸 판정은 열의 사이클 날짜로 한다.
      upcoming(아직 시작 전인 다음 날 열)의 ①~⑦ 기록 없는 칸은 waiting "예정"이다(⑧ 주간 튜닝은 그대로).
    입력 타입->반환 타입: (list[date], dict, date|None) -> dict
    """
    grid = {s: [] for s in STAGES}
    for d in map(cycle_of, days):
        run, log = _day_run(d, src)
        v = _day_version(d, src, run, log)
        cells = [_cell_collect(d, src), _cell_align(d, src), _cell_train(run, log, src), _cell_gate(v, run, log, src),
                 _cell_save(v, run, log, src), _cell_mlflow(run, log, src), _cell_registry(v, src), _cell_tuning(d, src)]
        cells[2:7] = _holiday_run_cells(d, src, run, log, v, cells[2:7])
        if d == upcoming:
            plan = f"아직 시작 전 — {데이터.fmt_day(d)} 저녁 수집 → {데이터.fmt_day(run_day(d))} 05:00 실행 예정"
            cells[:7] = [("waiting", "예정", plan) if (c[0], c[1]) == NO_RECORD else c for c in cells[:7]]  # ⑧ 튜닝은 주 1회
        for s, c in zip(STAGES, cells):
            grid[s].append(c)
    return grid


def today_cycle(now=None):
    """
    설명: 지금 시각이 속한 사이클의 열 날짜(수집 대상일)와 다음 05:00 실행 시각을 돌려준다.
    주의 사항: 12:00 전에는 어제 열(어제분 수집 → 오늘 05:00 실행), 12:00부터 오늘 열이다.
    입력 타입->반환 타입: (datetime|None) -> (date, datetime)
    """
    now = now or datetime.now(KST)
    nxt = now.replace(hour=5, minute=0, second=0, microsecond=0)
    if nxt <= now:
        nxt += timedelta(days=1)
    return column_day(now), nxt


def day_basis(column, src):
    """
    설명: 열 하나의 예측 기준일 → 대상일을 돌려준다(그 사이클 마지막 모델 버전의 마지막 기준일 예측 행, 없으면 MLflow 태그).
    주의 사항: 알 수 없으면 (None, None). 대상일은 DB 행이 있을 때만 채운다.
    입력 타입->반환 타입: (date, dict) -> (str|date|None, date|None)
    """
    c = cycle_of(column)
    run, log = _day_run(c, src)
    v = _day_version(c, src, run, log)
    p = src["db"]["preds"].get(v) if v and src["db"] is not None else None
    if p is not None:
        return p["prediction_date"], p["target_date"]
    if run is not None and run["version"] == v and pd.notna(run["prediction_date"]) and run["prediction_date"]:
        return run["prediction_date"], None
    return None, None


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

