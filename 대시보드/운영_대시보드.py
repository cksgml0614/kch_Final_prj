# 대시보드/운영_대시보드.py — 새 대시보드(2026-10-11 1단계 골격, 2026-10-18 2단계: 패널 A·B·C 실측 연결)
# 실행(프로젝트 루트에서 — .streamlit/config.toml 테마가 적용된다):
#   streamlit run 대시보드/운영_대시보드.py --server.port 8599
# 순서: 패널 A(파이프라인 상태) → 레벨 피처 트랙 → 패널 B → 백테스트 분리 → 패널 C.
# 규칙: 자동 새로고침(st.fragment run_every 등) 금지 — Neon 컴퓨트가 계속 깨어난다. 캐시 ttl 300 + 수동 새로고침만.
# st.info 상자와 marker_color 직접 지정 금지, 모든 차트는 테마.show_chart()를 거친다. DB는 db_manager로만(데이터.py).
# 외부 조회가 실패해도 앱은 멈추지 않고 그 자리에 "조회 실패" 칩을 그린다(원문은 콘솔에만 남는다).
# 패널 A의 출처·사이클 날짜 규칙은 파이프라인_상태.py 머리 주석 참고. 백테스트 재생 파일은 대시보드/백테스트_재생.py가 만든다.
# 화면 문구 원칙(2026-10-18 다듬기): 숫자를 잘못 읽지 않는 데 꼭 필요한 것만 한 줄로 두고, 정의·근거는 툴팁(ⓘ, title)으로 옮긴다.

import html
import re
import traceback

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import 데이터
import 테마
import 파이프라인_상태 as 상태

esc = html.escape


def lookup_failed(what):
    """
    설명: 조회가 실패한 자리에 "조회 실패" 칩과 일반 문구를 그리고, 원인은 콘솔에만 출력한다.
    주의 사항: except 블록 안에서 불러야 traceback이 찍힌다.
    입력 타입->반환 타입: (str) -> None
    """
    traceback.print_exc()
    st.html(f'{테마.status_chip("skipped", "조회 실패")} <span class="sub">{what} — 불러오지 못했습니다.</span>')


def md(d):
    """날짜를 요일 없이 '10/16'으로(좁은 자리용)."""
    return "-" if d is None or pd.isna(d) else f"{pd.Timestamp(d).month}/{pd.Timestamp(d).day}"


st.set_page_config(page_title="변동성 예측 운영", layout="wide")
테마.inject_css()

head_l, head_r = st.columns([5, 1], vertical_alignment="bottom")
with head_l:
    st.title("변동성 예측 운영", anchor=False)
    st.html('<div class="page-sub">시각 KST · 5분 캐시 · 새로고침 버튼으로만 다시 읽음</div>')
with head_r:
    if st.button("새로고침", width="stretch"):
        st.cache_data.clear()
        st.rerun()

# ── 패널 A: 파이프라인 상태 ──
GRID_DAYS = 14


def kpi_failed_html(label):
    """
    설명: 조회에 실패한 KPI 카드 내용(라벨 + "조회 실패" 칩)을 만든다.
    주의 사항: 원인은 콘솔에만 남긴다(lookup_failed와 달리 카드 높이를 맞추려고 HTML만 돌려준다).
    입력 타입->반환 타입: (str) -> str
    """
    traceback.print_exc()
    return (f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div>{테마.status_chip("skipped", "조회 실패")}</div>'
            '<div class="kpi-sub">불러오지 못했습니다.</div></div>')


def kpi_cards(src):
    """
    설명: KPI 4장(마지막 정상 실행, 배포 게이트, 데이터 최신일, 최신 예측)의 HTML을 만든다.
    주의 사항: 카드마다 따로 실패를 처리한다. 마지막 정상 실행·게이트는 MLflow 일일_자동화의 마지막 FINISHED run.
      보조 글자는 한 줄로 줄이고 전문(버전·전체 수치)은 마우스 올림(title)에 둔다.
    입력 타입->반환 타입: (dict) -> list[str]
    """
    cards = []
    try:
        run = 상태.last_finished_daily_run(src)
        if run is None:
            cards.append(테마.kpi_html("마지막 정상 실행", "기록 없음", "완료된 운영 run 없음",
                                     f"MLflow {상태.EXP_DAILY}에 FINISHED run 없음"))
            cards.append(테마.kpi_html("배포 게이트", "기록 없음", ""))
        else:
            n = "-" if pd.isna(run["n_success"]) else int(run["n_success"])
            cards.append(테마.kpi_html(
                "마지막 정상 실행", f"{데이터.fmt_day(run['start'])} {상태.hm(run['start'])}",
                f"예측 {n}/{상태.N_TICKERS} · 기준 {md(run['prediction_date'])}",
                f"버전 {run['version']} · 예측 {n}/{상태.N_TICKERS} · 기준일 {run['prediction_date']}"))
            ok = run["gate_passed"] is not None and run["gate_passed"] >= 1
            best = min(run["garch"], run["sma"], run["parkinson"])
            cards.append(테마.kpi_html(
                "배포 게이트", "통과" if ok else "미통과",
                f"{run['hybrid']:.4f} vs 최저 {best:.4f}",
                f"hybrid {run['hybrid']:.4f} vs GARCH {run['garch']:.4f} / SMA20 {run['sma']:.4f} / "
                f"Parkinson {run['parkinson']:.4f} (val 종목평균 RMSE, 버전 {run['version']})",
                status="success" if ok else "failure"))
    except Exception:
        cards += [kpi_failed_html("마지막 정상 실행"), kpi_failed_html("배포 게이트")]
    try:
        if src["db"] is None:
            raise RuntimeError("DB 조회 실패")
        lt = src["db"]["latest"]
        stock, kospi, kosdaq = lt.get("stock"), lt.get("KOSPI"), lt.get("KOSDAQ")
        full = (f"주가 {데이터.fmt_day(stock)} · KOSPI {데이터.fmt_day(kospi)} · KOSDAQ {데이터.fmt_day(kosdaq)}")
        sub = ("주가·KOSPI·KOSDAQ 같은 날" if stock == kospi == kosdaq
               else f"주가 {md(stock)} · KOSPI {md(kospi)} · KOSDAQ {md(kosdaq)}")
        cards.append(테마.kpi_html("데이터 최신일", 데이터.fmt_day(stock), sub, full))
    except Exception:
        cards.append(kpi_failed_html("데이터 최신일"))
    try:
        df = 데이터.latest_predictions()
        if df.empty:
            cards.append(테마.kpi_html("최신 예측", "기록 없음", ""))
        else:
            p, t = df["prediction_date"].iloc[0], df["target_date"].iloc[0]
            cards.append(테마.kpi_html("최신 예측", f"{len(df)}종목", f"기준 {md(p)} → 평가 {md(t)}",
                                     f"예측 기준일 {데이터.fmt_day(p)} → 평가일 {데이터.fmt_day(t)}"))
    except Exception:
        cards.append(kpi_failed_html("최신 예측"))
    return cards


HHMM = re.compile(r"\d{2}:\d{2}")
OVERWRITTEN = re.compile(r"^(\d+)\(덮어씀 (\d+)\)$")
SHORT_WORDS = {"등록·실측": "실측", "등록·덮어씀": "덮어씀", "실측 대기": "대기", "미등록": "미등록"}


def grid_short(stage_no, status, text):
    """
    설명: 격자 칸의 긴 글자(파이프라인_상태가 만든 것)를 5글자 이내 표시로 줄인다. 전문은 툴팁에 남는다.
    주의 사항: 표시만 바꾼다(상태·근거는 그대로). 돌려주는 값 = (보이는 글자, 덮어씀 여부, 덮어씀 툴팁 문구).
      성공 = 시각(있는 단계만), 경고 = 아이콘만, 실패 = 짧은 값, ⑤ = 행 수, ⑦ = 실측·대기 등 한 단어.
    입력 타입->반환 타입: (int, str, str) -> (str, bool, str)
    """
    if stage_no == 5:
        if (m := OVERWRITTEN.match(text)):
            rec, over = int(m.group(1)), int(m.group(2))
            return str(rec), True, f"이후 재실행으로 덮어씀, 저장 {rec} / DB 남은 {rec - over}"
        if (m := re.match(r"^(\d+)행$", text)):
            return (m.group(1) if status == "success" else text), False, ""
    if stage_no == 7 and text in SHORT_WORDS:
        return SHORT_WORDS[text], False, ""
    if (m := HHMM.search(text)):
        return m.group(0), False, ""
    if status == "failure":
        return ("미통과" if "미통과" in text else "실패"), False, ""
    if status == "success" and text.endswith("(로그)"):
        return "로그", False, ""
    return "", False, ""


def grid_cell(stage_no, stage, day, cell):
    """
    설명: 격자 칸 하나의 <td>를 만든다(색 + 아이콘 + 5글자 이내, 마우스 올림에 전문과 근거).
    주의 사항: 상태는 테마.STATUS 6종만 쓴다. 기록 없음은 글자 없이 흐린 점 하나.
    입력 타입->반환 타입: (int, str, date, tuple) -> str
    """
    status, text, tip = cell
    head = f"{stage} · {데이터.fmt_day(day)}"
    if (status, text) == 상태.NO_RECORD:
        return (f'<td class="none" data-status="none" title="{esc(f"{head} · 기록 없음 — {tip}")}">'
                '<span class="nodot"></span></td>')
    s = 테마.STATUS[status]
    short, over, over_tip = grid_short(stage_no, status, text)
    title = f"{head} · {s['label']} · {text}" + (f" ({over_tip})" if over else "") + f" — {tip}"
    cls = " ".join(c for c in ({"skipped": "status-skipped", "failure": "status-failure"}.get(status, ""),
                               "over" if over else "") if c)
    label = f"<span>{esc(short)}</span>" if short else ""
    return (f'<td class="{cls}" data-status="{status}" style="background:{s["bg"]};color:{s["fg"]}" '
            f'title="{esc(title)}"><span class="cell">{테마.icon(status)}{label}</span></td>')


def grid_header(day, today):
    """열 머리: 날짜와 요일 두 줄. 주말은 흐리게, 오늘 사이클 열은 위 선 + 옅은 배경."""
    cls = " ".join(c for c in ("weekend" if day.weekday() >= 5 else "", "today" if day == today else "") if c)
    return (f'<th class="{cls}" title="사이클 {데이터.fmt_day(day)}">{day.month}/{day.day}'
            f'<span class="wd">{데이터.WEEKDAYS[day.weekday()]}</span></th>')


def grid_legend():
    """범례 한 줄: 상태 6종 + 기록 없음 점 + 덮어씀 표시."""
    items = [f'<span style="color:{s["fg"]}">{테마.icon(k)}<span style="color:{테마.TEXT_MUTED}">{s["label"]}</span></span>'
             for k, s in 테마.STATUS.items()]
    items.append('<span><span class="swatch"><span class="nodot"></span></span>기록 없음</span>')
    items.append('<span><span class="swatch over"></span>이후 재실행이 덮어씀</span>')
    return '<div class="legend">' + "".join(items) + "</div>"


CYCLE_TIP = ("사이클 날짜 = 시각에서 12시간을 뺀 KST 날짜(저녁 수집과 다음 날 05:00 예측이 한 사이클). "
             "출처: GitHub Actions(최근 run 20개)·로컬 로그·MLflow·운영 DB.")
GRID_TIP = ("칸에 마우스를 올리면 전문과 근거가 보인다. 기록 없음 = 출처에 그 날짜 기록이 없음, 조회 실패 = 출처를 읽지 못함. "
            "① 수집: GitHub Actions·로컬 수집 / ② 가격 정렬 점검 / ③⑥ MLflow run(없으면 로그) / ④ 게이트 / "
            "⑤ 그 버전 마지막 기준일의 예측 행 수(100 성공·90 이상 경고·미만 실패, 이후 재실행이 덮어썼으면 실행 기록의 저장 수) / "
            "⑦ 모델 등록·실측 / ⑧ 주간 튜닝. 시각은 그 단계가 끝난 시각.")


def render_panel_a():
    """
    설명: KPI 4장, 오늘 사이클 표시줄, 최근 14일 × 8단계 격자를 출처에서 복원해 그린다.
    주의 사항: 출처(GitHub·MLflow·로그·DB) 하나가 실패해도 그 칸만 "조회 실패"가 된다.
    입력 타입->반환 타입: () -> None
    """
    today, next_run = 상태.today_cycle()
    days = 상태.days_back(today, GRID_DAYS)
    src = 상태.load_sources(days[0])
    failed = [k for k, v in src.items() if v is None]

    for i, (col, card) in enumerate(zip(st.columns(4), kpi_cards(src))):
        with col, st.container(key=f"card_kpi_{i}"):
            st.html(card)

    grid = 상태.build_grid(days, src)
    with st.container(key="card_stepper"):
        st.subheader(f"오늘 사이클 — {데이터.fmt_day(today)}", anchor=False, help=CYCLE_TIP)
        parts = []
        for i, stage in enumerate(상태.STAGES[:7]):
            status, text, tip = grid[stage][-1]
            arrow = '<span class="arrow">→</span>' if i else ""
            parts.append(f"<span>{arrow}{stage[2:]} {테마.status_chip(status, text, tip)}</span>")
        parts.append(f'<span><span class="arrow">→</span><span class="sub">다음 실행 {데이터.fmt_day(next_run)} 05:00</span></span>')
        st.html('<div class="stepper">' + "".join(parts) + "</div>")
        if failed:
            st.html(f'<div class="sub">{테마.status_chip("skipped", "조회 실패")} {esc(", ".join(failed))}</div>')

    with st.container(key="card_grid"):
        st.subheader(f"최근 {GRID_DAYS}일 × 단계", anchor=False, help=GRID_TIP)
        cols = '<colgroup><col class="stage">' + "<col>" * len(days) + "</colgroup>"
        head = "".join(grid_header(d, today) for d in days)
        rows = [f'<tr><th class="stage" scope="row">{stage}</th>'
                f'{"".join(grid_cell(no, stage, d, c) for d, c in zip(days, grid[stage]))}</tr>'
                for no, stage in enumerate(상태.STAGES, start=1)]
        st.html(f'<div class="gridwrap"><table class="grid14">{cols}<thead><tr><th class="stage"></th>{head}</tr></thead>'
                f'<tbody>{"".join(rows)}</tbody></table></div>{grid_legend()}')


render_panel_a()


# ── 패널 B: 오늘의 예측 ──
def render_panel_b():
    """
    설명: 가장 최근 prediction_date의 예측(상위 10 막대 + 전 종목 표)을 그린다.
    주의 사항: 게이트 미통과 또는 입력 의심이 하나라도 있으면 숨기지 않고 "참고용" 칩을 붙인다.
    입력 타입->반환 타입: () -> None
    """
    st.subheader("오늘의 예측", anchor=False)
    try:
        df = 데이터.latest_predictions()
    except Exception:
        lookup_failed("최신 예측")
        return
    if df.empty:
        st.html(테마.status_chip("waiting", "기록 없음") + ' <span class="sub">예측 행이 없습니다.</span>')
        return
    pdate, tdate = df["prediction_date"].iloc[0], df["target_date"].iloc[0]
    trusted = bool(df["gate_passed"].all() and not df["input_data_suspect"].any())
    chip = (테마.status_chip("success", "신뢰 조건 충족", "게이트 통과 · 입력 정상") if trusted
            else 테마.status_chip("warning", "참고용(신뢰 조건 미충족)", "게이트 미통과 또는 입력 의심 행이 있다"))
    versions = ", ".join(sorted(df["model_version"].unique()))
    tip = (f"모델 {versions} · 평소 대비 = 예측값 ÷ 최근 {데이터.USUAL_WINDOW}거래일 실현 변동성 평균 · "
           "평가일이 휴장이면 실측은 다음 거래일로 잡힌다")
    st.html(f'<div class="body"><b>예측 기준일 {데이터.fmt_day(pdate)} → 평가일 {데이터.fmt_day(tdate)}</b> {chip}</div>'
            f'<div class="sub num">{len(df)}종목 · 게이트 통과 {int(df["gate_passed"].sum())}/{len(df)} · '
            f'입력 의심 {int(df["input_data_suspect"].sum())}/{len(df)} {테마.info(tip)}</div>')

    t_l, t_r = st.columns([3, 2], vertical_alignment="center")
    with t_r, st.container(horizontal_alignment="right"):
        mode = st.segmented_control("막대 값", ["절대값", "평소 대비"], default="절대값", key="b_mode",
                                    label_visibility="collapsed") or "절대값"
    ratio_mode = mode == "평소 대비"
    with t_l:
        st.html(f'<div class="h4">상위 10종목 <span class="sub">{"평소 대비 배율 순" if ratio_mode else "예측 변동성 순"}</span> '
                + 테마.info("막대 끝 글자: 막대 값 · 다른 기준 값. 평소 대비 = 예측 ÷ 최근 "
                          f"{데이터.USUAL_WINDOW}거래일 실현 변동성 평균 — 평소 값(분모)이 작으면 배율이 커 보이므로 "
                          "배율 옆에 예측 ÷ 평소를 함께 적는다. 점선 = 평소 수준(1배). '입력 의심' = 오염 의심 입력으로 만든 예측.")
                + "</div>")
    # 순위와 막대 값이 같은 기준을 따른다. 평소 대비 모드는 배율 순(배율을 못 구한 종목은 뺀다), 표시 순위도 배율 순위.
    if ratio_mode:
        top = df.dropna(subset=["ratio"]).sort_values("ratio", ascending=False, kind="stable").head(10)
        x = top["ratio"]
        text = [f"{r.ratio:.2f}배 · {r.predicted_volatility:.2f} ÷ {r.usual_vol:.2f}" for r in top.itertuples()]
        title = "평소 대비 배율(배)"
    else:
        top = df.head(10)
        x = top["predicted_volatility"]
        text = [f"{r.predicted_volatility:.2f} · " + ("-" if pd.isna(r.ratio) else f"{r.ratio:.2f}배")
                for r in top.itertuples()]
        title = "예측 변동성(|로그수익률×100|)"
    labels = [f"{i}. {r.name or r.ticker} ({r.ticker})" + (" · 입력 의심" if r.input_data_suspect else "")
              for i, r in enumerate(top.itertuples(), start=1)]
    custom = list(zip(top["predicted_volatility"], top["usual_vol"], top["ratio"], top["n_days"], top["rank"]))
    fig = go.Figure(go.Bar(x=x, y=labels, orientation="h", text=text, textposition="outside", cliponaxis=False,
                           customdata=custom,
                           hovertemplate="%{y}<br>예측 %{customdata[0]:.3f} · 평소 %{customdata[1]:.3f}"
                                         "(%{customdata[3]}거래일 평균) · 배율 %{customdata[2]:.2f}배"
                                         "<br>예측 변동성 순위 %{customdata[4]}위<extra></extra>"))
    if ratio_mode:
        fig.add_vline(x=1, line=dict(dash="dot", color=테마.TEXT_MUTED, width=1))
    테마.show_chart(fig, height=384, yaxis=dict(autorange="reversed"), xaxis_title=title, margin=dict(l=8, r=136, t=8, b=8))

    table = pd.DataFrame({
        "순위": df["rank"], "코드": df["ticker"], "종목": df["name"].fillna("-"),
        "예측 변동성": df["predicted_volatility"], "평소 대비 배율": df["ratio"],
        "GARCH 예측": df["garch_baseline"], "입력 의심": df["input_data_suspect"].map({True: "의심", False: "-"}),
    })
    st.html(f'<div class="h4">전 종목 <span class="sub num">{len(df)}개</span> '
            f'{테마.info("열 머리를 누르면 정렬된다")}</div>')
    st.dataframe(table, hide_index=True, width="stretch", height=420, column_config={
        "예측 변동성": st.column_config.NumberColumn(format="%.3f"),
        "평소 대비 배율": st.column_config.NumberColumn(format="%.2f배"),
        "GARCH 예측": st.column_config.NumberColumn(format="%.3f"),
    })


with st.container(key="card_panel_b"):
    render_panel_b()


# ── 패널 C: 성능 ──
def day_chips(g):
    """
    설명: 운영 기록 평가일 하나(g = 그날 행들)의 상태 칩 HTML과 일별 RMSE 사용 여부를 돌려준다.
    주의 사항: 행이 MIN_ROWS_PER_DAY 미만이면 "불완전"(일별 RMSE 제외), 버전이 둘 이상이면 "버전 혼재".
    입력 타입->반환 타입: (pd.DataFrame) -> (str, bool)
    """
    chips, n = [], len(g)
    usable = n >= 데이터.MIN_ROWS_PER_DAY
    if not usable:
        chips.append(테마.status_chip("warning", f"불완전 {n}행",
                                    f"{n}행({데이터.MIN_ROWS_PER_DAY} 미만) — 일별 RMSE에서 제외"))
    nv = g["model_version"].nunique()
    if nv > 1:
        chips.append(테마.status_chip("warning", f"버전 혼재 {nv}개", ", ".join(sorted(g["model_version"].unique()))))
    n_pre = int((g["group"] == 데이터.GROUP_PRE).sum())
    pre_tip = "사전 예측 = 모델 버전 시각(KST)이 평가일 08:30보다 이른 행"
    if n_pre == n:
        chips.append(테마.status_chip("success", "사전 예측", pre_tip))
    elif n_pre == 0:
        chips.append(테마.status_chip("warning", "사후 계산", pre_tip))
    else:
        chips.append(테마.status_chip("warning", f"사전 {n_pre} · 사후 {n - n_pre}", pre_tip))
    n_gate = int(g["gate_passed"].sum())
    chips.append(테마.status_chip("success", f"게이트 통과 {n_gate}") if n_gate == n
                 else 테마.status_chip("failure", f"게이트 미통과 {n - n_gate}"))
    n_sus = int(g["input_data_suspect"].sum())
    chips.append(테마.status_chip("success", "입력 정상") if n_sus == 0 else 테마.status_chip("warning", f"입력 의심 {n_sus}"))
    return '<div class="chips">' + "".join(chips) + "</div>", usable


def short_range(a, b):
    """'2026-09-22', '2026-10-16' → '2026-09-22~10-16'(같은 해면 뒤쪽 연도 생략)."""
    a, b = str(a), str(b)
    return f"{a}~{b[5:]}" if a[:4] == b[:4] else f"{a}~{b}"


def summary_table(rows, colors):
    """
    설명: 그룹별 요약 표 HTML. 행마다 네 모델 중 가장 낮은 RMSE를 굵게, 숫자는 오른쪽 정렬·소수 4자리.
    주의 사항: 그룹 점 색은 산점도 계열 색과 같다(colors[i]). NaN은 '-'.
    입력 타입->반환 타입: (list[dict], list[str]) -> str
    """
    models = list(데이터.PREDICTION_COLS.values())
    head = ("<tr><th>그룹</th><th class='r'>평가일</th><th class='r'>행</th><th>기간</th>"
            + "".join(f"<th class='r'>{m}</th>" for m in models) + "</tr>")
    body = []
    for r, color in zip(rows, colors):
        vals = [r[m] for m in models]
        finite = [v for v in vals if pd.notna(v)]
        best = min(finite) if finite else None
        cells = "".join(f"<td class='r{' best' if best is not None and v == best else ''}'>"
                        f"{'-' if pd.isna(v) else f'{v:.4f}'}</td>" for v in vals)
        body.append(f"<tr><td><span class='gdot' style='background:{color}'></span>{esc(r['그룹'])}</td>"
                    f"<td class='r'>{r['평가일 수']}</td><td class='r'>{r['행 수']}</td><td>{r['기간']}</td>{cells}</tr>")
    return f'<div class="dwrap"><table class="dtable">{head}{"".join(body)}</table></div>'


PANEL_C_TIP = ("RMSE는 배포 게이트와 같은 종목평균(종목별 RMSE의 평균). 하루치는 종목당 1행이라 평균 절대 오차와 같다. "
               "운영 기록만 집계하고 시험 실행·과거 백테스트 실행은 뺐다. "
               "사전 예측 = 모델 버전 시각(KST)이 평가일 08:30보다 이른 행, 나머지는 사후 계산.")
OPS_TIP = ("평가일 = 예측 기준일 뒤 첫 거래일. 90행 미만인 날은 '불완전'으로 일별 RMSE에서 뺀다. "
           "칩: 사전/사후, 게이트, 입력 상태(신뢰 조건 = 게이트 통과 · 입력 정상).")


def render_panel_c():
    """
    설명: 운영 기록(사전·사후)과 백테스트 재생(최근 15거래일)의 예측 대 실현 성능을 그룹별로 나눠 그린다.
    주의 사항: 실측 대기 행은 계산에서 뺀다. 재생 파일이 없으면 그 구역만 안내 문구로 바꾼다.
    입력 타입->반환 타입: () -> None
    """
    st.subheader("성능 — 예측 대 실현", anchor=False, help=PANEL_C_TIP)
    st.html('<div class="sub">종목평균 RMSE · 낮을수록 좋음</div>')
    try:
        ops = 데이터.operational_records()
    except Exception:
        lookup_failed("운영 기록")
        ops = None
    try:
        rep, meta = 데이터.replay_records()
    except Exception:
        lookup_failed("백테스트 재생 파일")
        rep, meta = None, None

    # 재생 구역: 운영 기록이 아니라는 칩 + 한 줄
    rep_conf = None
    if rep is None:
        st.html(테마.status_chip("waiting", "재생 파일 없음")
                + ' <span class="sub">대시보드\\백테스트_재생.py를 실행하면 생긴다</span>')
    else:
        rep_conf = rep[rep["group"] == 데이터.GROUP_REPLAY]
        ep = meta.get("eval_period") or ["-", "-"]
        tip = (f"현재 모델(버전 {meta['checkpoint_version']})로 과거 날짜를 다시 계산한 값이며 운영 중 기록이 아니다. "
               f"학습·검증 구간(~{meta['split_val_end']}) 밖의 날짜만 사용. 생성 {meta['generated_at_kst']} KST, "
               f"대상 기간 {ep[0]} ~ {ep[1]}, 데이터 마지막 확정일 {meta['data_last_confirmed_date']}, "
               f"코드 {str(meta['code_commit'])[:12]}")
        gen = str(meta["generated_at_kst"])[5:16]
        st.html(f'<div class="body" title="{esc(tip)}"><span class="tag">백테스트 재생 · 운영 기록 아님</span> '
                f'<span class="sub num">모델 {meta["checkpoint_version"]} · {short_range(ep[0], ep[1])} '
                f'({meta["n_eval_days"]}거래일) · 학습·검증 구간 밖 · 생성 {gen}</span></div>')

    # 그룹별 요약 표
    groups = []
    if ops is not None:
        groups += [(name, ops[ops["group"] == name]) for name in (데이터.GROUP_PRE, 데이터.GROUP_POST)]
    if rep_conf is not None:
        groups.append((데이터.GROUP_REPLAY, rep_conf))
    plotted = [name for name, g in groups if g["realized_volatility"].notna().any()]
    color_of = {name: 테마.DATA_COLORS[i % len(테마.DATA_COLORS)] for i, name in enumerate(plotted)}
    summary = [{"그룹": name, "평가일 수": g["realized_date"].nunique(), "행 수": len(g),
                "기간": f"{g['realized_date'].min()} ~ {g['realized_date'].max()}" if len(g) else "-",
                **데이터.rmse_row(g)} for name, g in groups]
    if summary:
        st.html(summary_table(summary, [color_of.get(r["그룹"], 테마.BORDER) for r in summary]))

    # 일별 RMSE: 재생 4계열 선 + 운영 하이브리드 점(90행 이상인 평가일만)
    fig = go.Figure()
    if rep_conf is not None and len(rep_conf):
        for col, label in 데이터.PREDICTION_COLS.items():
            days = sorted(rep_conf["realized_date"].unique())
            vals = [데이터.tickermean_rmse(rep_conf[rep_conf["realized_date"] == d], col) for d in days]
            fig.add_trace(go.Scatter(x=days, y=vals, mode="lines+markers", name=f"재생 {label}"))
    op_days = []
    if ops is not None:
        conf = ops[ops["realized_status"] == "confirmed"]
        for d, g in conf.groupby("realized_date"):
            chips, usable = day_chips(g)
            op_days.append((d, g, chips, usable))
        pts = [(d, 데이터.tickermean_rmse(g, "predicted_volatility"), bool((g["group"] == 데이터.GROUP_PRE).all()))
               for d, g, _, usable in op_days if usable]
        if pts:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers", name="운영 하이브리드 (● 사전 · ✕ 사후·혼재)",
                marker=dict(size=11, symbol=["circle" if p[2] else "x" for p in pts]),
                hovertemplate="%{x}<br>RMSE %{y:.4f}<extra></extra>"))
    st.html('<div class="h4">일별 종목평균 RMSE</div>')
    테마.show_chart(fig, height=360, yaxis_title="RMSE", xaxis_title="평가일")

    # 산점도: 그룹별 예측(하이브리드) 대 실현
    sc, hi = go.Figure(), 0.0
    for name, g in groups:
        g = g.dropna(subset=["realized_volatility"])
        if len(g):
            sc.add_trace(go.Scattergl(x=g["realized_volatility"], y=g["predicted_volatility"], mode="markers",
                                      name=f"{name} ({len(g)}행)", opacity=0.5, marker=dict(size=5),
                                      hovertemplate="실현 %{x:.2f}<br>예측 %{y:.2f}<extra></extra>"))
            hi = max(hi, float(g["realized_volatility"].max()), float(g["predicted_volatility"].max()))
    sc.add_shape(type="line", x0=0, y0=0, x1=hi, y1=hi, line=dict(dash="dot", color=테마.TEXT_MUTED, width=1))
    st.html('<div class="h4">예측 대 실현 <span class="sub">하이브리드 · 점선 = 예측과 실현이 같음</span></div>')
    테마.show_chart(sc, height=416, xaxis_title="실현 변동성", yaxis_title="예측 변동성")

    # 운영 기록 표(평가일별)
    if ops is None:
        return
    models = list(데이터.PREDICTION_COLS.values())
    head = ("<tr><th>평가일</th><th>예측 기준일</th><th class='r'>행</th><th>상태</th>"
            + "".join(f"<th class='r'>{m}</th>" for m in models) + "</tr>")
    body = []
    for t, g in ops[ops["realized_status"] != "confirmed"].groupby("target_date"):
        body.append(f'<tr><td>{데이터.fmt_day(t)}</td><td>{데이터.fmt_day(g["prediction_date"].iloc[0])}</td>'
                    f'<td class="r">{len(g)}</td><td>{테마.status_chip("waiting", "실측 대기")}</td>'
                    + "<td class='r'>-</td>" * len(models) + "</tr>")
    for d, g, chips, usable in sorted(op_days, key=lambda x: x[0], reverse=True):
        vals = 데이터.rmse_row(g) if usable else {}
        cells = "".join(f"<td class='r'>{vals[k]:.4f}</td>" if usable else "<td class='r'>-</td>" for k in models)
        pds = ", ".join(데이터.fmt_day(x) for x in sorted(g["prediction_date"].unique()))
        body.append(f'<tr><td>{데이터.fmt_day(d)}</td><td>{pds}</td><td class="r">{len(g)}</td>'
                    f'<td>{chips}</td>{cells}</tr>')
    dates = ops[ops["realized_status"] == "confirmed"]["realized_date"]
    st.html(f'<div class="h4">평가일별 운영 기록 <span class="sub num">평가일 {dates.nunique()}개 · '
            f'{dates.min()} ~ {dates.max()}</span> {테마.info(OPS_TIP)}</div>')
    st.html(f'<div class="dwrap"><table class="dtable compact">{head}{"".join(body)}</table></div>')


with st.container(key="card_panel_c"):
    render_panel_c()
