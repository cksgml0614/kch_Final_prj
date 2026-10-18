# 대시보드/운영_대시보드.py — 새 대시보드(2026-10-11 1단계 골격, 2026-10-18 2단계: 패널 B 실측 연결)
# 실행(프로젝트 루트에서 — .streamlit/config.toml 테마가 적용된다):
#   streamlit run 대시보드/운영_대시보드.py --server.port 8599
# 순서: 패널 A(파이프라인 상태) → 레벨 피처 트랙 → 패널 B → 백테스트 분리 → 패널 C.
# 규칙: 자동 새로고침(st.fragment run_every 등) 금지 — Neon 컴퓨트가 계속 깨어난다. 캐시 ttl 300 + 수동 새로고침만.
# st.info 상자와 marker_color 직접 지정 금지, 모든 차트는 테마.show_chart()를 거친다. DB는 db_manager로만(데이터.py).
# 외부 조회가 실패해도 앱은 멈추지 않고 그 자리에 "조회 실패" 칩을 그린다(원문은 콘솔에만 남는다).
# ⚠️ 패널 A(KPI·단계 표시줄·격자)의 값은 아직 하드코딩한 더미다.

import traceback

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import 데이터
import 테마


def lookup_failed(what):
    """
    설명: 조회가 실패한 자리에 "조회 실패" 칩과 일반 문구를 그리고, 원인은 콘솔에만 출력한다.
    주의 사항: except 블록 안에서 불러야 traceback이 찍힌다.
    입력 타입->반환 타입: (str) -> None
    """
    traceback.print_exc()
    st.html(f'{테마.status_chip("skipped", "조회 실패")} <span>{what}을(를) 불러오지 못했습니다.</span>')

st.set_page_config(page_title="변동성 예측 운영", layout="wide")
테마.inject_css()

head_l, head_r = st.columns([5, 1], vertical_alignment="bottom")
with head_l:
    st.title("변동성 예측 운영")
    st.caption("시각은 KST. 데이터는 5분 동안 캐시되며, 새로고침 버튼으로만 다시 읽습니다.")
with head_r:
    if st.button("새로고침", width="stretch"):
        st.cache_data.clear()
        st.rerun()

# ── KPI 카드 4개 ──
KPIS = [
    ("마지막 정상 실행", "10/11(일) 12:54", "버전 20261011_035645 · 예측 100/100 (더미)"),
    ("배포 게이트", "✓ 통과", "hybrid 1.9700 vs GARCH 2.1486 / SMA20 2.0252 / Parkinson 2.0090 (더미)"),
    ("데이터 최신일", "10/08(목)", "주가·KOSPI 같은 날 · 오늘·미래 휴장 여부는 다음 수집 후 확정 (더미)"),
    ("최신 예측", "100종목", "예측 기준일 10/08(목) → 평가일 10/12(월) 예정 (더미)"),
]
for i, (col, (label, value, sub)) in enumerate(zip(st.columns(4), KPIS)):
    with col, st.container(key=f"card_kpi_{i}"):
        st.html(테마.kpi_html(label, value, sub))

# ── 단계 표시줄 ──
STEPS = [("수집", "success", "완료 10/10 19:58"), ("데이터 반영", "success", "100/100"),
         ("학습", "success", "완료 12:56"), ("게이트", "success", "통과"),
         ("예측 저장", "running", "진행 64/100"), ("MLflow", "waiting", "대기"),
         ("registry·실측", "skipped", "평가일 전")]
with st.container(key="card_stepper"):
    st.subheader("오늘 사이클")
    parts = []
    for i, (name, status, text) in enumerate(STEPS):
        if i:
            parts.append('<span class="arrow">→</span>')
        parts.append(f"<span><b>{name}</b> {테마.status_chip(status, text)}</span>")
    parts.append('<span class="arrow">→ 다음 실행 10/12(월) 05:00</span>')
    st.html('<div class="stepper">' + "".join(parts) + "</div>")

# ── 최근 14일 × 7단계 격자 (6가지 상태가 모두 보이도록 하드코딩) ──
DAYS = ["9/28(월)", "9/29(화)", "9/30(수)", "10/1(목)", "10/2(금)", "10/3(토)", "10/4(일)",
        "10/5(월)", "10/6(화)", "10/7(수)", "10/8(목)", "10/9(금)", "10/10(토)", "10/11(일)"]
STAGES = ["① 수집", "② 데이터 반영", "③ 학습", "④ 게이트", "⑤ 예측 저장", "⑥ MLflow", "⑦ registry·실측"]
PATTERN = "SSSWSKSSFSSSRZ"   # S 성공, W 경고, F 실패, R 진행, Z 대기, K 건너뜀 — 더미
CODE = {"S": "success", "W": "warning", "F": "failure", "R": "running", "Z": "waiting", "K": "skipped"}
with st.container(key="card_grid"):
    st.subheader("최근 14일 × 단계")
    head = "".join(f"<th>{d}</th>" for d in DAYS)
    rows = []
    for si, stage in enumerate(STAGES):
        cells = []
        for di in range(len(DAYS)):
            status = CODE[PATTERN[(di + si) % len(PATTERN)]]
            s = 테마.STATUS[status]
            cls = {"skipped": ' class="status-skipped"', "failure": ' class="status-failure"'}.get(status, "")
            cells.append(f'<td{cls} style="background:{s["bg"]};color:{s["fg"]}" '
                         f'title="{stage} · {DAYS[di]} · {s["label"]} (더미)">{s["icon"]} {s["label"]}</td>')
        rows.append(f'<tr><td class="stage">{stage}</td>{"".join(cells)}</tr>')
    st.html(f'<table class="grid14"><tr><th></th>{head}</tr>{"".join(rows)}</table>')
    legend = " ".join(테마.status_chip(k) for k in 테마.STATUS)
    st.html(f'<div style="margin-top:8px">{legend}</div>')

# ── 패널 B: 오늘의 예측 ──
def render_panel_b():
    """
    설명: 가장 최근 prediction_date의 예측(상위 10 막대 + 전 종목 표)을 그린다.
    주의 사항: 게이트 미통과 또는 입력 의심이 하나라도 있으면 숨기지 않고 "참고용" 칩을 붙인다.
    입력 타입->반환 타입: () -> None
    """
    st.subheader("오늘의 예측")
    try:
        df = 데이터.latest_predictions()
    except Exception:
        lookup_failed("최신 예측")
        return
    if df.empty:
        st.html(테마.status_chip("waiting", "기록 없음") + " model_predictions에 예측 행이 없습니다.")
        return
    pdate, tdate = df["prediction_date"].iloc[0], df["target_date"].iloc[0]
    trusted = bool(df["gate_passed"].all() and not df["input_data_suspect"].any())
    chip = (테마.status_chip("success", "신뢰 조건 충족(게이트 통과 · 입력 정상)") if trusted
            else 테마.status_chip("warning", "참고용(신뢰 조건 미충족)"))
    versions = ", ".join(sorted(df["model_version"].unique()))
    st.html(f"<div><b>예측 기준일 {데이터.fmt_day(pdate)} → 평가일 {데이터.fmt_day(tdate)} (KST)</b> {chip}</div>"
            f'<div style="color:{테마.TEXT_MUTED};font-size:.85rem;margin-top:4px">'
            f"{len(df)}종목 · 모델 {versions} · 게이트 통과 {int(df['gate_passed'].sum())}/{len(df)} · "
            f"입력 의심 {int(df['input_data_suspect'].sum())}/{len(df)} · "
            f"평소 대비 = 예측값 ÷ 최근 {데이터.USUAL_WINDOW}거래일 실현 변동성 평균 · "
            "평가일이 휴장이면 실측은 다음 거래일로 잡힌다</div>")

    mode = st.segmented_control("막대 값", ["절대값", "평소 대비"], default="절대값", key="b_mode") or "절대값"
    top = df.head(10)
    col, title, fmt = (("ratio", "평소 대비 배율(배)", ".2f") if mode == "평소 대비"
                       else ("predicted_volatility", "예측 변동성(|로그수익률×100|)", ".2f"))
    labels = [f"{r.rank}. {r.name or r.ticker} ({r.ticker})" for r in top.itertuples()]
    fig = go.Figure(go.Bar(x=top[col], y=labels, orientation="h", texttemplate=f"%{{x:{fmt}}}",
                           textposition="outside", hovertemplate="%{y}<br>%{x:.3f}<extra></extra>"))
    st.caption(f"상위 10종목(예측 변동성 순) · 막대 값: {mode}")
    테마.show_chart(fig, height=400, yaxis=dict(autorange="reversed"), xaxis_title=title, margin=dict(l=8, r=40, t=12, b=8))

    table = pd.DataFrame({
        "순위": df["rank"], "코드": df["ticker"], "종목": df["name"].fillna("-"),
        "예측 변동성": df["predicted_volatility"], "평소 대비 배율": df["ratio"],
        "GARCH 예측": df["garch_baseline"], "입력 의심": df["input_data_suspect"].map({True: "의심", False: "-"}),
    })
    st.caption(f"전 종목 {len(df)}개 — 열 머리를 누르면 정렬된다")
    st.dataframe(table, hide_index=True, width="stretch", height=420, column_config={
        "예측 변동성": st.column_config.NumberColumn(format="%.3f"),
        "평소 대비 배율": st.column_config.NumberColumn(format="%.2f배"),
        "GARCH 예측": st.column_config.NumberColumn(format="%.3f"),
    })


with st.container(key="card_panel_b"):
    render_panel_b()
