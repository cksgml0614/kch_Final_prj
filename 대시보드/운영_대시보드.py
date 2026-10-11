# 대시보드/운영_대시보드.py — 새 대시보드(2026-10-11, 1단계: 화면 골격, 데이터 연결 없음)
# 실행(프로젝트 루트에서 — .streamlit/config.toml 테마가 적용된다):
#   streamlit run 대시보드/운영_대시보드.py --server.port 8599
# 순서: 패널 A(파이프라인 상태) → 레벨 피처 트랙 → 패널 B → 백테스트 분리 → 패널 C.
# 규칙: 자동 새로고침(st.fragment run_every 등) 금지 — Neon 컴퓨트가 계속 깨어난다. 캐시 ttl 300 + 수동 새로고침만.
# st.info 상자와 marker_color 직접 지정 금지, 모든 차트는 테마.show_chart()를 거친다. DB는 db_manager로만(2단계부터).
# ⚠️ 지금 화면의 값은 전부 하드코딩한 더미다.

import plotly.graph_objects as go
import streamlit as st

import 테마

st.set_page_config(page_title="변동성 예측 운영", layout="wide")
테마.inject_css()

head_l, head_r = st.columns([5, 1], vertical_alignment="bottom")
with head_l:
    st.title("변동성 예측 운영")
    st.caption("1단계 골격 — 모든 값은 더미입니다(DB·GitHub·MLflow 조회 없음). 시각은 KST.")
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

# ── 공통 레이아웃 확인용 더미 차트 ──
with st.container(key="card_chart"):
    st.subheader("레이아웃 확인(더미)")
    x = DAYS[-7:]
    fig = go.Figure([go.Bar(name="예측", x=x, y=[2.1, 2.4, 1.9, 2.2, 2.6, 2.0, 2.3]),
                     go.Bar(name="실측", x=x, y=[2.0, 2.9, 1.7, 2.4, 3.1, 1.8, None])])
    테마.show_chart(fig, height=320, barmode="group", yaxis_title="변동성(|로그수익률×100|)")
