# 대시보드/테마.py — 새 대시보드의 디자인 토큰(2026-10-11, 사람 결정)
# 밝은 테마 고정, 연회색 배경 + 흰 카드. 파랑은 데이터(막대·선)에만 쓰고 배경·제목·상태에는 쓰지 않는다.
# 색 값은 참고 이미지(저장소 밖)에서 뽑은 실제 픽셀 값이다. 상태 글자색은 흰 바탕·칸 배경·페이지
# 배경 모두에 대해 WCAG 대비 4.5 이상인 진한 변형이고, 연한 원색은 칸 배경과 큰 면에만 쓴다.
# 차트는 모두 apply_layout()과 show_chart()를 거친다 — marker_color를 직접 지정하지 않는다.

import html

import streamlit as st

# ── 중립색 ──
CARD_BG = "#FFFFFF"
SUBTLE_BG = "#F8F9FB"     # 표 머리·줄무늬
BORDER = "#EEF0F3"
TEXT = "#383B42"          # 대비 11.22(흰)
TEXT_MUTED = "#6E7076"    # 보조 글자, 대비 4.95(흰) — 카드(흰 바탕) 안에서만 쓴다(페이지 배경 위 4.46)
GRID = "#EEF0F3"
FONT = "'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif"

# ── 데이터 색(차트 순서 고정). 노랑은 큰 면에만, 상태에는 쓰지 않는다 ──
BLUE = "#4673D8"
TEAL = "#5FC3B9"
YELLOW = "#F7C846"
ORANGE = "#E17331"
LILAC = "#D3A5E1"
DATA_COLORS = [BLUE, TEAL, YELLOW, ORANGE, LILAC]

# ── 상태 6종: 칸 배경(원색 16% 혼합) / 글자·아이콘(진한 변형) / 아이콘 / 글자 ──
# 대비(글자 vs 흰, 칸 배경, 페이지 배경): success 5.20/4.63/4.69, running 5.15/4.65/4.64, warning 5.47/4.63/4.92,
# failure 5.74/4.68/5.17, waiting 5.12/4.66/4.62, skipped 5.41/4.62/4.88
STATUS = {
    "success": {"bg": "#E5F5F4", "fg": "#2D7871", "icon": "✓", "label": "성공"},
    "running": {"bg": "#F8F1FA", "fg": "#A242C0", "icon": "◐", "label": "진행"},
    "warning": {"bg": "#FAE9DE", "fg": "#AA4F19", "icon": "⚠", "label": "경고"},
    "failure": {"bg": "#F9E3E2", "fg": "#BE3128", "icon": "✕", "label": "실패"},
    "waiting": {"bg": "#F4F4F6", "fg": "#676E7C", "icon": "○", "label": "대기"},
    "skipped": {"bg": "#EDEDED", "fg": "#696A6C", "icon": "–", "label": "건너뜀"},
}

CSS = f"""
<style>
[class*="st-key-card"] {{
  background: {CARD_BG}; border-radius: 16px; border: 1px solid {BORDER};
  box-shadow: 0 1px 3px rgba(16, 24, 40, .06); padding: 20px 24px;
}}
.status-chip {{ display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 999px;
  font-size: 0.82rem; white-space: nowrap; }}
.status-skipped {{ background: repeating-linear-gradient(45deg, #EDEDED 0 4px, #F8F8F9 4px 8px) !important; }}
/* 실패는 굵게 + 안쪽 테두리 — 경고(주황)와 붙어 있어도 모양으로 구분(색은 그대로) */
.status-failure {{ font-weight: 700; box-shadow: inset 0 0 0 1.5px {STATUS["failure"]["fg"]}; }}
.stepper {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }}
.stepper .arrow {{ color: {TEXT_MUTED}; }}
.grid14 {{ border-collapse: separate; border-spacing: 3px; font-size: 0.78rem; width: 100%; }}
.grid14 th {{ color: {TEXT_MUTED}; font-weight: 500; background: {SUBTLE_BG}; padding: 4px; border-radius: 6px; }}
.grid14 td {{ text-align: center; padding: 4px 2px; border-radius: 6px; white-space: nowrap; }}
.grid14 td.stage {{ text-align: left; color: {TEXT}; background: transparent; padding-right: 8px; }}
/* KPI: 값은 줄바꿈 허용(잘림 방지), 보조 글자는 2줄에서 자른다(전문은 마우스 올림). 높이 고정으로 카드 4개를 맞춘다 */
.kpi {{ height: 10.5rem; display: flex; flex-direction: column; gap: 4px; overflow: hidden; }}
.kpi-label {{ color: {TEXT_MUTED}; font-size: 0.875rem; }}
.kpi-value {{ color: {TEXT}; font-size: 1.75rem; line-height: 1.2; font-weight: 600; overflow-wrap: anywhere; }}
.kpi-sub {{ color: {TEXT_MUTED}; font-size: 0.8rem; line-height: 1.4; margin-top: auto; display: -webkit-box;
  -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }}
</style>
"""


def inject_css():
    st.html(CSS)


def status_chip(status, text=None):
    """상태를 색 + 아이콘 + 글자로 그리는 HTML 조각. text가 없으면 상태 이름을 쓴다."""
    s = STATUS[status]
    cls = "status-chip" + {"skipped": " status-skipped", "failure": " status-failure"}.get(status, "")
    label = html.escape(text or s["label"])
    return (f'<span class="{cls}" style="background:{s["bg"]};color:{s["fg"]}">'
            f'<b>{s["icon"]}</b> {label}</span>')


def kpi_html(label, value, sub=""):
    """KPI 카드 내용. st.metric은 값을 한 줄로 자르므로(10/11 화면 확인) 줄바꿈되는 HTML로 그린다."""
    e = html.escape
    return (f'<div class="kpi"><div class="kpi-label">{e(label)}</div><div class="kpi-value">{e(value)}</div>'
            f'<div class="kpi-sub" title="{e(sub)}">{e(sub)}</div></div>')


PLOTLY_LAYOUT = dict(
    colorway=DATA_COLORS,
    font=dict(family=FONT, color=TEXT, size=13),
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    margin=dict(l=64, r=8, t=36, b=8),   # 왼쪽은 y축 제목·눈금 자리(8이면 잘림 — 2026-10-11 스크래치 렌더 확인)
    xaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False, automargin=True),
    yaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False, automargin=True),
    legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(font_family=FONT),
)


def apply_layout(fig, **overrides):
    fig.update_layout(**PLOTLY_LAYOUT)
    if overrides:
        fig.update_layout(**overrides)
    return fig


def show_chart(fig, **overrides):
    """공통 레이아웃을 적용해 그린다. theme=None으로 Streamlit 테마가 레이아웃을 덮어쓰지 않게 한다."""
    st.plotly_chart(apply_layout(fig, **overrides), theme=None, width="stretch")
