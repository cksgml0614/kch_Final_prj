# 대시보드/테마.py — 운영 대시보드의 디자인 토큰(색·타입 스케일·상태 6종·아이콘), CSS, HTML 조각, plotly 공통 레이아웃.
# 밝은 테마 고정(연회색 배경 + 흰 카드). 파랑은 데이터(막대·선)에만 쓰고 배경·제목·상태에는 쓰지 않는다.
# 상태 글자색은 흰 바탕·칸 배경·페이지 배경 모두에 WCAG 대비 4.5 이상인 진한 변형, 연한 원색은 칸 배경과 큰 면에만 쓴다.
# 차트는 모두 show_chart()(apply_layout)를 거친다 — marker_color를 직접 지정하지 않는다.
# 배경: 결정기록.md의 "밝은 테마 고정, 파랑은 데이터 전용"

import html
from urllib.parse import quote

import streamlit as st

# ── 중립색 ──
CARD_BG = "#FFFFFF"
SUBTLE_BG = "#F8F9FB"     # 표 머리·줄무늬·기록 없는 칸
BORDER = "#EEF0F3"
TEXT = "#383B42"          # 대비 11.22(흰)
TEXT_MUTED = "#6E7076"    # 보조 글자, 대비 4.95(흰) — 카드(흰 바탕) 안에서만 쓴다(페이지 배경 위 4.46)
TEXT_MUTED_PAGE = "#65686E"  # 페이지 배경 위 보조 글자, 대비 5.03(페이지)·5.59(흰)
DOT = "#8B909A"           # 격자 "기록 없음" 점, 대비 3.04(기록 없는 칸 배경)·3.20(흰)
GRID = "#EEF0F3"
FONT = "'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif"

# ── 타입 스케일(px): KPI 값 24(좁은 화면은 18까지 줄어든다) / 제목 20 / 섹션 16 / 본문 14 / 보조 12 ──
FS_KPI, FS_TITLE, FS_SECTION, FS_BODY, FS_SMALL = 24, 20, 16, 14, 12

# ── 데이터 색(차트 순서 고정). 노랑은 큰 면에만, 상태에는 쓰지 않는다 ──
BLUE = "#4673D8"
TEAL = "#5FC3B9"
YELLOW = "#F7C846"
ORANGE = "#E17331"
LILAC = "#D3A5E1"
DATA_COLORS = [BLUE, TEAL, YELLOW, ORANGE, LILAC]

# ── 상태 6종: 칸 배경(원색 16% 혼합) / 글자·아이콘(진한 변형) / 글자 ──
# 대비(글자 vs 흰, 칸 배경, 페이지 배경): success 5.20/4.63/4.69, running 5.15/4.65/4.64, warning 5.47/4.63/4.92,
# failure 5.74/4.68/5.17, waiting 5.12/4.66/4.62, skipped 5.41/4.62/4.88
STATUS = {
    "success": {"bg": "#E5F5F4", "fg": "#2D7871", "label": "성공"},
    "running": {"bg": "#F8F1FA", "fg": "#A242C0", "label": "진행"},
    "warning": {"bg": "#FAE9DE", "fg": "#AA4F19", "label": "경고"},
    "failure": {"bg": "#F9E3E2", "fg": "#BE3128", "label": "실패"},
    "waiting": {"bg": "#F4F4F6", "fg": "#676E7C", "label": "대기"},
    "skipped": {"bg": "#EDEDED", "fg": "#696A6C", "label": "건너뜀"},
}

# 상태 아이콘: 16px 격자, 선 굵기 2의 같은 계열. 색만으로 구분하지 않도록 모양이 모두 다르다.
# 문자(✓⚠ 등)는 글꼴에 따라 이모지로 그려져 굵기·크기가 들쭉날쭉해서 CSS mask로 그린다(색 = currentColor).
_ICON_SVG = {
    "success": '<path d="M3.5 8.5l3 3 6-7" fill="none" stroke="#000" stroke-width="2" stroke-linecap="round" '
               'stroke-linejoin="round"/>',
    "running": '<circle cx="8" cy="8" r="5.5" fill="none" stroke="#000" stroke-width="2"/>'
               '<path d="M8 2.5a5.5 5.5 0 0 1 0 11z"/>',
    "warning": '<path d="M8 2.5L14 13H2z" fill="none" stroke="#000" stroke-width="2" stroke-linejoin="round"/>'
               '<path d="M8 6.8v2.6M8 11.1v.1" stroke="#000" stroke-width="2" stroke-linecap="round"/>',
    "failure": '<path d="M4 4l8 8M12 4l-8 8" stroke="#000" stroke-width="2" stroke-linecap="round"/>',
    "waiting": '<circle cx="8" cy="8" r="5" fill="none" stroke="#000" stroke-width="2"/>',
    "skipped": '<path d="M3.5 8h9" stroke="#000" stroke-width="2" stroke-linecap="round"/>',
    # ⑧ 주간 튜닝 결과(같은 굵기): 채택 ↑, 유지 =, 복귀 ↩, 기본 유지 □
    "tune-adopted": '<path d="M8 13V3.5M4 7.5l4-4 4 4" fill="none" stroke="#000" stroke-width="2" stroke-linecap="round" '
                    'stroke-linejoin="round"/>',
    "tune-kept": '<path d="M3.5 6h9M3.5 10h9" stroke="#000" stroke-width="2" stroke-linecap="round"/>',
    "tune-reverted": '<path d="M6 3.5l-3 3 3 3" fill="none" stroke="#000" stroke-width="2" stroke-linecap="round" '
                     'stroke-linejoin="round"/><path d="M3.5 6.5H10a3 3 0 0 1 0 6H7" fill="none" stroke="#000" '
                     'stroke-width="2" stroke-linecap="round"/>',
    "tune-none": '<rect x="3.5" y="3.5" width="9" height="9" rx="1.5" fill="none" stroke="#000" stroke-width="2"/>',
}


def _icon_css():
    """
    설명: _ICON_SVG의 아이콘마다 .ico-<이름> CSS 규칙(SVG data URI를 mask-image로)을 만들어 한 문자열로 돌려준다.
    주의 사항: 색은 mask라 background-color(= currentColor)를 따른다.
    입력 타입->반환 타입: () -> str
    """
    rules = []
    for k, body in _ICON_SVG.items():
        uri = "data:image/svg+xml," + quote(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">{body}</svg>')
        rules.append(f'.ico-{k} {{ -webkit-mask-image: url("{uri}"); mask-image: url("{uri}"); }}')
    return "\n".join(rules)


CSS = f"""
<style>
/* 타입 스케일 */
[data-testid="stHeading"] h1 {{ font-size: {FS_TITLE}px; font-weight: 700; line-height: 1.4; padding: 0; }}
[data-testid="stHeading"] h3, [data-testid="stHeading"] h3 * {{ font-size: {FS_SECTION}px; font-weight: 700; line-height: 1.5; }}
[data-testid="stHeading"] h3 {{ padding: 0 0 8px 0; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ font-size: {FS_SMALL}px; color: {TEXT_MUTED}; }}
.page-sub {{ font-size: {FS_SMALL}px; color: {TEXT_MUTED_PAGE}; }}
.sub {{ font-size: {FS_SMALL}px; color: {TEXT_MUTED}; line-height: 1.5; }}
.body {{ font-size: {FS_BODY}px; color: {TEXT}; line-height: 1.5; }}
.h4 {{ font-size: {FS_BODY}px; font-weight: 600; color: {TEXT}; }}
.num {{ font-variant-numeric: tabular-nums; }}
.info {{ color: {TEXT_MUTED}; cursor: help; font-size: {FS_SMALL}px; }}

[class*="st-key-card"] {{
  background: {CARD_BG}; border-radius: 16px; border: 1px solid {BORDER}; padding: 16px 24px; gap: 8px;
}}

/* 상태 아이콘(같은 굵기·크기) */
.ico {{ display: inline-block; width: 12px; height: 12px; flex: none; background-color: currentColor;
  -webkit-mask-size: contain; mask-size: contain; -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat;
  vertical-align: -1px; }}
{_icon_css()}

.status-chip {{ display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 999px;
  font-size: {FS_SMALL}px; line-height: 1.5; white-space: nowrap; font-variant-numeric: tabular-nums; }}
.status-skipped {{ background: repeating-linear-gradient(45deg, #EDEDED 0 4px, #F8F8F9 4px 8px) !important; }}
/* 실패는 굵게 + 안쪽 테두리 — 경고(주황)와 붙어 있어도 모양으로 구분(색은 그대로) */
.status-failure {{ font-weight: 700; box-shadow: inset 0 0 0 1.5px {STATUS["failure"]["fg"]}; }}
.tag {{ display: inline-flex; align-items: center; padding: 2px 8px; border-radius: 999px; font-size: {FS_SMALL}px;
  font-weight: 600; color: {TEXT}; background: {SUBTLE_BG}; border: 1px solid {BORDER}; white-space: nowrap; }}
.gdot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; vertical-align: 1px; }}

.stepper {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; font-size: {FS_BODY}px; }}
.stepper .arrow {{ color: {TEXT_MUTED}; }}
.stepper > span {{ display: inline-flex; align-items: center; gap: 4px; white-space: nowrap; }}

/* 14일 × 단계 격자: 줄넘김 없음, 칸 크기 고정, 좁으면 가로 스크롤 + 첫 열 고정 */
.gridwrap {{ overflow-x: auto; }}
.grid14 {{ table-layout: fixed; border-collapse: separate; border-spacing: 3px; width: 100%; min-width: 1064px;
  font-size: {FS_SMALL}px; font-variant-numeric: tabular-nums; }}
.grid14 col.stage {{ width: 120px; }}
.grid14 th, .grid14 td {{ white-space: nowrap; overflow: hidden; text-align: center; }}
.grid14 thead th {{ font-weight: 600; color: {TEXT}; padding: 4px 0; line-height: 1.3; border-top: 2px solid transparent; }}
.grid14 thead th .wd {{ display: block; font-size: 11px; font-weight: 400; color: {TEXT_MUTED}; }}
.grid14 thead th.weekend {{ color: {TEXT_MUTED}; font-weight: 400; }}
.grid14 thead th.today {{ border-top-color: {TEXT}; background: {SUBTLE_BG}; border-radius: 0 0 6px 6px; }}
.grid14 thead th.now {{ border-top: 2px dashed {TEXT_MUTED}; }}
.grid14 th.stage {{ position: sticky; left: 0; z-index: 1; background: {CARD_BG}; text-align: left; font-weight: 500;
  color: {TEXT}; padding: 0 8px 0 0; box-shadow: 3px 0 0 {CARD_BG}; }}
.grid14 tbody th.stage {{ border-bottom: 1px solid {BORDER}; }}
.grid14 td {{ height: 36px; min-width: 64px; padding: 0 4px; border-radius: 6px; position: relative; }}
.grid14 td .cell {{ display: inline-flex; align-items: center; gap: 4px; }}
.grid14 td.none {{ background: {SUBTLE_BG}; }}
.nodot {{ display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: {DOT}; vertical-align: middle; }}
/* 덮어씀 표시: 칸 오른쪽 위 작은 삼각형 */
.over::after {{ content: ""; position: absolute; top: 0; right: 0; border-style: solid; border-width: 0 8px 8px 0;
  border-color: transparent {TEXT_MUTED} transparent transparent; border-top-right-radius: 6px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 16px; font-size: {FS_SMALL}px; color: {TEXT_MUTED}; margin-top: 8px; }}
.legend > span {{ display: inline-flex; align-items: center; gap: 4px; white-space: nowrap; }}
.legend .swatch {{ position: relative; display: inline-flex; align-items: center; justify-content: center;
  width: 20px; height: 16px; border-radius: 4px; background: {SUBTLE_BG}; }}

/* 수치 표(패널 C 요약·평가일별) */
.dwrap {{ overflow-x: auto; }}
.dtable {{ border-collapse: collapse; width: 100%; font-size: {FS_BODY}px; color: {TEXT}; font-variant-numeric: tabular-nums; }}
.dtable th {{ font-size: {FS_SMALL}px; font-weight: 500; color: {TEXT_MUTED}; text-align: left; padding: 8px;
  border-bottom: 1px solid {BORDER}; white-space: nowrap; }}
.dtable td {{ padding: 8px; border-bottom: 1px solid {BORDER}; white-space: nowrap; }}
.dtable tr:last-child td {{ border-bottom: none; }}
.dtable .r {{ text-align: right; }}
.dtable td.best {{ font-weight: 700; color: {TEXT}; }}
.dtable .chips {{ display: flex; flex-wrap: wrap; gap: 4px; }}  /* 좁은 폭에서 칩만 줄바꿈 — 수치 열이 화면 밖으로 밀리지 않게 */
.dtable.compact td {{ padding: 4px 8px; font-size: {FS_SMALL}px; }}

/* KPI: 값 한 줄, 보조 글자 한 줄(넘치면 말줄임, 전문은 마우스 올림). 높이 고정으로 카드 4개를 맞춘다 */
.kpi {{ height: 88px; display: flex; flex-direction: column; gap: 4px; }}
.kpi-label {{ color: {TEXT_MUTED}; font-size: {FS_SMALL}px; }}
.kpi-value {{ color: {TEXT}; font-size: clamp(18px, 1.8vw, {FS_KPI}px); line-height: 1.3; font-weight: 700; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; display: flex; align-items: center; gap: 8px;
  font-variant-numeric: tabular-nums; }}
.kpi-value .ico {{ width: 20px; height: 20px; }}
.kpi-sub {{ color: {TEXT_MUTED}; font-size: {FS_SMALL}px; line-height: 1.5; margin-top: auto; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; font-variant-numeric: tabular-nums; }}
</style>
"""


def inject_css():
    """
    설명: 페이지에 대시보드 CSS를 넣는다.
    주의 사항: 페이지 맨 앞에서 한 번 부른다.
    입력 타입->반환 타입: () -> None
    """
    st.html(CSS)


def icon(status):
    """
    설명: 상태(또는 tune-* 튜닝 결과) 아이콘 HTML을 돌려준다.
    주의 사항: 색은 감싼 요소의 글자색을 따른다. 장식용이라 aria-hidden.
    입력 타입->반환 타입: (str) -> str
    """
    return f'<span class="ico ico-{status}" aria-hidden="true"></span>'


def status_chip(status, text=None, title=None):
    """
    설명: 상태를 색 + 아이콘 + 글자로 그리는 칩 HTML을 돌려준다.
    주의 사항: status는 STATUS 6종 중 하나. text가 없으면 상태 이름을 쓰고, title은 마우스 올림 설명. 글자는 escape한다.
    입력 타입->반환 타입: (str, str|None, str|None) -> str
    """
    s = STATUS[status]
    cls = "status-chip" + {"skipped": " status-skipped", "failure": " status-failure"}.get(status, "")
    label = html.escape(text or s["label"])
    tip = f' title="{html.escape(title)}"' if title else ""
    return f'<span class="{cls}" style="background:{s["bg"]};color:{s["fg"]}"{tip}>{icon(status)}{label}</span>'


def info(tip):
    """
    설명: 접힌 설명 ⓘ의 HTML을 돌려준다(마우스를 올리면 tip 전문이 보인다).
    주의 사항: tip은 escape한다.
    입력 타입->반환 타입: (str) -> str
    """
    return f'<span class="info" title="{html.escape(tip)}">ⓘ</span>'


def kpi_html(label, value, sub="", tip="", status=None):
    """
    설명: KPI 카드 내용(라벨·값·보조 글자, 선택적 상태 아이콘) HTML을 돌려준다.
    주의 사항: st.metric은 긴 값을 잘라서 HTML로 그린다. 값·보조 글자는 한 줄로 말줄임되고 전문은 title(tip이 없으면 sub)에 둔다.
    입력 타입->반환 타입: (str, str, str, str, str|None) -> str
    """
    e = html.escape
    ico = f'<span style="color:{STATUS[status]["fg"]};display:inline-flex">{icon(status)}</span>' if status else ""
    return (f'<div class="kpi"><div class="kpi-label">{e(label)}</div>'
            f'<div class="kpi-value" title="{e(value)}">{ico}<span>{e(value)}</span></div>'
            f'<div class="kpi-sub" title="{e(tip or sub)}">{e(sub)}</div></div>')


PLOTLY_LAYOUT = dict(
    colorway=DATA_COLORS,
    font=dict(family=FONT, color=TEXT, size=FS_SMALL),
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    margin=dict(l=64, r=8, t=36, b=8),   # 왼쪽은 y축 제목·눈금 자리(8이면 잘린다)
    xaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False, automargin=True),
    yaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False, automargin=True),
    legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(font_family=FONT),
)


def apply_layout(fig, **overrides):
    """
    설명: 그림에 PLOTLY_LAYOUT을 적용하고 overrides로 덮어쓴 뒤 같은 그림을 돌려준다.
    주의 사항: fig를 직접 바꾼다.
    입력 타입->반환 타입: (go.Figure, **object) -> go.Figure
    """
    fig.update_layout(**PLOTLY_LAYOUT)
    if overrides:
        fig.update_layout(**overrides)
    return fig


def show_chart(fig, **overrides):
    """
    설명: 공통 레이아웃을 적용해 차트를 그린다.
    주의 사항: theme=None으로 Streamlit 테마가 레이아웃을 덮어쓰지 않게 한다.
    입력 타입->반환 타입: (go.Figure, **object) -> None
    """
    st.plotly_chart(apply_layout(fig, **overrides), theme=None, width="stretch")
