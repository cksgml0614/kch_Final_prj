# 대시보드/model_predictions_dashboard.py — model_predictions 조회 전용 Streamlit 대시보드
# (2026-09-20). 상시 서버로 띄우지 않고 필요할 때 로컬에서 실행한다:
#   streamlit run 대시보드/model_predictions_dashboard.py
#
# ⚠️ config.py의 USE_CLOUD_DB 토글과 무관하게 항상 NEON_DB_URL(pooled)로 직접 접속한다 —
# 이 대시보드는 "지금 로컬 개발 환경이 어느 DB를 보고 있는지"와 상관없이 항상 운영(클라우드)
# 데이터를 보여주는 것이 목적이라, db_manager.get_db_connection()의 로컬/클라우드 분기를
# 거치지 않고 os.getenv("NEON_DB_URL")을 바로 쓴다.
#
# actual_volatility 백필(가격예측/actual_volatility_백필.py)이 일일 파이프라인에 아직
# 안 물려있어(CLAUDE.md "알려진 이슈" 참고) 최근 며칠 target_date는 항상 NULL로 보인다 —
# 이게 버그가 아니라 알려진 상태라는 걸 화면에 명시한다(상단 상태 표시 + (c)의 표본 수 표기).

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import psycopg
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# `streamlit run 대시보드/...`로 실행하면 sys.path에 대시보드/만 들어가므로 프로젝트 루트를 추가한다
# (constants.STOCK_NAMES 종목명 표시용, 2026-09-27).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import STOCK_NAMES  # noqa: E402

st.set_page_config(page_title="변동성 예측 대시보드", layout="wide")

NEON_DB_URL = os.getenv("NEON_DB_URL")


@st.cache_data(ttl=300)
def load_predictions():
    if not NEON_DB_URL:
        st.error("NEON_DB_URL이 .env에 없습니다 — 클라우드 DB 접속 정보를 확인하세요.")
        st.stop()
    with psycopg.connect(NEON_DB_URL) as conn:
        df = pd.read_sql(
            """
            SELECT ticker, target_date, prediction_date, predicted_volatility,
                   garch_baseline, sma20_baseline, parkinson_sma20_baseline,
                   actual_volatility, gate_passed, gate_vs_garch, gate_vs_sma20,
                   gate_vs_parkinson, model_version, input_data_suspect
            FROM model_predictions
            ORDER BY target_date, ticker
            """,
            conn,
        )
    df["target_date"] = pd.to_datetime(df["target_date"])
    return df


df = load_predictions()

st.title("변동성 예측 대시보드 (model_predictions)")

# ── 상단 상태 표시: actual_volatility 채움 비율 ──
total = len(df)
filled = int(df["actual_volatility"].notna().sum())
null_n = total - filled
null_ratio = (null_n / total * 100) if total else 0.0
st.info(
    f"전체 {total:,}행 중 실측(actual_volatility) 확정 **{filled:,}행**"
    f"({100 - null_ratio:.1f}%) / 미확정(NULL) **{null_n:,}행**({null_ratio:.1f}%). "
    "actual_volatility 백필이 일일 파이프라인에 자동으로 물려있지 않아, "
    "최근 며칠치 target_date는 항상 비어있는 게 정상입니다(버그 아님)."
)

tickers = sorted(df["ticker"].unique())
selected = st.sidebar.selectbox("종목 선택 ((a)/(c)에 적용)", ["전체(평균)"] + tickers)

if selected == "전체(평균)":
    filtered = df
else:
    filtered = df[df["ticker"] == selected]

# ============================================================
# (a) 예측 vs 실측 시계열
# ============================================================
st.subheader("(a) 예측 vs 실측 시계열")

if selected == "전체(평균)":
    plot_df = (
        filtered.groupby("target_date")
        .agg(predicted_volatility=("predicted_volatility", "mean"),
             actual_volatility=("actual_volatility", "mean"))
        .reset_index()
    )
    title_suffix = "(전체 종목 평균)"
else:
    plot_df = filtered.sort_values("target_date")
    title_suffix = f"({selected})"

confirmed = plot_df.dropna(subset=["actual_volatility"])

fig_a = go.Figure()
fig_a.add_trace(go.Scatter(
    x=plot_df["target_date"], y=plot_df["predicted_volatility"],
    mode="lines+markers", name="예측(predicted_volatility)",
))
fig_a.add_trace(go.Scatter(
    x=confirmed["target_date"], y=confirmed["actual_volatility"],
    mode="lines+markers", name="실측(actual_volatility, 확정)",
))
if not confirmed.empty and plot_df["target_date"].max() > confirmed["target_date"].max():
    fig_a.add_vrect(
        x0=confirmed["target_date"].max(), x1=plot_df["target_date"].max(),
        fillcolor="gray", opacity=0.15, line_width=0,
        annotation_text="실측 미확정 구간", annotation_position="top left",
    )
fig_a.update_layout(title=f"예측 vs 실측 {title_suffix}", xaxis_title="target_date",
                     yaxis_title="변동성(|로그수익률x100|)", height=450)
st.plotly_chart(fig_a, use_container_width=True)

# ============================================================
# (b) 최근일 종목별 결과 테이블 (필터와 무관하게 항상 전 종목)
# ============================================================
st.subheader("(b) 최근일 종목별 결과")
latest_date = df["target_date"].max()
latest = df[df["target_date"] == latest_date].sort_values("ticker").reset_index(drop=True)
st.caption(f"기준일(target_date): {latest_date.date()}  —  {len(latest)}종목")
st.dataframe(
    latest[["ticker", "predicted_volatility", "garch_baseline", "sma20_baseline",
            "parkinson_sma20_baseline", "actual_volatility",
            "gate_passed", "gate_vs_garch", "gate_vs_sma20", "gate_vs_parkinson"]],
    use_container_width=True,
    column_config={
        "gate_passed": st.column_config.CheckboxColumn("게이트 통과"),
        "gate_vs_garch": st.column_config.CheckboxColumn("vs GARCH"),
        "gate_vs_sma20": st.column_config.CheckboxColumn("vs SMA20"),
        "gate_vs_parkinson": st.column_config.CheckboxColumn("vs Parkinson"),
    },
)

# ============================================================
# (c) 시간에 따른 RMSE/MAE 추이 (실측 확정된 행만)
# ============================================================
st.subheader("(c) 시간에 따른 RMSE/MAE 추이")
confirmed_all = filtered.dropna(subset=["actual_volatility"]).copy()
st.caption(f"{len(filtered):,}건 중 실측 확정 {len(confirmed_all):,}건만 사용(미확정 행 자동 제외)")

if confirmed_all.empty:
    st.warning("선택한 종목에는 아직 실측이 확정된 행이 없습니다.")
else:
    confirmed_all["abs_error"] = (confirmed_all["predicted_volatility"] - confirmed_all["actual_volatility"]).abs()
    confirmed_all["sq_error"] = (confirmed_all["predicted_volatility"] - confirmed_all["actual_volatility"]) ** 2
    daily = (
        confirmed_all.groupby("target_date")
        .agg(mae=("abs_error", "mean"), rmse=("sq_error", lambda s: s.mean() ** 0.5), n=("abs_error", "count"))
        .reset_index()
    )
    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(x=daily["target_date"], y=daily["rmse"], mode="lines+markers", name="RMSE"))
    fig_c.add_trace(go.Scatter(x=daily["target_date"], y=daily["mae"], mode="lines+markers", name="MAE"))
    fig_c.update_layout(title=f"일별 RMSE/MAE 추이 {title_suffix}", xaxis_title="target_date",
                         yaxis_title="오차", height=400)
    st.plotly_chart(fig_c, use_container_width=True)

# ============================================================
# (d) 날짜별 전 종목 뷰 + 예측 상위 10종목 (2026-09-27 추가)
# ============================================================
# 종목 필터와 무관하게 선택한 target_date 하루의 전 종목을 보여준다. 2026-09-28 실행부터 게이트
# 집계 버그가 수정돼 gate_passed=true 행이 쌓이기 시작하므로, 날짜별 통과 여부·입력 오염 플래그
# (input_data_suspect)를 함께 표시해 "이 날 예측을 믿어도 되는지"를 바로 볼 수 있게 한다.
st.subheader("(d) 날짜별 전 종목 예측 + 상위 10종목")
TOP_N = 10
dates = sorted(df["target_date"].dt.date.unique(), reverse=True)
# ?date=YYYY-MM-DD 쿼리 파라미터로 기본 날짜 지정 가능(특정 날짜 링크 공유용), 없으면 최신일
q = st.query_params.get("date")
default_idx = next((i for i, d in enumerate(dates) if str(d) == q), 0)
sel_date = st.selectbox("target_date 선택", dates, index=default_idx, key="d_date")
day = df[df["target_date"].dt.date == sel_date].copy()
day["종목명"] = day["ticker"].map(STOCK_NAMES).fillna("")
day["label"] = day["종목명"].where(day["종목명"] != "", day["ticker"])   # 축은 종목명만(100개라 겹침 방지), 코드는 hover·표에
day = day.sort_values("predicted_volatility", ascending=False).reset_index(drop=True)
day["rank"] = day.index + 1

n_pass, n_suspect, n_actual = int(day["gate_passed"].sum()), int(day["input_data_suspect"].sum()), int(day["actual_volatility"].notna().sum())
status = []
status.append(f"게이트 통과 {n_pass}/{len(day)}")
status.append(f"입력 오염 의심 {n_suspect}/{len(day)}")
status.append(f"실측 확정 {n_actual}/{len(day)}")
pred_dates = sorted(day["prediction_date"].astype(str).unique())
versions = sorted(day["model_version"].unique())
st.caption(f"prediction_date {', '.join(pred_dates)} · 모델 버전 {', '.join(versions)} · " + " · ".join(status))
if n_suspect:
    st.warning("이 날짜 예측은 input_data_suspect=true(입력 데이터 오염 의심)입니다 — 참고용으로만 보세요.")
elif n_pass == 0:
    st.warning("이 날짜 예측은 배포 게이트 미통과(gate_passed=false)입니다. 2026-09-27 이전 판정은 게이트 집계 버그 "
               "(CLAUDE.md 참고) 영향을 받았을 수 있습니다.")

is_top = day["rank"] <= TOP_N
fig_d = go.Figure()
fig_d.add_trace(go.Bar(
    x=day["label"], y=day["predicted_volatility"], name="예측",
    marker_color=["#d62728" if t else "#9ecae1" for t in is_top],
    customdata=day[["rank", "gate_passed", "input_data_suspect", "ticker"]].values,
    hovertemplate="%{x}(%{customdata[3]})<br>예측=%{y:.3f}<br>순위=%{customdata[0]}<br>gate_passed=%{customdata[1]}"
                  "<br>suspect=%{customdata[2]}<extra></extra>",
))
if n_actual:
    act = day.dropna(subset=["actual_volatility"])
    fig_d.add_trace(go.Scatter(
        x=act["label"], y=act["actual_volatility"], mode="markers", name="실측",
        marker=dict(symbol="diamond", size=8, color="#ffb000", line=dict(color="#333333", width=1)),
        hovertemplate="%{x}<br>실측=%{y:.3f}<extra></extra>",
    ))
fig_d.update_layout(
    title=f"{sel_date} 전 종목 예측 변동성(내림차순, 빨강=상위 {TOP_N})" + (" + 실측" if n_actual else ""),
    xaxis=dict(tickangle=-90, tickfont=dict(size=8), tickmode="linear", dtick=1), yaxis_title="변동성(|로그수익률x100|)",
    height=560, bargap=0.15, legend=dict(orientation="h", y=1.08),
)
st.plotly_chart(fig_d, use_container_width=True)

top = day[is_top][["rank", "ticker", "종목명", "predicted_volatility", "actual_volatility",
                  "gate_passed", "input_data_suspect"]]
st.dataframe(
    top, use_container_width=True, hide_index=True,
    column_config={
        "rank": st.column_config.NumberColumn("순위"),
        "ticker": st.column_config.TextColumn("종목코드"),
        "predicted_volatility": st.column_config.NumberColumn("예측값", format="%.3f"),
        "actual_volatility": st.column_config.NumberColumn("실측", format="%.3f"),
        "gate_passed": st.column_config.CheckboxColumn("게이트 통과"),
        "input_data_suspect": st.column_config.CheckboxColumn("입력 오염 의심"),
    },
)
