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

import pandas as pd
import plotly.graph_objects as go
import psycopg
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

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
                   gate_vs_parkinson, model_version
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
