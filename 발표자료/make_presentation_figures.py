# 발표자료/make_presentation_figures.py — 발표용 그래프 3종 프로토타입(2026-09-27)
# 입력: fold3 val(2024-10-11~2025-09-29, 100종목) 하이브리드 예측(seed 42, 운영 설정 재현) 행 단위 parquet.
# 원칙: 실제 데이터 그대로 — 막대 y축 0 시작, 산점도 전체 범위·x/y 동일 스케일, 급변일 정의는 기존 그대로.
# 실행: python -m 발표자료.make_presentation_figures --rows <val_diag_rows.parquet>
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#b4b2ab"      # 검증된 참조 팔레트 slot1/slot2 + 중립 회색
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 15,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
OUT = os.path.dirname(os.path.abspath(__file__))
PERIOD = "fold3 val 2024-10-11 ~ 2025-09-29 · 100종목"


def fig_rmse():
    # 게이트 집계 버그 수정 후 대칭 비교 값(종목별 RMSE 평균, 같은 23,737행) — CLAUDE.md "게이트 집계 버그" 절
    names = ["하이브리드\n(GARCH+Transformer)", "GARCH(1,1)", "SMA20", "Parkinson-SMA20"]
    vals = [1.9594, 2.1450, 2.0177, 2.0013]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    bars = ax.bar(names, vals, color=[BLUE, NEUTRAL, NEUTRAL, NEUTRAL], width=0.6, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.4f}", ha="center", va="bottom",
                fontsize=16, color=INK, fontweight="bold" if b is bars[0] else "normal")
    ax.set_ylim(0, 2.5)                                   # 0 시작(막대 길이 = 값)
    ax.set_ylabel("RMSE (낮을수록 좋음)")
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_title("다음날 변동성 예측 오차 — 하이브리드가 세 baseline보다 낮음", fontsize=18, color=INK, pad=14, loc="left")
    best = min(vals[1:])
    fig.text(0.01, 0.01, f"{PERIOD} · 종목별 RMSE 평균 · 최선 baseline(Parkinson-SMA20) 대비 {1 - vals[0] / best:.1%} 낮음",
             fontsize=11, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p = os.path.join(OUT, "fig1_rmse_comparison.png"); fig.savefig(p, dpi=200); plt.close(fig); return p


def fig_scatter(v):
    """왼쪽: 전체 범위(아무것도 숨기지 않음), 오른쪽: 0~ZOOM 확대(범위 밖 점 수 명시). 둘 다 x·y 동일 스케일."""
    calm, spike = v[~v.spike], v[v.spike]
    lim = float(np.ceil(max(v.realized.max(), v.hybrid.max())))
    ZOOM = 10.0
    edges = np.array([0, 1, 2, 3, 4, 6, 8, 12, 20, lim])
    mid, med = [], []
    for a, b in zip(edges[:-1], edges[1:]):          # 실측 구간별 예측 중앙값(요약선, 데이터 가공 아님)
        m = (v.realized >= a) & (v.realized < b)
        if m.sum() >= 30: mid.append(v.realized[m].median()); med.append(v.hybrid[m].median())
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.6))
    for ax, L, title in ((axes[0], lim, "전체 범위"), (axes[1], ZOOM, f"0~{ZOOM:g} 확대")):
        ax.scatter(calm.realized, calm.hybrid, s=9, c=BLUE, alpha=0.25, linewidths=0, label=f"평상일 (n={len(calm):,})", zorder=2)
        ax.scatter(spike.realized, spike.hybrid, s=12, c=ORANGE, alpha=0.45, linewidths=0, label=f"급변일 (n={len(spike):,})", zorder=3)
        ax.plot([0, L], [0, L], color=INK, lw=1.5, ls="--", zorder=4, label="완벽 예측선 (예측 = 실측)")
        ax.plot(mid, med, color=INK, lw=2.2, marker="o", ms=6, zorder=5, label="실측 구간별 예측 중앙값")
        ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
        ax.set_xlabel("실측 변동성 (|로그수익률|×100)"); ax.set_ylabel("예측 변동성")
        ax.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.set_title(title, fontsize=15, color=INK, loc="left")
    out = int((v.realized > ZOOM).sum())
    axes[1].text(ZOOM * 0.98, ZOOM * 0.03, f"범위 밖(실측 > {ZOOM:g}) {out:,}점({out / len(v):.1%})은 왼쪽 그림에 표시",
                 ha="right", va="bottom", fontsize=11.5, color=INK2)
    axes[0].legend(loc="upper left", frameon=False, fontsize=12.5, markerscale=2.2)
    fig.suptitle("예측 vs 실측 — 실측이 작으면 위로, 크면 아래로: 예측이 평균 쪽으로 압축(분산 붕괴)",
                 fontsize=17, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, f"{PERIOD} · n={len(v):,} 전수(샘플링 없음) · 급변일 = |수익률| ≥ 2 × 직전 60일 평균 · 두 그림 모두 x·y 동일 스케일",
             fontsize=10.5, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    p = os.path.join(OUT, "fig2_pred_vs_actual.png"); fig.savefig(p, dpi=200); plt.close(fig); return p


def fig_std_ratio(v):
    ratio = float(v.hybrid.std() / v.realized.std())
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.barh([0], [1.0], color=GRID, height=0.42, zorder=1)
    ax.barh([0], [ratio], color=BLUE, height=0.42, zorder=2)
    ax.axvline(1.0, color=INK, lw=1.5, zorder=3)
    ax.text(ratio, 0.36, f"예측  {ratio:.0%}", ha="center", va="bottom", fontsize=15, color=INK, fontweight="bold")
    ax.text(1.0, 0.36, "실측  100%", ha="right", va="bottom", fontsize=13, color=INK2)
    ax.set_xlim(0, 1.05); ax.set_ylim(-0.5, 0.9); ax.set_yticks([])
    ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.set_title(f"예측의 변동폭은 실제 변동폭의 약 {ratio:.0%} 수준", fontsize=17, color=INK, pad=10, loc="left")
    fig.text(0.01, 0.02, f"표준편차 비율 = 예측 std {v.hybrid.std():.3f} ÷ 실측 std {v.realized.std():.3f} · {PERIOD}",
             fontsize=10.5, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    p = os.path.join(OUT, "fig3_std_ratio.png"); fig.savefig(p, dpi=200); plt.close(fig); return p, ratio


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--rows", required=True); a = ap.parse_args()
    v = pd.read_parquet(a.rows)
    v["spike"] = v["spike"].fillna(False).astype(bool)
    print(fig_rmse()); print(fig_scatter(v)); print(fig_std_ratio(v))
