# taskf_validate.py
# Task F 게이팅(taskf_gating.py) 결과가 실제 신호인지, 평가 방식의 artifact인지 검증한다.
# Task T-1의 seed_stability_check.py/pr_curve_vs_random.py와 같은 성격 — "단일 실행 결과를
# 그대로 믿어도 되는가"를 묻는 진단이며, 결과가 부정적이어도(무작위와 구분 안 됨) 그대로
# 판정한다. KR-FinBERT 투입 전 검증이 목적이라 여기서 KoBERT/KR-FinBERT를 학습하지 않는다.
#
# 대상: D+1 거래일 매칭(taskf_gating.py 2026-09-01 수정판) 기준 각 라벨 유형 최고 조합 3개
#   방향 3-class/N=60, 변동성 2-class/N=120, 방향 5-class/N=60
#
# 진단 4종:
#   1) 레이블 셔플 테스트 — train 라벨만 무작위로 섞고(날짜-뉴스 대응 유지) 동일 파이프라인
#      재학습, 셔플 30회의 test macro F1 분포와 실제 값을 비교 + 경험적 p-value
#   2) "시드" 안정성 — ⚠️ LogisticRegression(solver='lbfgs', 기본값)은 결정론적이라 random_state를
#      바꿔도 fit 결과가 완전히 동일하다(실측 확인, 아래 참고). 그래서 "동일 설정, 무작위성만
#      다른 여러 실행"의 실질적 대응물로 train 셋 부트스트랩 재추출(복원추출, 크기 동일)을 쓴다
#      — 훈련 표본이 조금 달라져도 결과가 크게 흔들리는지 보는 것이 이 진단의 실제 목적이다
#   3) 클래스 사전분포를 따르는 무작위 예측기 — 클래스 수가 다른 스킴(2/3/5-class)끼리 macro F1
#      절대값을 비교하려면 "무작위 기대값"이 스킴마다 다르다는 걸 감안해야 한다
#   4) 표본 크기 — split별 클래스당 거래일 수 / 뉴스 건수

import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from constants import ACTIVE_TICKERS
from 감성분석.taskf_gating import LABEL_SCHEMES, VECTORIZER_KWARGS, build_text, load_news_with_z, majority_baseline

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TARGET_COMBOS = [
    ("방향 3-class", 60),
    ("변동성 2-class", 120),
    ("방향 5-class", 60),
]

N_SHUFFLE = 30       # 최소 20회 요청 — 여유를 둬 백분위 추정을 더 안정화
N_BOOTSTRAP = 10      # 최소 7회 요청 — 부트스트랩은 계산비용이 낮아 조금 더 돌림
N_RANDOM_PRIOR = 200  # 사전분포 무작위 예측기 기대값 추정(요청엔 횟수 지정 없음, 안정적 추정 위해 넉넉히)

SIG_ALPHA = 0.05


def prepare_combo(scheme_name, window_n, merged):
    """merged(load_news_with_z 결과)에서 해당 조합의 라벨을 파생하고 train/val/test로 나눈다."""
    scheme = LABEL_SCHEMES[scheme_name]
    z_col = f"z_{window_n}"
    df = merged.copy()
    df["label"] = df[z_col].apply(lambda v: scheme["fn"](v, **scheme["kwargs"]))
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df, scheme["classes"]


def fit_predict_f1(X_train, y_train, X_test, y_test, classes):
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    f1 = f1_score(y_test, pred, average="macro", labels=classes, zero_division=0)
    acc = accuracy_score(y_test, pred)
    return f1, acc


def run_validation(scheme_name, window_n, merged):
    print("\n" + "#" * 92)
    print(f"# 검증 대상: [{scheme_name} / N={window_n}]")
    print("#" * 92)

    train_df, val_df, test_df, classes = prepare_combo(scheme_name, window_n, merged)

    vec = TfidfVectorizer(**VECTORIZER_KWARGS)
    X_train = vec.fit_transform(build_text(train_df))
    X_test = vec.transform(build_text(test_df))
    y_train_real = train_df["label"].values
    y_test = test_df["label"].values

    test_base_f1, test_base_acc, test_majority = majority_baseline(y_test, classes)

    # 실제(real label) 결과 — taskf_gating.py와 동일 설정으로 재확인
    real_f1, real_acc = fit_predict_f1(X_train, y_train_real, X_test, y_test, classes)
    real_delta = real_f1 - test_base_f1
    print(f"\n실제 결과: test macro F1={real_f1:.4f} (baseline {test_base_f1:.4f}, Δ{real_delta:+.4f}) accuracy={real_acc:.4f}")
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)} (뉴스 건수 기준)")

    # ── 1) 레이블 셔플 테스트 ──────────────────────────────────────────
    shuffle_f1s = []
    for i in range(N_SHUFFLE):
        rng = np.random.RandomState(1000 + i)
        y_shuffled = y_train_real.copy()
        rng.shuffle(y_shuffled)  # train 내에서만 순서 섞음 — 날짜/텍스트 행 위치는 그대로, 라벨만 재배치
        f1, _ = fit_predict_f1(X_train, y_shuffled, X_test, y_test, classes)
        shuffle_f1s.append(f1)
    shuffle_f1s = np.array(shuffle_f1s)
    shuffle_deltas = shuffle_f1s - test_base_f1
    shuffle_mean, shuffle_std = shuffle_deltas.mean(), shuffle_deltas.std()
    ci_lo, ci_hi = np.percentile(shuffle_deltas, [2.5, 97.5])
    # 경험적 단측 p-value: 셔플 결과 중 실제 값 이상이 나온 비율 (add-one 보정)
    p_value = (np.sum(shuffle_f1s >= real_f1) + 1) / (N_SHUFFLE + 1)
    z_like = (real_delta - shuffle_mean) / shuffle_std if shuffle_std > 0 else float("inf")

    print(f"\n[1] 레이블 셔플 테스트 (train 라벨만 {N_SHUFFLE}회 무작위 순열)")
    print(f"  셔플 Δtest: 평균={shuffle_mean:+.4f}  표준편차={shuffle_std:.4f}  95% 구간=[{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  실제 Δtest={real_delta:+.4f} → 셔플 분포 대비 {z_like:+.2f}σ, 95% 구간 상단 초과 여부: {real_delta > ci_hi}")
    print(f"  경험적 p-value (셔플 f1 >= 실제 f1 비율): {p_value:.4f} "
          f"({'유의(p<0.05)' if p_value < SIG_ALPHA else '유의 아님(p>=0.05)'})")

    # ── 2) "시드" 안정성 — lbfgs 결정론 확인 + 부트스트랩 대체 ───────────
    seed_f1s = []
    for seed in [42, 0, 1, 7, 123, 2024, 777]:
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
        clf.fit(X_train, y_train_real)
        pred = clf.predict(X_test)
        seed_f1s.append(f1_score(y_test, pred, average="macro", labels=classes, zero_division=0))
    seed_f1s = np.array(seed_f1s)
    print(f"\n[2a] random_state만 변경(7회, solver=lbfgs 기본값): "
          f"test F1 표준편차={seed_f1s.std():.6f} "
          f"({'완전 동일(결정론적) — random_state는 이 파이프라인에 영향 없음' if seed_f1s.std() == 0 else '미세한 차이 있음'})")

    boot_f1s = []
    n_train = X_train.shape[0]
    for b in range(N_BOOTSTRAP):
        rng = np.random.RandomState(2000 + b)
        idx = rng.randint(0, n_train, size=n_train)  # 복원추출, 크기 동일
        f1, _ = fit_predict_f1(X_train[idx], y_train_real[idx], X_test, y_test, classes)
        boot_f1s.append(f1)
    boot_f1s = np.array(boot_f1s)
    boot_deltas = boot_f1s - test_base_f1
    print(f"[2b] train 부트스트랩 재추출 {N_BOOTSTRAP}회(실제 대체 지표): "
          f"Δtest 평균={boot_deltas.mean():+.4f} 표준편차={boot_deltas.std():.4f} "
          f"최소={boot_deltas.min():+.4f} 최대={boot_deltas.max():+.4f}")
    unstable = boot_deltas.std() >= abs(real_delta) * 0.5
    print(f"  판정: 표준편차가 Δ 자체의 절반 이상인가? {unstable} "
          f"({'불안정 — 단일 실행 신뢰 어려움' if unstable else '상대적으로 안정'})")

    # ── 3) 클래스 사전분포 무작위 예측기 ──────────────────────────────
    train_counts = pd.Series(y_train_real).value_counts().reindex(classes, fill_value=0)
    prior = (train_counts / train_counts.sum()).values
    prior_f1s = []
    for i in range(N_RANDOM_PRIOR):
        rng = np.random.RandomState(5000 + i)
        pred = rng.choice(classes, size=len(y_test), p=prior)
        prior_f1s.append(f1_score(y_test, pred, average="macro", labels=classes, zero_division=0))
    prior_f1s = np.array(prior_f1s)
    print(f"\n[3] 클래스 사전분포 무작위 예측기 ({N_RANDOM_PRIOR}회, train 클래스 비율 {np.round(prior, 3).tolist()})")
    print(f"  기대 macro F1 = {prior_f1s.mean():.4f} ± {prior_f1s.std():.4f} "
          f"(다수결 baseline {test_base_f1:.4f}, 실제 모델 {real_f1:.4f})")

    # ── 4) 표본 크기 ──────────────────────────────────────────────
    print("\n[4] 표본 크기 (test split)")
    trading_day_counts = (
        merged[merged["split"] == "test"]
        .drop_duplicates(subset=["trading_date"])[f"z_{window_n}"]
        .apply(lambda v: LABEL_SCHEMES[scheme_name]["fn"](v, **LABEL_SCHEMES[scheme_name]["kwargs"]))
        .value_counts().reindex(classes, fill_value=0)
    )
    news_counts = test_df["label"].value_counts().reindex(classes, fill_value=0)
    print(f"  {'클래스':>8}{'거래일 수':>12}{'뉴스 건수':>12}")
    for cls in classes:
        print(f"  {cls:>8}{int(trading_day_counts[cls]):>12}{int(news_counts[cls]):>12}")

    return {
        "scheme": scheme_name, "window_n": window_n,
        "real_f1": real_f1, "real_delta": real_delta, "test_base_f1": test_base_f1,
        "shuffle_mean": shuffle_mean, "shuffle_std": shuffle_std, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "p_value": p_value, "z_like": z_like,
        "boot_std": boot_deltas.std(), "unstable": unstable,
        "prior_f1_mean": prior_f1s.mean(), "prior_f1_std": prior_f1s.std(),
        "min_test_day_count": int(trading_day_counts.min()),
    }


def verdict(r):
    """종합 판정: 유의함 / 판정 불가(표본 부족) / 유의하지 않음"""
    if r["min_test_day_count"] < 10:
        return "판정 불가(표본 부족)", f"test 최소 클래스 거래일 수 {r['min_test_day_count']}개 < 10"
    if r["p_value"] < SIG_ALPHA and r["real_delta"] > r["ci_hi"] and not r["unstable"]:
        return "유의함", (
            f"p={r['p_value']:.4f}<0.05, 실제 Δ가 셔플 95% 구간 상단 초과, "
            f"부트스트랩 변동폭이 Δ의 절반 미만(안정)"
        )
    if r["p_value"] >= SIG_ALPHA or r["real_delta"] <= r["ci_hi"]:
        return "유의하지 않음", f"p={r['p_value']:.4f}, 실제 Δ가 셔플 분포와 구분 안 됨"
    if r["unstable"]:
        return "판정 불가(불안정)", "셔플 검정은 통과했으나 부트스트랩 변동폭이 Δ와 비슷한 크기"
    return "판정 불가", "명확한 기준에 걸리지 않음 — 개별 수치 재검토 필요"


def main():
    for ticker in ACTIVE_TICKERS:
        merged, _ = load_news_with_z(ticker)
        if merged.empty:
            print(f"⚠️ {ticker}: 조인된 뉴스 없음 — 스킵")
            continue

        results = []
        for scheme_name, window_n in TARGET_COMBOS:
            r = run_validation(scheme_name, window_n, merged)
            results.append(r)

        print("\n" + "=" * 100)
        print("=== 종합 판정 ===")
        print("=" * 100)
        header = f"{'라벨':<16}{'N':>5}{'실제Δ':>9}{'셔플p':>8}{'셔플95%상단':>12}{'부트std':>9}{'판정':>18}"
        print(header)
        for r in results:
            v, reason = verdict(r)
            print(
                f"{r['scheme']:<16}{r['window_n']:>5}{r['real_delta']:>9.4f}{r['p_value']:>8.4f}"
                f"{r['ci_hi']:>12.4f}{r['boot_std']:>9.4f}{v:>18}"
            )
            print(f"    근거: {reason}")


if __name__ == "__main__":
    main()
