# taskf_gating_horizon.py
# Task F 확장 — 익일(h=1) 대신 h거래일(3/5/10) 누적 초과수익률 라벨로 게이팅 + 검증을 함께
# 수행한다. 익일 예측은 노이즈 대비 신호가 너무 약할 수 있다는 판단(2026-09-02)에 따른 재설계.
#
# 조합: 라벨 3종(방향 5-class/3-class, 변동성 2-class) x horizon 3종(3/5/10) = 9개.
# 윈도우 N은 60으로 고정 — 결과_TaskE_라벨링.md에서 N=20/60/120 차이가 ±1%p 이내로 미미함을
# 이미 확인했고(CLAUDE.md D-2), 여기에 horizon 축까지 더하면 27개는 개별 검증(셔플+사전분포
# 무작위)까지 함께 돌리기엔 과함 — D-2가 "기본"으로 제시했던 N=60 하나로 좁혔다.
#
# 뉴스-라벨 매칭 규칙은 taskf_gating.py(2026-09-01 수정판)와 완전히 동일하다: D일 뉴스는
# D보다 뒤인 첫 거래일 T에 매칭한다(merge_asof forward, allow_exact_matches=False). 달라지는
# 것은 T에 저장된 라벨의 의미뿐이다 — "T 하루의 수익률"이 아니라 "T부터 h거래일 누적 수익률".
#
# 검증(taskf_validate.py 경험 반영, 2026-09-01 게이팅에서 셔플과 구분 안 됐던 결과 이후):
#   - 레이블 셔플 테스트를 "블록 셔플"로 한다 — 누적 수익률은 겹치는 윈도우라 인접 거래일의
#     라벨이 독립이 아니다. 개별 행을 무작위로 섞으면(순수 무작위 셔플) 이 상관구조가 깨져
#     검증이 실제보다 느슨해질 수 있다. 대신 거래일을 h일 단위 블록으로 묶고, 블록 "순서"만
#     섞는다 — 블록 내부의 날짜-라벨 상관구조는 보존한 채 텍스트-라벨 대응만 끊는다.
#   - 클래스 사전분포를 따르는 무작위 예측기와도 비교한다 — 지난 검증에서 이게 결정적이었다
#     (다수결 baseline만 이기는 건 클래스 불균형 때문일 뿐 신호가 아닐 수 있음).
#
# 뉴스 테이블에는 아무것도 쓰지 않는다 — 전부 조회 전용.

import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from constants import ACTIVE_TICKERS, TRAIN_PERIOD_END, TRAIN_PERIOD_START
from 감성분석.kobert_dataset import load_labeled_news
from 감성분석.taskf_gating import LABEL_SCHEMES, SOURCE, VECTORIZER_KWARGS, build_text, majority_baseline
from 라벨.라벨_공통 import load_labels, split_dates_by_ratio

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FIXED_WINDOW_N = 60
HORIZONS = [3, 5, 10]

N_SHUFFLE = 30
N_RANDOM_PRIOR = 200
SIG_ALPHA = 0.05


def load_news_with_horizon_z(ticker, window_n=FIXED_WINDOW_N, horizons=HORIZONS):
    """제목 텍스트 + horizon별 z_score(wide)를 뉴스 (ticker,date) 단위로 결합한다.
    매칭 로직은 taskf_gating.py의 load_news_with_z()와 동일 — 차이는 daily_labels 조회 시
    horizon_h를 지정한다는 점뿐."""
    news = load_labeled_news(ticker=ticker, table_name="daily_news", label_column=None, source=SOURCE)
    if news.empty:
        return pd.DataFrame(), None
    news = news[["ticker", "date", "title"]].copy()
    news["date"] = pd.to_datetime(news["date"])
    news = news.sort_values("date")

    wide = None
    for h in horizons:
        long_h = load_labels(ticker, horizon_h=h)
        if long_h.empty:
            raise RuntimeError(f"{ticker}: daily_labels에 horizon_h={h} 데이터 없음 — 라벨_생성.py를 먼저 실행할 것")
        sub = long_h.loc[long_h["window_n"] == window_n, ["date", "z_score"]].rename(columns={"z_score": f"z_h{h}"})
        wide = sub if wide is None else wide.merge(sub, on="date", how="outer")
    wide = wide.sort_values("date").reset_index(drop=True)

    # D일 뉴스 -> D보다 뒤인 첫 거래일(taskf_gating.py와 동일 규칙, allow_exact_matches=False)
    trading_calendar = pd.DataFrame({"trading_date": sorted(wide["date"].unique())})
    merged = pd.merge_asof(
        news, trading_calendar, left_on="date", right_on="trading_date",
        direction="forward", allow_exact_matches=False,
    )
    merged = merged.dropna(subset=["trading_date"])

    tp_start, tp_end = pd.Timestamp(TRAIN_PERIOD_START), pd.Timestamp(TRAIN_PERIOD_END)
    merged = merged[(merged["trading_date"] >= tp_start) & (merged["trading_date"] <= tp_end)]
    merged = merged.merge(wide.rename(columns={"date": "trading_date"}), on="trading_date", how="left")

    tp_dates = wide.loc[(wide["date"] >= tp_start) & (wide["date"] <= tp_end), "date"]
    train_dates, val_dates, test_dates = split_dates_by_ratio(tp_dates)
    train_set, val_set, test_set = set(train_dates), set(val_dates), set(test_dates)

    def assign_split(d):
        if d in train_set:
            return "train"
        if d in val_set:
            return "val"
        if d in test_set:
            return "test"
        return None

    merged["split"] = merged["trading_date"].apply(assign_split)
    merged = merged.dropna(subset=["split"])
    return merged, (train_dates, val_dates, test_dates)


def prepare_combo(scheme_name, horizon_h, merged):
    scheme = LABEL_SCHEMES[scheme_name]
    z_col = f"z_h{horizon_h}"
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


def block_shuffle_labels(train_df, block_size, rng):
    """train_df의 (trading_date -> label) 매핑을 블록 단위로 셔플한다. 연속된 거래일
    block_size개를 한 블록으로 묶어 블록 "순서"만 뒤섞고, 블록 내부는 원래 순서를 유지한다
    — 인접 거래일 라벨 간 상관구조(겹치는 horizon 윈도우 때문에 생김)를 보존한 채 텍스트-라벨
    대응만 끊는다. 반환: {trading_date: shuffled_label} 딕셔너리."""
    date_label = (
        train_df.drop_duplicates(subset=["trading_date"])
        .sort_values("trading_date")[["trading_date", "label"]]
    )
    dates = date_label["trading_date"].tolist()
    labels = date_label["label"].tolist()
    n = len(dates)

    blocks = [labels[i:i + block_size] for i in range(0, n, block_size)]
    order = rng.permutation(len(blocks))
    shuffled_labels = np.concatenate([blocks[i] for i in order])
    return dict(zip(dates, shuffled_labels))


def run_combo(scheme_name, horizon_h, merged):
    train_df, val_df, test_df, classes = prepare_combo(scheme_name, horizon_h, merged)

    vec = TfidfVectorizer(**VECTORIZER_KWARGS)
    X_train = vec.fit_transform(build_text(train_df))
    X_val = vec.transform(build_text(val_df))
    X_test = vec.transform(build_text(test_df))

    real_f1, real_acc = fit_predict_f1(X_train, train_df["label"].values, X_test, test_df["label"].values, classes)
    val_f1, val_acc = fit_predict_f1(X_train, train_df["label"].values, X_val, val_df["label"].values, classes)

    test_base_f1, test_base_acc, test_majority = majority_baseline(test_df["label"].values, classes)
    val_base_f1, val_base_acc, _ = majority_baseline(val_df["label"].values, classes)

    # ── 검증 1: 블록 셔플 (train 라벨을 h일 블록 단위로 섞음) ──
    y_test = test_df["label"].values
    shuffle_f1s = []
    for i in range(N_SHUFFLE):
        rng = np.random.RandomState(1000 + i)
        shuffled_map = block_shuffle_labels(train_df, block_size=horizon_h, rng=rng)
        y_shuffled = train_df["trading_date"].map(shuffled_map).values
        f1, _ = fit_predict_f1(X_train, y_shuffled, X_test, y_test, classes)
        shuffle_f1s.append(f1)
    shuffle_f1s = np.array(shuffle_f1s)
    shuffle_deltas = shuffle_f1s - test_base_f1
    real_delta = real_f1 - test_base_f1
    shuffle_mean, shuffle_std = shuffle_deltas.mean(), shuffle_deltas.std()
    ci_lo, ci_hi = np.percentile(shuffle_deltas, [2.5, 97.5])
    p_value = (np.sum(shuffle_f1s >= real_f1) + 1) / (N_SHUFFLE + 1)

    # ── 검증 2: 클래스 사전분포 무작위 예측기 ──
    train_counts = pd.Series(train_df["label"].values).value_counts().reindex(classes, fill_value=0)
    prior = (train_counts / train_counts.sum()).values
    prior_f1s = []
    for i in range(N_RANDOM_PRIOR):
        rng = np.random.RandomState(5000 + i)
        pred = rng.choice(classes, size=len(y_test), p=prior)
        prior_f1s.append(f1_score(y_test, pred, average="macro", labels=classes, zero_division=0))
    prior_f1s = np.array(prior_f1s)

    # ── 표본 크기 (test) ──
    test_day_counts = (
        merged[merged["split"] == "test"]
        .drop_duplicates(subset=["trading_date"])[f"z_h{horizon_h}"]
        .apply(lambda v: LABEL_SCHEMES[scheme_name]["fn"](v, **LABEL_SCHEMES[scheme_name]["kwargs"]))
        .value_counts().reindex(classes, fill_value=0)
    )
    min_test_day_count = int(test_day_counts.min())

    return {
        "scheme": scheme_name, "horizon_h": horizon_h,
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "real_f1": real_f1, "real_acc": real_acc, "real_delta": real_delta,
        "val_f1": val_f1, "val_base_f1": val_base_f1,
        "test_base_f1": test_base_f1, "test_base_acc": test_base_acc, "test_majority": test_majority,
        "shuffle_mean": shuffle_mean, "shuffle_std": shuffle_std, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "p_value": p_value,
        "prior_f1_mean": prior_f1s.mean(), "prior_f1_std": prior_f1s.std(),
        "min_test_day_count": min_test_day_count,
        "vec": vec, "clf_X_train": X_train, "train_df": train_df,
    }


def verdict(r):
    if r["min_test_day_count"] < 10:
        return "판정 불가(표본 부족)", f"test 최소 클래스 거래일 수 {r['min_test_day_count']}개 < 10"
    beats_shuffle = r["p_value"] < SIG_ALPHA and r["real_delta"] > r["ci_hi"]
    beats_prior = r["real_f1"] > r["prior_f1_mean"] + 2 * r["prior_f1_std"]
    if beats_shuffle and beats_prior:
        return "유의함", f"p={r['p_value']:.4f}<0.05·실제Δ가 셔플95%상단 초과, 사전분포 무작위 예측기(+2σ)도 상회"
    if not beats_shuffle:
        return "유의하지 않음", f"p={r['p_value']:.4f}, 실제 Δ가 블록셔플 분포와 구분 안 됨"
    return "유의하지 않음", "블록셔플은 통과했으나 사전분포 무작위 예측기와 구분 안 됨"


def print_top_features(scheme_name, horizon_h, merged, top_n=12):
    scheme = LABEL_SCHEMES[scheme_name]
    z_col = f"z_h{horizon_h}"
    df = merged.copy()
    df["label"] = df[z_col].apply(lambda v: scheme["fn"](v, **scheme["kwargs"]))
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    train_df = df[df["split"] == "train"]

    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3)
    X_train = vec.fit_transform(build_text(train_df))
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X_train, train_df["label"])

    feature_names = np.array(vec.get_feature_names_out())
    print(f"\n--- 상위 피처: [{scheme_name} / h={horizon_h}] word 1-2gram (진단 전용) ---")
    for i, cls in enumerate(clf.classes_):
        coefs = clf.coef_[i]
        top_idx = np.argsort(coefs)[::-1][:top_n]
        print(f"  [label={int(cls):+d}] " + ", ".join(f"{feature_names[j]}({coefs[j]:.2f})" for j in top_idx))


def main():
    for ticker in ACTIVE_TICKERS:
        print("\n" + "#" * 96)
        print(f"# Task F horizon 게이팅+검증 — {ticker}, window_n={FIXED_WINDOW_N} 고정, horizon={HORIZONS}")
        print("#" * 96)

        merged, split_dates = load_news_with_horizon_z(ticker)
        if merged.empty:
            print(f"⚠️ {ticker}: 조인된 뉴스 없음 — 스킵")
            continue
        train_dates, val_dates, test_dates = split_dates
        print(
            f"거래일 기준 split: train {len(train_dates)}일({train_dates.min().date()}~{train_dates.max().date()}) / "
            f"val {len(val_dates)}일({val_dates.min().date()}~{val_dates.max().date()}) / "
            f"test {len(test_dates)}일({test_dates.min().date()}~{test_dates.max().date()})"
        )

        results = []
        for scheme_name in LABEL_SCHEMES:
            for h in HORIZONS:
                print(f"\n[{scheme_name} / h={h}] 학습+검증 중...")
                r = run_combo(scheme_name, h, merged)
                results.append(r)
                print(
                    f"  n(train/val/test)={r['n_train']}/{r['n_val']}/{r['n_test']}  "
                    f"test macro F1={r['real_f1']:.4f} (baseline {r['test_base_f1']:.4f}, Δ{r['real_delta']:+.4f})  "
                    f"accuracy={r['real_acc']:.4f}"
                )
                print(
                    f"  [검증] 블록셔플({N_SHUFFLE}회) Δ평균={r['shuffle_mean']:+.4f}±{r['shuffle_std']:.4f} "
                    f"95%=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] p={r['p_value']:.4f}  |  "
                    f"사전분포 무작위({N_RANDOM_PRIOR}회) F1={r['prior_f1_mean']:.4f}±{r['prior_f1_std']:.4f}  |  "
                    f"test 최소클래스 거래일={r['min_test_day_count']}"
                )

        print("\n" + "=" * 110)
        print("=== 종합 요약: 조합 9개, test macro F1 기준 baseline 대비 차이 내림차순 ===")
        print("=" * 110)
        header = (f"{'라벨':<16}{'h':>4}{'testF1':>9}{'base':>8}{'Δtest':>9}"
                  f"{'셔플p':>8}{'사전F1':>9}{'판정':>18}")
        print(header)
        ranked = sorted(results, key=lambda x: x["real_delta"], reverse=True)
        for r in ranked:
            v, _ = verdict(r)
            print(
                f"{r['scheme']:<16}{r['horizon_h']:>4}{r['real_f1']:>9.4f}{r['test_base_f1']:>8.4f}"
                f"{r['real_delta']:>9.4f}{r['p_value']:>8.4f}{r['prior_f1_mean']:>9.4f}{v:>18}"
            )

        print("\n=== horizon(3/5/10)별 경향 — 라벨 스킴별 평균 Δtest ===")
        for scheme_name in LABEL_SCHEMES:
            row = [r for r in results if r["scheme"] == scheme_name]
            row = sorted(row, key=lambda r: r["horizon_h"])
            print("  " + scheme_name + ": " + ", ".join(f"h={r['horizon_h']}:Δ{r['real_delta']:+.4f}(p={r['p_value']:.3f})" for r in row))

        print("\n=== 종합 판정 ===")
        any_significant = False
        for r in ranked:
            v, reason = verdict(r)
            print(f"  [{r['scheme']} / h={r['horizon_h']}] {v} — {reason}")
            if v == "유의함":
                any_significant = True

        if any_significant:
            print("\n유의한 조합의 상위 피처(사후 서술 어휘 여부 확인):")
            for r in ranked:
                v, _ = verdict(r)
                if v == "유의함":
                    print_top_features(r["scheme"], r["horizon_h"], merged)
        else:
            print("\n유의한 조합 없음 — 상위 피처 점검 생략(피처를 볼 필요가 없을 만큼 신호가 확인되지 않음).")


if __name__ == "__main__":
    main()
