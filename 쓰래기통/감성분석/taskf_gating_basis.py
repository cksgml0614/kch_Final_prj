# taskf_gating_basis.py
# Task F 추가 실험(2026-09-06) — 라벨 정의 basis 비교: 초과수익률(excess_return, 기존, 종목-KOSPI)
# vs 절대수익률(absolute_return, 신규, KOSPI 차감 없는 종목 순수 change_rate).
#
# 배경: excess_return은 시장 전체 움직임을 제거하는데, "시장 전체에 좋은 뉴스"(예: 반도체 업황
# 전반 호재)로 인한 신호까지 함께 지워버릴 수 있다는 우려가 있었다. absolute_return 기준
# z-score를 daily_labels에 추가 적재(라벨_공통.py/라벨_생성.py, label_basis 컬럼·PK 확장,
# 2026-09-06)하고, 이 스크립트가 두 basis를 나란히 게이팅+검증한다.
#
# 뉴스-라벨 매칭 규칙은 taskf_gating.py/taskf_gating_horizon.py와 완전히 동일하게 유지한다:
# D일 뉴스는 D보다 뒤인 첫 거래일 T에 매칭(merge_asof forward, allow_exact_matches=False).
#
# horizon: h=1(익일)과 h=3(2026-09-02 horizon 확장 실험에서 raw Δtest가 가장 컸던 조합—
# 방향 3-class/h=3, +0.1493 — 을 대표로 채택. 결과_TaskF_게이팅검증.md [3] 참고). 윈도우
# N=60 고정(D-2/결과_TaskE_라벨링.md에서 N=20/60/120 차이가 ±1%p 이내로 미미함을 확인 —
# taskf_gating_horizon.py와 동일한 축소 근거).
#
# 검증: 블록 셔플(h=1은 block_size=1로 사실상 일별 순열, h=3은 block_size=3 — 겹치는 누적
# 윈도우로 인한 인접 표본 상관을 보존한 채 텍스트-라벨 대응만 끊음) + 클래스 사전분포 무작위
# 예측기(200회). taskf_gating_horizon.py의 block_shuffle_labels/verdict을 그대로 재사용한다.
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
from 감성분석.taskf_gating_horizon import block_shuffle_labels, verdict
from 라벨.라벨_공통 import LABEL_BASES, load_labels, split_dates_by_ratio

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FIXED_WINDOW_N = 60
HORIZONS = [1, 3]

N_SHUFFLE = 30
N_RANDOM_PRIOR = 200

# 상위 피처 어휘 구성 진단용(요구사항 4) — 휴리스틱 키워드 목록. "판정"이 아니라 정성적 참고용이며,
# 목록에 없는 단어는 전부 "기타"로 분류된다. 삼성전자 고유(실적/제품/경영진) vs 업황·거시
# (반도체 산업 전반/금리/환율 등 시장 전체에 영향을 줄 수 있는 어휘) 두 축만 구분한다.
COMPANY_SPECIFIC_KEYWORDS = [
    "삼성전자", "삼성", "갤럭시", "언팩", "엑시노스", "파운드리", "이재용", "한종희", "경계현",
    "노태문", "잠정실적", "어닝", "영업이익", "자사주", "배당", "신제품", "출시", "양산",
    "갤럭시s", "폴더블", "가전", "반도체부문", "메모리사업부", "주주총회", "이사회",
]
SECTOR_MACRO_KEYWORDS = [
    "반도체", "메모리", "d램", "낸드", "hbm", "환율", "금리", "물가", "인플레이션", "연준",
    "금통위", "코스피", "나스닥", "필라델피아", "반도체지수", "수출", "무역", "관세",
    "엔비디아", "tsmc", "마이크론", "화웨이", "중국", "미국", "글로벌", "ai", "공급망",
]


def categorize_feature(word):
    w = word.lower()
    if any(k.lower() in w for k in COMPANY_SPECIFIC_KEYWORDS):
        return "삼성전자 고유"
    if any(k.lower() in w for k in SECTOR_MACRO_KEYWORDS):
        return "업황/거시"
    return "기타"


def load_news_with_basis_horizon_z(ticker, label_basis, window_n=FIXED_WINDOW_N, horizons=HORIZONS):
    """제목 텍스트 + (basis, horizon)별 z_score(wide)를 뉴스 (ticker,date) 단위로 결합한다.
    taskf_gating_horizon.load_news_with_horizon_z()와 동일 로직 — load_labels 호출에
    label_basis만 추가로 넘긴다."""
    news = load_labeled_news(ticker=ticker, table_name="daily_news", label_column=None, source=SOURCE)
    if news.empty:
        return pd.DataFrame(), None
    news = news[["ticker", "date", "title"]].copy()
    news["date"] = pd.to_datetime(news["date"])
    news = news.sort_values("date")

    wide = None
    for h in horizons:
        long_h = load_labels(ticker, horizon_h=h, label_basis=label_basis)
        if long_h.empty:
            raise RuntimeError(
                f"{ticker}: daily_labels에 horizon_h={h}, label_basis={label_basis!r} 데이터 없음 "
                "— 라벨_생성.py를 먼저 실행할 것"
            )
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


def run_combo(label_basis, scheme_name, horizon_h, merged):
    train_df, val_df, test_df, classes = prepare_combo(scheme_name, horizon_h, merged)

    vec = TfidfVectorizer(**VECTORIZER_KWARGS)
    X_train = vec.fit_transform(build_text(train_df))
    X_test = vec.transform(build_text(test_df))

    y_train_real = train_df["label"].values
    y_test = test_df["label"].values

    real_f1, real_acc = fit_predict_f1(X_train, y_train_real, X_test, y_test, classes)
    test_base_f1, test_base_acc, test_majority = majority_baseline(y_test, classes)
    real_delta = real_f1 - test_base_f1

    # ── 검증 1: 블록 셔플 (h=1은 block_size=1 — 사실상 일별 순열, h=3은 겹치는 누적윈도우 상관 보존) ──
    shuffle_f1s = []
    for i in range(N_SHUFFLE):
        rng = np.random.RandomState(1000 + i)
        shuffled_map = block_shuffle_labels(train_df, block_size=horizon_h, rng=rng)
        y_shuffled = train_df["trading_date"].map(shuffled_map).values
        f1, _ = fit_predict_f1(X_train, y_shuffled, X_test, y_test, classes)
        shuffle_f1s.append(f1)
    shuffle_f1s = np.array(shuffle_f1s)
    shuffle_deltas = shuffle_f1s - test_base_f1
    shuffle_mean, shuffle_std = shuffle_deltas.mean(), shuffle_deltas.std()
    ci_lo, ci_hi = np.percentile(shuffle_deltas, [2.5, 97.5])
    p_value = (np.sum(shuffle_f1s >= real_f1) + 1) / (N_SHUFFLE + 1)

    # ── 검증 2: 클래스 사전분포 무작위 예측기 ──
    train_counts = pd.Series(y_train_real).value_counts().reindex(classes, fill_value=0)
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
        "label_basis": label_basis, "scheme": scheme_name, "horizon_h": horizon_h,
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "real_f1": real_f1, "real_acc": real_acc, "real_delta": real_delta,
        "test_base_f1": test_base_f1, "test_base_acc": test_base_acc, "test_majority": test_majority,
        "shuffle_mean": shuffle_mean, "shuffle_std": shuffle_std, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "p_value": p_value,
        "prior_f1_mean": prior_f1s.mean(), "prior_f1_std": prior_f1s.std(),
        "min_test_day_count": min_test_day_count,
    }


def print_top_features_composition(label_basis, scheme_name, horizon_h, merged, top_n=20):
    """유의한 조합에 한해: 상위 피처(word 1-2gram)를 삼성전자 고유 vs 업황/거시 vs 기타로
    휴리스틱 분류해 비율을 보고한다(요구사항 4). 판정이 아니라 정성적 참고 지표."""
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
    print(f"\n--- 상위 피처 구성비: [{label_basis} / {scheme_name} / h={horizon_h}] word 1-2gram, 클래스당 상위 {top_n}개 ---")
    all_counts = {"삼성전자 고유": 0, "업황/거시": 0, "기타": 0}
    for i, cls in enumerate(clf.classes_):
        coefs = clf.coef_[i]
        top_idx = np.argsort(coefs)[::-1][:top_n]
        cats = [categorize_feature(feature_names[j]) for j in top_idx]
        for c in cats:
            all_counts[c] += 1
        print(f"  [label={int(cls):+d}] " + ", ".join(
            f"{feature_names[j]}({coefs[j]:.2f})[{cat}]" for j, cat in zip(top_idx, cats)
        ))
    total = sum(all_counts.values())
    print(
        f"  구성비(전체 {total}개, 클래스 {len(clf.classes_)}개 x 상위 {top_n}): "
        f"삼성전자 고유 {all_counts['삼성전자 고유']}개({all_counts['삼성전자 고유']/total:.0%}) / "
        f"업황·거시 {all_counts['업황/거시']}개({all_counts['업황/거시']/total:.0%}) / "
        f"기타 {all_counts['기타']}개({all_counts['기타']/total:.0%})"
    )


def main():
    for ticker in ACTIVE_TICKERS:
        print("\n" + "#" * 100)
        print(f"# Task F basis 비교 게이팅+검증 — {ticker}, window_n={FIXED_WINDOW_N} 고정, "
              f"basis={LABEL_BASES}, horizon={HORIZONS}")
        print("#" * 100)

        results = []
        for basis in LABEL_BASES:
            merged, split_dates = load_news_with_basis_horizon_z(ticker, basis)
            if merged.empty:
                print(f"⚠️ {ticker}/{basis}: 조인된 뉴스 없음 — 스킵")
                continue
            train_dates, val_dates, test_dates = split_dates
            print(
                f"\n[{basis}] 거래일 기준 split: train {len(train_dates)}일 / val {len(val_dates)}일 / "
                f"test {len(test_dates)}일 ({test_dates.min().date()}~{test_dates.max().date()})"
            )

            for scheme_name in LABEL_SCHEMES:
                for h in HORIZONS:
                    print(f"  [{basis} / {scheme_name} / h={h}] 학습+검증 중...")
                    r = run_combo(basis, scheme_name, h, merged)
                    r["_merged"] = merged
                    results.append(r)
                    print(
                        f"    n(train/val/test)={r['n_train']}/{r['n_val']}/{r['n_test']}  "
                        f"test F1={r['real_f1']:.4f} (baseline {r['test_base_f1']:.4f}, Δ{r['real_delta']:+.4f})  "
                        f"셔플p={r['p_value']:.4f}  사전분포F1={r['prior_f1_mean']:.4f}±{r['prior_f1_std']:.4f}  "
                        f"최소클래스거래일={r['min_test_day_count']}"
                    )

        if not results:
            print(f"⚠️ {ticker}: 비교 대상 없음 — 스킵")
            continue

        print("\n" + "=" * 130)
        print("=== 종합 비교표: excess_return vs absolute_return, test macro F1 기준 Δtest 내림차순 ===")
        print("=" * 130)
        header = (f"{'basis':<16}{'라벨':<16}{'h':>3}{'testF1':>9}{'base':>8}{'Δtest':>9}"
                  f"{'셔플p':>8}{'사전F1':>9}{'판정':>18}")
        print(header)
        ranked = sorted(results, key=lambda x: x["real_delta"], reverse=True)
        for r in ranked:
            v, _ = verdict(r)
            print(
                f"{r['label_basis']:<16}{r['scheme']:<16}{r['horizon_h']:>3}{r['real_f1']:>9.4f}"
                f"{r['test_base_f1']:>8.4f}{r['real_delta']:>9.4f}{r['p_value']:>8.4f}"
                f"{r['prior_f1_mean']:>9.4f}{v:>18}"
            )

        print("\n=== basis별(excess_return vs absolute_return) 직접 대응 비교 (같은 라벨x h) ===")
        for scheme_name in LABEL_SCHEMES:
            for h in HORIZONS:
                pair = {r["label_basis"]: r for r in results if r["scheme"] == scheme_name and r["horizon_h"] == h}
                if len(pair) != 2:
                    continue
                er, ar = pair["excess_return"], pair["absolute_return"]
                print(
                    f"  [{scheme_name} / h={h}] excess_return Δ{er['real_delta']:+.4f}(p={er['p_value']:.3f}) "
                    f"vs absolute_return Δ{ar['real_delta']:+.4f}(p={ar['p_value']:.3f})"
                )

        print("\n=== 종합 판정 ===")
        any_significant = False
        for r in ranked:
            v, reason = verdict(r)
            print(f"  [{r['label_basis']} / {r['scheme']} / h={r['horizon_h']}] {v} — {reason}")
            if v == "유의함":
                any_significant = True

        if any_significant:
            print("\n유의한 조합의 상위 피처 구성비(삼성전자 고유 vs 업황/거시 vs 기타):")
            for r in ranked:
                v, _ = verdict(r)
                if v == "유의함":
                    print_top_features_composition(r["label_basis"], r["scheme"], r["horizon_h"], r["_merged"])
        else:
            print("\n유의한 조합 없음 — 상위 피처 구성비 점검 생략"
                  "(신호가 확인되지 않은 조합의 피처를 사후에 들여다보는 것은 해석 남용 위험).")


if __name__ == "__main__":
    main()
