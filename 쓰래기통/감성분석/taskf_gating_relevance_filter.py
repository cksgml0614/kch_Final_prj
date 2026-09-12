# taskf_gating_relevance_filter.py
# Task F 추가 실험(2026-09-06) — 주가 민감 키워드 사후 필터. daily_news(search_backfill)를
# 재크롤링하지 않고, 기존에 이미 있는 기사에 제목 키워드 필터를 사후 적용해 "관련성 높음"
# 부분집합만으로 게이팅을 다시 돌린다. 원본 daily_news 테이블은 건드리지 않는다 — 필터는
# 이 스크립트 안에서 title 컬럼에 정규식으로 즉석 적용하는 런타임 플래그(is_relevant)일 뿐,
# DB에 컬럼/뷰를 추가하지 않는다(요구사항 "원본 데이터는 건드리지 마"를 가장 단순하게 만족).
#
# 배경: 기존 필터(제목에 종목명 포함 + 언론사 화이트리스트)는 실적/공시 같은 사건성 기사와
# 제품 홍보·사회공헌 등 무관한 기사가 섞여 있다. 관련성을 더 좁히면 노이즈가 줄어 신호가
# 드러나는지 확인하는 것이 목적이다.
#
# 키워드 선정 원칙(사람 지정, 2026-09-06): "미래 주가에 영향을 줄 사건"만 포함한다.
# "상한가/하한가/급등/급락"처럼 이미 일어난 가격 변동 자체를 서술하는 단어는 제외 —
# D+1 매칭 규칙을 지켜도 이런 단어는 사실상 "그날 있었던 결과"를 사후 서술하는 것이라
# 필터 취지(예측 신호 발굴)와 맞지 않고 오히려 착시를 만들 수 있다. "실적/공시/배당/자사주"도
# 사건인지 결과 서술인지 애매해 이번엔 제외했다.
#
# 뉴스-라벨 매칭 규칙은 기존과 완전히 동일하다: D일 뉴스 -> D보다 뒤인 첫 거래일
# (merge_asof forward, allow_exact_matches=False). load_news_with_basis_horizon_z()를
# taskf_gating_basis.py에서 그대로 재사용한다.
#
# 대상 조합(4개, "지금까지 결과가 나은 편" 기준 — 결과_TaskF_게이팅검증.md [3]/[5] 참고):
#   excess_return  / 방향 3-class  / h=3  (Δ+0.1501, 전체 실험 중 raw Δ 최댓값)
#   absolute_return/ 방향 3-class  / h=3  (Δ+0.1467, basis 비교에서 두 번째)
#   excess_return  / 변동성 2-class/ h=1  (Δ+0.1009, h=1 대표 조합)
#   absolute_return/ 변동성 2-class/ h=1  (Δ+0.1020, h=1에서 basis 비교 시 근소 우세)
# 윈도우 N=60 고정(기존 축소 근거 동일).
#
# 검증: 블록 셔플(h=1은 block_size=1, h=3은 block_size=3) + 클래스 사전분포 무작위 예측기(200회).
# 판정 기준은 taskf_gating_horizon.py/taskf_gating_basis.py와 동일(verdict() 재사용).
#
# 뉴스 테이블에는 아무것도 쓰지 않는다 — 전부 조회 전용.

import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from constants import ACTIVE_TICKERS
from 감성분석.taskf_gating import LABEL_SCHEMES, VECTORIZER_KWARGS, build_text, majority_baseline
from 감성분석.taskf_gating_basis import load_news_with_basis_horizon_z
from 감성분석.taskf_gating_horizon import block_shuffle_labels, verdict

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── 사람이 지정한 주가 민감 키워드(사건성, "미래에 영향을 줄 사건"만) ─────────────
RELEVANCE_KEYWORDS = [
    "계약", "수주", "소송", "특허", "인수", "합병", "감산", "증산",
    "리콜", "파업", "제재", "승인", "허가", "유상증자", "무상증자", "감사",
    "목표주가", "투자의견",
]
RELEVANCE_PATTERN = re.compile("|".join(re.escape(k) for k in RELEVANCE_KEYWORDS))

# 상위 피처 점검용 — 이미 일어난 가격 변동/사건 결과를 사후 서술하는 어휘 감시 목록
# (제목에 RELEVANCE_KEYWORDS가 없어도 본문식 서술이 top feature에 섞여 들어오는지 확인하는 용도)
POST_HOC_WATCH_WORDS = [
    "상한가", "하한가", "급등", "급락", "신고가", "신저가", "최고치", "최저", "반토막",
    "폭락", "폭등", "일단락", "타결", "가결", "종료", "마감", "강세", "약세",
]

FIXED_WINDOW_N = 60
# ⚠️ 2026-09-06: 초기 실행은 기존 관례대로 N_SHUFFLE=30을 썼는데, 그 결과 하나의 조합이
# p=0.0323(=1/31, N=30에서 나올 수 있는 가장 작은 0 초과 p값 그 자체)으로 "유의함" 판정을
# 받았다. 이게 진짜 신호인지 셔플 해상도 부족(경계값 우연)인지 구분하려고 N=200으로 재검증한
# 결과 p=0.0796으로 뒤집혔다(유의하지 않음) — 상세는 결과 보고 참고. 이후 이 스크립트는
# 처음부터 N=200으로 실행해 같은 경계 오판을 재발시키지 않는다.
N_SHUFFLE = 200
N_RANDOM_PRIOR = 200

TARGET_COMBOS = [
    ("excess_return", "방향 3-class", 3),
    ("absolute_return", "방향 3-class", 3),
    ("excess_return", "변동성 2-class", 1),
    ("absolute_return", "변동성 2-class", 1),
]


def is_relevant_title(title):
    return bool(RELEVANCE_PATTERN.search(title or ""))


def print_prefilter_stats(ticker):
    """요구사항 1 — 실행 전에 필터링 후 남는 기사 수를 먼저 보여준다."""
    merged, _ = load_news_with_basis_horizon_z(ticker, "excess_return", window_n=FIXED_WINDOW_N, horizons=[1, 3])
    merged = merged.drop_duplicates(subset=["date", "title"])  # 기사 단위 집계(여러 horizon 컬럼 merge로 인한 중복 없음, 방어적)
    merged["is_relevant"] = merged["title"].apply(is_relevant_title)

    total = len(merged)
    relevant = int(merged["is_relevant"].sum())
    pct = relevant / total if total else 0.0

    print("=" * 90)
    print("[사전 점검] 키워드 필터 적용 시 남는 기사 수 (TRAIN_PERIOD 내, D+1 매칭 이후 기준)")
    print("=" * 90)
    print(f"전체 {total}건 -> 관련성 높음 {relevant}건 ({pct:.1%})")
    if pct < 0.10:
        print(f"⚠️ 10% 미만입니다 — 표본이 적어질 수 있습니다.")

    print("\nsplit별 분포:")
    for split in ["train", "val", "test"]:
        sub = merged[merged["split"] == split]
        n_total = len(sub)
        n_rel = int(sub["is_relevant"].sum())
        print(f"  {split}: 전체 {n_total} -> 관련성 높음 {n_rel} ({n_rel/n_total:.1%} of split)" if n_total else f"  {split}: 0건")

    print("\n키워드별 매칭 건수 (제목에 해당 키워드가 있는 기사 수, 중복 집계 가능):")
    cnt = Counter()
    for t in merged["title"]:
        for k in RELEVANCE_KEYWORDS:
            if k in t:
                cnt[k] += 1
    for k in RELEVANCE_KEYWORDS:
        print(f"  {k}: {cnt.get(k, 0)}")
    zero_hit = [k for k in RELEVANCE_KEYWORDS if cnt.get(k, 0) == 0]

    print("\n제안:")
    if pct < 0.10:
        print(
            "  - 10% 미만이지만 절대 건수는 " + str(relevant) + "건으로 TF-IDF 학습 자체는 가능한 수준입니다. "
            "그대로 진행하되 결과 해석 시 표본 크기를 감안할 것."
        )
        if zero_hit:
            print(f"  - 0건 키워드({', '.join(zero_hit)})는 이 종목·기간에 해당 사건이 없었다는 뜻 — 목록에서 빼도 결과에 영향 없음.")
        print(
            "  - 표본을 늘리고 싶다면 '단독'/'MOU'/'협력'/'양해각서'처럼 사건성이되 아직 사람이 배제하지 않은 "
            "키워드를 추가하는 방법이 있으나, 이는 원 키워드 목록의 설계 원칙(결과 서술 배제)을 벗어나지 "
            "않는 범위에서 사람이 재검토할 사항."
        )
    else:
        print("  - 10% 이상이라 목록 조정 없이 진행합니다.")
    print()
    return merged


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


def run_combo(tag, label_basis, scheme_name, horizon_h, merged):
    train_df, val_df, test_df, classes = prepare_combo(scheme_name, horizon_h, merged)

    if len(train_df) == 0 or len(test_df) == 0 or train_df["label"].nunique() < 2:
        return {
            "tag": tag, "label_basis": label_basis, "scheme": scheme_name, "horizon_h": horizon_h,
            "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
            "insufficient": True,
        }

    vec = TfidfVectorizer(**VECTORIZER_KWARGS)
    X_train = vec.fit_transform(build_text(train_df))
    X_test = vec.transform(build_text(test_df))

    y_train_real = train_df["label"].values
    y_test = test_df["label"].values

    real_f1, real_acc = fit_predict_f1(X_train, y_train_real, X_test, y_test, classes)
    test_base_f1, test_base_acc, test_majority = majority_baseline(y_test, classes)
    real_delta = real_f1 - test_base_f1

    shuffle_f1s = []
    for i in range(N_SHUFFLE):
        rng = np.random.RandomState(1000 + i)
        shuffled_map = block_shuffle_labels(train_df, block_size=horizon_h, rng=rng)
        y_shuffled = train_df["trading_date"].map(shuffled_map).values
        f1, _ = fit_predict_f1(X_train, y_shuffled, X_test, y_test, classes)
        shuffle_f1s.append(f1)
    shuffle_f1s = np.array(shuffle_f1s)
    shuffle_deltas = shuffle_f1s - test_base_f1
    ci_lo, ci_hi = np.percentile(shuffle_deltas, [2.5, 97.5])
    p_value = (np.sum(shuffle_f1s >= real_f1) + 1) / (N_SHUFFLE + 1)

    train_counts = pd.Series(y_train_real).value_counts().reindex(classes, fill_value=0)
    prior = (train_counts / train_counts.sum()).values
    prior_f1s = []
    for i in range(N_RANDOM_PRIOR):
        rng = np.random.RandomState(5000 + i)
        pred = rng.choice(classes, size=len(y_test), p=prior)
        prior_f1s.append(f1_score(y_test, pred, average="macro", labels=classes, zero_division=0))
    prior_f1s = np.array(prior_f1s)

    test_day_counts = (
        merged[merged["split"] == "test"]
        .drop_duplicates(subset=["trading_date"])[f"z_h{horizon_h}"]
        .apply(lambda v: LABEL_SCHEMES[scheme_name]["fn"](v, **LABEL_SCHEMES[scheme_name]["kwargs"]))
        .value_counts().reindex(classes, fill_value=0)
    )
    min_test_day_count = int(test_day_counts.min())

    return {
        "tag": tag, "label_basis": label_basis, "scheme": scheme_name, "horizon_h": horizon_h,
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "real_f1": real_f1, "real_acc": real_acc, "real_delta": real_delta,
        "test_base_f1": test_base_f1, "test_base_acc": test_base_acc, "test_majority": test_majority,
        "shuffle_mean": shuffle_deltas.mean(), "shuffle_std": shuffle_deltas.std(), "ci_lo": ci_lo, "ci_hi": ci_hi,
        "p_value": p_value,
        "prior_f1_mean": prior_f1s.mean(), "prior_f1_std": prior_f1s.std(),
        "min_test_day_count": min_test_day_count,
        "insufficient": False,
        "_merged": merged,
    }


def print_top_features_with_posthoc_check(tag, label_basis, scheme_name, horizon_h, merged, top_n=15):
    scheme = LABEL_SCHEMES[scheme_name]
    z_col = f"z_h{horizon_h}"
    df = merged.copy()
    df["label"] = df[z_col].apply(lambda v: scheme["fn"](v, **scheme["kwargs"]))
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    train_df = df[df["split"] == "train"]

    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2)
    X_train = vec.fit_transform(build_text(train_df))
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X_train, train_df["label"])

    feature_names = np.array(vec.get_feature_names_out())
    print(f"\n--- 상위 피처: [{tag} / {label_basis} / {scheme_name} / h={horizon_h}] word 1-2gram ---")
    posthoc_hits = []
    for i, cls in enumerate(clf.classes_):
        coefs = clf.coef_[i]
        top_idx = np.argsort(coefs)[::-1][:top_n]
        words = [feature_names[j] for j in top_idx]
        for w in words:
            if any(p in w for p in POST_HOC_WATCH_WORDS):
                posthoc_hits.append((int(cls), w))
        print(f"  [label={int(cls):+d}] " + ", ".join(f"{feature_names[j]}({coefs[j]:.2f})" for j in top_idx))

    if posthoc_hits:
        print(f"  ⚠️ 사후 서술 의심 어휘 발견: {posthoc_hits}")
    else:
        print("  사후 서술 의심 어휘(POST_HOC_WATCH_WORDS) 없음")


def main():
    for ticker in ACTIVE_TICKERS:
        print("\n" + "#" * 100)
        print(f"# Task F 키워드 필터 게이팅 — {ticker}, window_n={FIXED_WINDOW_N} 고정")
        print("#" * 100)

        # ── 1) 사전 점검 (요구사항 1) ──
        print_prefilter_stats(ticker)

        # ── 2)+3) 필터 전/후 게이팅 + 검증 ──
        results = []
        merged_cache = {}
        for basis in ["excess_return", "absolute_return"]:
            merged, _ = load_news_with_basis_horizon_z(ticker, basis, window_n=FIXED_WINDOW_N, horizons=[1, 3])
            if merged.empty:
                continue
            merged["is_relevant"] = merged["title"].apply(is_relevant_title)
            merged_cache[basis] = merged

        for basis, scheme_name, h in TARGET_COMBOS:
            merged = merged_cache.get(basis)
            if merged is None:
                print(f"⚠️ {ticker}/{basis}: 조인된 뉴스 없음 — 스킵")
                continue

            print(f"\n[{basis} / {scheme_name} / h={h}] 필터 전(all) 학습+검증...")
            r_all = run_combo("all", basis, scheme_name, h, merged)
            results.append(r_all)
            if r_all.get("insufficient"):
                print(f"  ⚠️ 표본 부족 — n_train={r_all['n_train']}, n_test={r_all['n_test']}")
            else:
                print(
                    f"  n(train/val/test)={r_all['n_train']}/{r_all['n_val']}/{r_all['n_test']}  "
                    f"test F1={r_all['real_f1']:.4f} (baseline {r_all['test_base_f1']:.4f}, Δ{r_all['real_delta']:+.4f})  "
                    f"셔플p={r_all['p_value']:.4f}  사전분포F1={r_all['prior_f1_mean']:.4f}±{r_all['prior_f1_std']:.4f}"
                )

            print(f"[{basis} / {scheme_name} / h={h}] 필터 후(relevant only) 학습+검증...")
            merged_rel = merged[merged["is_relevant"]].reset_index(drop=True)
            r_rel = run_combo("relevant", basis, scheme_name, h, merged_rel)
            results.append(r_rel)
            if r_rel.get("insufficient"):
                print(f"  ⚠️ 표본 부족 — n_train={r_rel['n_train']}, n_test={r_rel['n_test']} (판정 불가)")
            else:
                print(
                    f"  n(train/val/test)={r_rel['n_train']}/{r_rel['n_val']}/{r_rel['n_test']}  "
                    f"test F1={r_rel['real_f1']:.4f} (baseline {r_rel['test_base_f1']:.4f}, Δ{r_rel['real_delta']:+.4f})  "
                    f"셔플p={r_rel['p_value']:.4f}  사전분포F1={r_rel['prior_f1_mean']:.4f}±{r_rel['prior_f1_std']:.4f}"
                )

        # ── 4) 종합 비교표 ──
        valid_results = [r for r in results if not r.get("insufficient")]
        print("\n" + "=" * 130)
        print("=== 필터 전(all) vs 필터 후(relevant) 종합 비교표 ===")
        print("=" * 130)
        header = (f"{'구분':<10}{'basis':<16}{'라벨':<16}{'h':>3}{'n_train':>9}{'n_test':>8}"
                  f"{'testF1':>9}{'base':>8}{'Δtest':>9}{'셔플p':>8}{'사전F1':>9}{'판정':>18}")
        print(header)
        for r in valid_results:
            v, _ = verdict(r)
            print(
                f"{r['tag']:<10}{r['label_basis']:<16}{r['scheme']:<16}{r['horizon_h']:>3}"
                f"{r['n_train']:>9}{r['n_test']:>8}{r['real_f1']:>9.4f}{r['test_base_f1']:>8.4f}"
                f"{r['real_delta']:>9.4f}{r['p_value']:>8.4f}{r['prior_f1_mean']:>9.4f}{v:>18}"
            )
        for r in results:
            if r.get("insufficient"):
                print(f"{r['tag']:<10}{r['label_basis']:<16}{r['scheme']:<16}{r['horizon_h']:>3}"
                      f"{r['n_train']:>9}{r['n_test']:>8}   판정 불가(표본 부족: train 클래스 다양성 또는 표본 자체 부족)")

        print("\n=== 필터 전/후 직접 대응 비교 (같은 basis x 라벨 x h) ===")
        for basis, scheme_name, h in TARGET_COMBOS:
            pair = {r["tag"]: r for r in results if r["label_basis"] == basis and r["scheme"] == scheme_name and r["horizon_h"] == h}
            all_r, rel_r = pair.get("all"), pair.get("relevant")
            if all_r is None or rel_r is None:
                continue
            all_desc = f"Δ{all_r['real_delta']:+.4f}(p={all_r['p_value']:.3f})" if not all_r.get("insufficient") else "표본부족"
            rel_desc = f"Δ{rel_r['real_delta']:+.4f}(p={rel_r['p_value']:.3f})" if not rel_r.get("insufficient") else "표본부족"
            print(f"  [{basis} / {scheme_name} / h={h}] 전체: {all_desc}  vs  관련성높음: {rel_desc}")

        print("\n=== 종합 판정 ===")
        any_significant = False
        for r in valid_results:
            v, reason = verdict(r)
            print(f"  [{r['tag']} / {r['label_basis']} / {r['scheme']} / h={r['horizon_h']}] {v} — {reason}")
            if v == "유의함":
                any_significant = True

        if any_significant:
            print("\n유의한 조합의 상위 피처 + 사후 서술 어휘 점검(요구사항 4):")
            for r in valid_results:
                v, _ = verdict(r)
                if v == "유의함":
                    print_top_features_with_posthoc_check(r["tag"], r["label_basis"], r["scheme"], r["horizon_h"], r["_merged"])
        else:
            print("\n유의한 조합 없음 — 상위 피처 점검 생략(신호가 확인되지 않은 조합의 피처를 사후에 "
                  "들여다보는 것은 해석 남용 위험).")


if __name__ == "__main__":
    main()
