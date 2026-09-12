# taskf_gating.py
# Task F 게이팅 — TASK_EF_라벨링_비교실험.md "게이팅 규칙": 조합(라벨 3종 x 윈도우 3종 = 9개)마다
# TF-IDF + 로지스틱 회귀를 먼저 돌려, 해당 split의 실제 최빈 클래스 baseline을 유의하게 상회하는
# 조합에만 KR-FinBERT를 투입한다. 여기서는 게이팅까지만 — 정지 지점(완료 조건 출력 후 정지).
#
# 데이터: daily_news(source='search_backfill', ticker=005930) 제목만. daily_labels와
# (ticker, date)로 조인한다 — 뉴스 date가 휴장일(주말/공휴일)이면 다음 거래일로 귀속한다
# (TASK_EF "휴장일 뉴스 귀속 규칙"). 빅카인즈 교차 평가는 이 게이팅 결과를 본 뒤 진행.
#
# 뉴스 테이블에는 아무것도 쓰지 않는다 — daily_news/daily_labels 모두 조회 전용.

import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from constants import ACTIVE_TICKERS, TRAIN_PERIOD_END, TRAIN_PERIOD_START
from 감성분석.kobert_dataset import load_labeled_news
from 라벨.라벨_공통 import (
    CONFIRMED_VOLATILITY_THRESHOLD,
    DIRECTION_3CLASS_THRESHOLD,
    DIRECTION_5CLASS_THRESHOLDS,
    WINDOW_SIZES,
    label_direction_3class,
    label_direction_5class,
    label_volatility_2class,
    load_labels,
    split_dates_by_ratio,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SOURCE = "search_backfill"

# 2026-08-30 확정 임계값(CLAUDE.md D-2) 그대로 — 3종 라벨 스킴, 게이팅 단계에서는 후보 비교가
# 아니라 확정값만 쓴다.
LABEL_SCHEMES = {
    "방향 5-class": {
        "fn": label_direction_5class,
        "classes": [-2, -1, 0, 1, 2],
        "kwargs": {"t1": DIRECTION_5CLASS_THRESHOLDS[0], "t2": DIRECTION_5CLASS_THRESHOLDS[1]},
    },
    "방향 3-class": {
        "fn": label_direction_3class,
        "classes": [-1, 0, 1],
        "kwargs": {"t": DIRECTION_3CLASS_THRESHOLD},
    },
    "변동성 2-class": {
        "fn": label_volatility_2class,
        "classes": [0, 1],
        "kwargs": {"t": CONFIRMED_VOLATILITY_THRESHOLD},
    },
}

# Task A 진단(baseline_tfidf.py)에서 word 1-2gram보다 char_wb 2-4gram이 test macro F1이 더
# 높았다(0.1498 vs 0.1313) — 제목만 쓰는 짧은 텍스트에서도 서브워드 신호가 더 안정적일 것으로
# 보고 게이팅 단계는 이 한 설정으로 통일한다(9개 조합을 벡터라이저까지 곱하면 통제가 어려워짐).
VECTORIZER_KWARGS = dict(analyzer="char_wb", ngram_range=(2, 4), min_df=2)


def build_text(df):
    return df["title"].fillna("").astype(str).str.strip()


def load_news_with_z(ticker):
    """제목 텍스트 + 윈도우별 z_score(wide)를 뉴스 (ticker,date) 단위로 결합한다.
    반환 컬럼: ticker, title, date(원 기사 date), trading_date(귀속된 거래일), split,
               z_20, z_60, z_120"""
    news = load_labeled_news(ticker=ticker, table_name="daily_news", label_column=None, source=SOURCE)
    if news.empty:
        return pd.DataFrame()
    news = news[["ticker", "date", "title"]].copy()
    news["date"] = pd.to_datetime(news["date"])
    news = news.sort_values("date")

    long_labels = load_labels(ticker)  # 전체 이력(2020~), NULL 없는 신뢰 가능한 거래일 캘린더
    if long_labels.empty:
        raise RuntimeError(f"{ticker}: daily_labels 비어있음 — 라벨_생성.py를 먼저 실행할 것")

    wide = long_labels.pivot(index="date", columns="window_n", values="z_score")
    wide.columns = [f"z_{int(c)}" for c in wide.columns]
    wide = wide.reset_index()

    # D일 뉴스 -> D보다 뒤인 첫 거래일(=D+1 거래일 또는 그 다음, 휴장일이 끼면 더 뒤)에 귀속.
    # search_backfill/빅카인즈 둘 다 발행 "시각"이 없어 D일 뉴스가 그날 장중/장마감 후 어느
    # 시점에 나왔는지 구분할 수 없다 — D일 뉴스를 D일 라벨(그날 종가 기준 excess_return)에
    # 매칭하면 장 마감 후 뉴스가 "그날 결과를 사후 서술"했을 가능성을 배제하지 못해 누수가
    # 생긴다. allow_exact_matches=False로 D 당일은 매칭 후보에서 제외해 항상 D보다 엄격히
    # 뒤인 거래일에만 귀속시킨다(feature_loader.py의 published_date < 거래일 누수방지 조인과
    # 동일한 관용구, 방향만 반대). 휴장일 뉴스도 이 규칙 하나로 자동 처리된다 — 휴장일 자체는
    # 거래일 후보가 아니므로 "다음 거래일"이 곧 "D보다 뒤인 첫 거래일"과 같다.
    # 2026-09-01 수정: 이전 버전은 allow_exact_matches 기본값(True)이라 거래일 뉴스가 당일
    # 라벨에 매칭되는 누수가 있었다(TASK_EF_라벨링_비교실험.md "뉴스 라벨 매칭 규칙" 참고).
    trading_calendar = pd.DataFrame({"trading_date": sorted(wide["date"].unique())})
    merged = pd.merge_asof(
        news, trading_calendar, left_on="date", right_on="trading_date",
        direction="forward", allow_exact_matches=False,
    )
    merged = merged.dropna(subset=["trading_date"])  # 마지막 거래일 이후 뉴스(있다면) 제외

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


def majority_baseline(y, classes):
    counts = pd.Series(y).value_counts()
    majority_cls = counts.index[0]
    y_pred = np.full(len(y), majority_cls)
    f1 = f1_score(y, y_pred, average="macro", labels=classes, zero_division=0)
    acc = accuracy_score(y, y_pred)
    return f1, acc, majority_cls


def run_combo(scheme_name, window_n, merged):
    scheme = LABEL_SCHEMES[scheme_name]
    z_col = f"z_{window_n}"

    df = merged.copy()
    df["label"] = df[z_col].apply(lambda v: scheme["fn"](v, **scheme["kwargs"]))
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    classes = scheme["classes"]

    vec = TfidfVectorizer(**VECTORIZER_KWARGS)
    X_train = vec.fit_transform(build_text(train_df))
    X_val = vec.transform(build_text(val_df))
    X_test = vec.transform(build_text(test_df))

    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X_train, train_df["label"])

    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    val_f1 = f1_score(val_df["label"], val_pred, average="macro", labels=classes, zero_division=0)
    val_acc = accuracy_score(val_df["label"], val_pred)
    test_f1 = f1_score(test_df["label"], test_pred, average="macro", labels=classes, zero_division=0)
    test_acc = accuracy_score(test_df["label"], test_pred)

    val_base_f1, val_base_acc, val_majority = majority_baseline(val_df["label"], classes)
    test_base_f1, test_base_acc, test_majority = majority_baseline(test_df["label"], classes)

    return {
        "scheme": scheme_name,
        "window_n": window_n,
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "val_f1": val_f1, "val_acc": val_acc, "val_base_f1": val_base_f1, "val_base_acc": val_base_acc,
        "test_f1": test_f1, "test_acc": test_acc, "test_base_f1": test_base_f1, "test_base_acc": test_base_acc,
        "test_majority": test_majority,
    }


def print_top_features(scheme_name, window_n, merged, top_n=12):
    """1차 실험(TASK 0)이 날짜 파편·UI 텍스트를 '감성'으로 오인했던 실패를 반복하지 않는지
    육안 확인하는 진단. char_wb는 해석이 어려우므로(baseline_tfidf.py와 동일 이유) 이 진단에
    한해 word 단위 벡터라이저를 별도로 써서 최고 성능 조합 하나만 점검한다 — 9개 조합 정식
    비교와는 무관한 sanity check."""
    scheme = LABEL_SCHEMES[scheme_name]
    z_col = f"z_{window_n}"
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
    print(f"\n--- sanity check: [{scheme_name} / N={window_n}] word 1-2gram 상위 계수 (진단 전용) ---")
    for i, cls in enumerate(clf.classes_):
        coefs = clf.coef_[i]
        top_idx = np.argsort(coefs)[::-1][:top_n]
        print(f"  [label={int(cls):+d}] " + ", ".join(f"{feature_names[j]}({coefs[j]:.2f})" for j in top_idx))


def main():
    for ticker in ACTIVE_TICKERS:
        print("\n" + "#" * 90)
        print(f"# Task F 게이팅 — {ticker}, source={SOURCE!r}, 제목만, TF-IDF(char_wb 2-4gram)+LogisticRegression")
        print("#" * 90)

        merged, (train_dates, val_dates, test_dates) = load_news_with_z(ticker)
        if merged.empty:
            print(f"⚠️ {ticker}: 조인된 뉴스 없음 — 스킵")
            continue

        print(
            f"\n뉴스(제목) 총 {len(merged)}건 — train {(merged['split']=='train').sum()} / "
            f"val {(merged['split']=='val').sum()} / test {(merged['split']=='test').sum()}"
        )
        print(
            f"거래일 기준 split: train {len(train_dates)}일({train_dates.min().date()}~{train_dates.max().date()}) / "
            f"val {len(val_dates)}일({val_dates.min().date()}~{val_dates.max().date()}) / "
            f"test {len(test_dates)}일({test_dates.min().date()}~{test_dates.max().date()})"
        )

        results = []
        for scheme_name in LABEL_SCHEMES:
            for window_n in WINDOW_SIZES:
                r = run_combo(scheme_name, window_n, merged)
                results.append(r)
                print(
                    f"\n[{scheme_name} / N={window_n}] train={r['n_train']} val={r['n_val']} test={r['n_test']}\n"
                    f"  val : macro F1 {r['val_f1']:.4f} (baseline {r['val_base_f1']:.4f}, "
                    f"Δ{r['val_f1']-r['val_base_f1']:+.4f}) | accuracy {r['val_acc']:.4f} (baseline {r['val_base_acc']:.4f})\n"
                    f"  test: macro F1 {r['test_f1']:.4f} (baseline {r['test_base_f1']:.4f}, "
                    f"Δ{r['test_f1']-r['test_base_f1']:+.4f}) | accuracy {r['test_acc']:.4f} (baseline {r['test_base_acc']:.4f}) "
                    f"| test 최빈클래스={r['test_majority']}"
                )

        print("\n" + "=" * 90)
        print("게이팅 요약 — 9개 조합, test macro F1 기준 baseline 대비 차이 내림차순")
        print("=" * 90)
        header = f"{'라벨':<14}{'N':>5}{'val F1':>9}{'val base':>10}{'test F1':>9}{'test base':>10}{'Δtest':>9}"
        print(header)
        print("-" * len(header))
        ranked = sorted(results, key=lambda x: x["test_f1"] - x["test_base_f1"], reverse=True)
        for r in ranked:
            delta = r["test_f1"] - r["test_base_f1"]
            print(
                f"{r['scheme']:<14}{r['window_n']:>5}{r['val_f1']:>9.4f}{r['val_base_f1']:>10.4f}"
                f"{r['test_f1']:>9.4f}{r['test_base_f1']:>10.4f}{delta:>9.4f}"
            )

        best = ranked[0]
        print_top_features(best["scheme"], best["window_n"], merged)


if __name__ == "__main__":
    main()
