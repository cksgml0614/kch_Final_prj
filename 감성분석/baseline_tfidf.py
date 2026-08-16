"""
Task 0 — TF-IDF + 로지스틱 회귀 베이스라인 및 라벨 신호 진단

목적: KoBERT 1차 학습의 저조한 성능(test macro F1=0.1473)이
      "라벨 신호 자체의 부족" 때문인지 "KoBERT 설정 문제" 때문인지 분리한다.

실행: 프로젝트 루트에서 `python -m 감성분석.baseline_tfidf`
"""
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score

from 감성분석.kobert_dataset import LABEL_MAP, load_labeled_news, split_by_ratio

# 클래스 인덱스 -> 실제 라벨(-2~2) 역매핑
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = len(LABEL_MAP)
TARGET_NAMES = [f"score={INV_LABEL_MAP[i]}" for i in range(NUM_CLASSES)]


def build_text(df):
    """title + ' ' + summary 로 입력 텍스트 결합"""
    title = df["title"].fillna("").astype(str)
    summary = df["summary"].fillna("").astype(str)
    return (title + " " + summary).str.strip()


def print_label_distribution(name, df):
    counts = df["label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
    total = len(df)
    print(f"\n[{name}] 총 {total}건")
    for cls in range(NUM_CLASSES):
        cnt = int(counts[cls])
        ratio = cnt / total * 100 if total else 0.0
        print(f"  score={INV_LABEL_MAP[cls]:+d} (idx {cls}): {cnt:6d}건  {ratio:5.2f}%")


def diagnose_dataset(df):
    """라벨-텍스트 대응이 성립하는지 판단하기 위한 데이터 구조 진단"""
    print("\n" + "=" * 70)
    print("진단 1. (ticker, date) 조합당 뉴스 행 개수 분포")
    print("=" * 70)

    group_sizes = df.groupby(["ticker", "date"]).size()
    print(f"  전체 (ticker, date) 고유 조합 수 : {len(group_sizes)}")
    print(f"  전체 뉴스 행 수                  : {len(df)}")
    print(f"  조합당 기사 수 - 최소            : {int(group_sizes.min())}")
    print(f"  조합당 기사 수 - 중앙값          : {group_sizes.median():.1f}")
    print(f"  조합당 기사 수 - 평균            : {group_sizes.mean():.2f}")
    print(f"  조합당 기사 수 - 최대            : {int(group_sizes.max())}")
    print(
        "  → 하나의 등락률 라벨을 몇 개의 기사가 공유하는지를 뜻함. "
        "실질 표본 수는 기사 수가 아니라 고유 조합 수에 가깝다."
    )

    print("\n  ticker별 고유 날짜 수:")
    for ticker, cnt in df.groupby("ticker")["date"].nunique().items():
        rows = int((df["ticker"] == ticker).sum())
        print(f"    {ticker}: 고유 날짜 {cnt}개 / 기사 {rows}건")

    print("\n" + "=" * 70)
    print("진단 2. 제목 완전 중복")
    print("=" * 70)
    title_counts = df["title"].fillna("").value_counts()
    dup_titles = title_counts[title_counts > 1]
    dup_rows = int(dup_titles.sum() - len(dup_titles))  # 중복으로 인한 초과 행 수
    print(f"  고유 제목 수            : {len(title_counts)}")
    print(f"  2회 이상 등장한 제목 수 : {len(dup_titles)}")
    print(f"  중복으로 인한 초과 행 수: {dup_rows} (전체의 {dup_rows / len(df) * 100:.2f}%)")
    print("\n  중복 제목 상위 10개:")
    for title, cnt in dup_titles.head(10).items():
        shown = title if len(title) <= 60 else title[:57] + "..."
        print(f"    {cnt:4d}회 | {shown}")


def majority_baseline(test_df):
    """test set 최빈 클래스로 전부 예측했을 때의 accuracy / macro F1"""
    y_test = test_df["label"].values
    counts = pd.Series(y_test).value_counts()
    majority_cls = int(counts.index[0])
    y_pred = np.full_like(y_test, majority_cls)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", labels=range(NUM_CLASSES), zero_division=0)
    return majority_cls, acc, macro_f1


def run_vectorizer(name, vectorizer, train_df, val_df, test_df):
    """주어진 벡터라이저 설정으로 학습 후 test 결과를 출력하고 (macro_f1, model, vectorizer) 반환"""
    print("\n" + "=" * 70)
    print(f"모델: TF-IDF ({name}) + LogisticRegression(class_weight='balanced')")
    print("=" * 70)

    X_train = vectorizer.fit_transform(build_text(train_df))
    X_val = vectorizer.transform(build_text(val_df))
    X_test = vectorizer.transform(build_text(test_df))
    print(f"  피처 수: {X_train.shape[1]}")

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X_train, y_train)

    val_pred = clf.predict(X_val)
    val_f1 = f1_score(y_val, val_pred, average="macro", labels=range(NUM_CLASSES), zero_division=0)
    print(f"  val  macro F1 : {val_f1:.4f}  (accuracy {accuracy_score(y_val, val_pred):.4f})")

    test_pred = clf.predict(X_test)
    test_f1 = f1_score(y_test, test_pred, average="macro", labels=range(NUM_CLASSES), zero_division=0)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"  test macro F1 : {test_f1:.4f}  (accuracy {test_acc:.4f})")

    print("\n  [test set classification_report]")
    report = classification_report(
        y_test,
        test_pred,
        labels=range(NUM_CLASSES),
        target_names=TARGET_NAMES,
        zero_division=0,
    )
    print("  " + report.replace("\n", "\n  "))

    return {"name": name, "val_f1": val_f1, "test_f1": test_f1, "test_acc": test_acc}, clf, vectorizer


def print_top_features(clf, vectorizer, top_n=15):
    """라벨별 로지스틱 회귀 계수 상/하위 top_n개 단어 출력"""
    print("\n" + "=" * 70)
    print(f"진단 3. 라벨별 상위/하위 특징 단어 (word TF-IDF 계수 기준, 각 {top_n}개)")
    print("=" * 70)
    print("  → 의미 있는 단어인지, 종목명·기자명·언론사 같은 무의미 토큰인지 육안 확인")

    feature_names = np.array(vectorizer.get_feature_names_out())
    for i, cls in enumerate(clf.classes_):
        coefs = clf.coef_[i]
        top_idx = np.argsort(coefs)[::-1][:top_n]
        bottom_idx = np.argsort(coefs)[:top_n]

        print(f"\n  [score={INV_LABEL_MAP[int(cls)]:+d} (idx {int(cls)})]")
        print("    (+) " + ", ".join(f"{feature_names[j]}({coefs[j]:.2f})" for j in top_idx))
        print("    (-) " + ", ".join(f"{feature_names[j]}({coefs[j]:.2f})" for j in bottom_idx))


def main():
    print("=" * 70)
    print("Task 0 — TF-IDF 베이스라인 & 라벨 신호 진단")
    print("=" * 70)

    df = load_labeled_news()
    if df.empty:
        print("⚠️ 라벨링된 뉴스가 없습니다. 종료합니다.")
        return

    print(f"\n라벨링된 뉴스 총 {len(df)}건 "
          f"(기간: {df['date'].min()} ~ {df['date'].max()})")

    diagnose_dataset(df)

    # KoBERT와 동일한 시계열 순서 분할 (70/15/15, 셔플 없음)
    train_df, val_df, test_df = split_by_ratio(df)

    print("\n" + "=" * 70)
    print("진단 4. split별 라벨 분포 (KoBERT와 동일한 시계열 분할)")
    print("=" * 70)
    print_label_distribution("train", train_df)
    print_label_distribution("val", val_df)
    print_label_distribution("test", test_df)
    print(f"\n  train 기간: {train_df['date'].min()} ~ {train_df['date'].max()}")
    print(f"  val   기간: {val_df['date'].min()} ~ {val_df['date'].max()}")
    print(f"  test  기간: {test_df['date'].min()} ~ {test_df['date'].max()}")

    # 다수결 baseline
    majority_cls, maj_acc, maj_f1 = majority_baseline(test_df)
    print("\n" + "=" * 70)
    print("기준선: 다수결 baseline (test set 최빈 클래스로 전부 예측)")
    print("=" * 70)
    print(f"  최빈 클래스   : score={INV_LABEL_MAP[majority_cls]:+d} (idx {majority_cls})")
    print(f"  accuracy      : {maj_acc:.4f}")
    print(f"  macro F1      : {maj_f1:.4f}")

    results = []

    word_result, word_clf, word_vec = run_vectorizer(
        "word 1-2gram, min_df=2",
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2),
        train_df, val_df, test_df,
    )
    results.append(word_result)

    char_result, _, _ = run_vectorizer(
        "char_wb 2-4gram, min_df=2",
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2),
        train_df, val_df, test_df,
    )
    results.append(char_result)

    # 계수 해석은 단어 단위 모델로만 (char n-gram은 육안 해석 불가)
    print_top_features(word_clf, word_vec)

    print("\n" + "=" * 70)
    print("최종 비교")
    print("=" * 70)
    print(f"  {'설정':<32} {'test macro F1':>14} {'test accuracy':>14}")
    print(f"  {'-' * 32} {'-' * 14} {'-' * 14}")
    print(f"  {'다수결 baseline':<32} {maj_f1:>14.4f} {maj_acc:>14.4f}")
    for r in results:
        print(f"  {'TF-IDF ' + r['name']:<32} {r['test_f1']:>14.4f} {r['test_acc']:>14.4f}")
    print(f"  {'KoBERT 1차 학습 (기존 결과)':<32} {0.1473:>14.4f} {0.15:>14.4f}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()