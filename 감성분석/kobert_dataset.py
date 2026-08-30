import re

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from db_manager import get_db_connection

MODEL_NAME = "skt/kobert-base-v1"
LABEL_MAP = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}

# table_name/label_column은 SQL 식별자라 %s로 파라미터화할 수 없다(psycopg는 값만 이스케이프
# 가능) — 직접 문자열로 끼워넣는 대신 화이트리스트/정규식으로 검증한 뒤 사용한다
# (TASK_EF_라벨링_비교실험.md "공통 설계 원칙": 데이터 소스 하드코딩 금지).
ALLOWED_TABLES = {"daily_news", "daily_news_bigkinds"}
_SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# daily_news/daily_news_bigkinds 공통 컬럼만 반환한다. daily_news는 sentiment_score(폐기
# 대상 라벨)를, daily_news_bigkinds는 keywords/category/bigkinds_id를 따로 갖는 등 스키마가
# 다르므로, 소스를 바꿔 끼워도 다운스트림(baseline_tfidf.py, kobert_train.py)이 항상 같은
# 컬럼 집합을 받도록 통일했다 — Task F 데이터소스 교차 평가(예: 빅카인즈로 학습해 크롤링으로
# 평가)가 성립하려면 컬럼 구조가 소스에 따라 달라지면 안 된다. 테이블 전용 컬럼이 필요하면
# 호출부에서 별도 조회할 것.
COMMON_COLUMNS = ["ticker", "date", "title", "summary", "press", "article_url", "target_date"]


def load_labeled_news(ticker=None, table_name="daily_news", label_column="sentiment_score", source=None):
    """
    table_name(daily_news / daily_news_bigkinds)에서 공통 컬럼 + label_column을 로드한다.

    - label_column=None: 라벨 필터(IS NOT NULL) 없이 전체를 로드한다. daily_news_bigkinds처럼
      아직 라벨이 없는 소스에서 Task E 착수 전에 텍스트만 먼저 확인할 때, 혹은 Task F처럼 라벨을
      daily_labels에서 별도로 조인해 붙일 때 쓴다.
    - label_column="sentiment_score"(기본값, 기존 호출부 하위 호환): 기존과 동일하게
      LABEL_MAP으로 매핑한 "label" 컬럼을 추가로 채운다.
    - 그 외 label_column(Task E에서 추가될 신규 라벨 등): 스킴이 아직 정해지지 않았으므로
      "label" 컬럼을 자동 생성하지 않고 raw 값 그대로 둔다 — 매핑은 호출부가 결정한다.
    - source: daily_news.source 값으로 필터(예: 'search_backfill'). None이면 전 소스(legacy/
      finance_crawl/search_backfill 등) 통합 — D-10에 따라 학습 정본은 'search_backfill'만이므로
      Task F 이후 호출부는 명시적으로 지정할 것. 값 파라미터라 %s로 안전하게 바인딩된다(테이블/
      컬럼명과 달리 식별자 검증 불필요).
    """
    if table_name not in ALLOWED_TABLES:
        raise ValueError(f"허용되지 않은 table_name: {table_name!r} (허용: {sorted(ALLOWED_TABLES)})")
    if label_column is not None and not _SAFE_IDENTIFIER_RE.match(label_column):
        raise ValueError(f"허용되지 않은 label_column: {label_column!r}")

    select_columns = list(COMMON_COLUMNS)
    if label_column is not None:
        select_columns.append(label_column)

    conn = get_db_connection()
    empty_columns = select_columns + (["label"] if label_column == "sentiment_score" else [])
    if not conn:
        return pd.DataFrame(columns=empty_columns)

    try:
        query = f"SELECT {', '.join(select_columns)} FROM {table_name}"
        conditions = []
        params = []
        if label_column is not None:
            conditions.append(f"{label_column} IS NOT NULL")
        if ticker is not None:
            conditions.append("ticker = %s")
            params.append(ticker)
        if source is not None:
            conditions.append("source = %s")
            params.append(source)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY date ASC"

        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]

        df = pd.DataFrame(rows, columns=colnames)
        if label_column == "sentiment_score":
            df["label"] = df["sentiment_score"].astype(int).map(LABEL_MAP)
        return df
    finally:
        conn.close()


def split_by_ratio(df, train_ratio=0.7, val_ratio=0.15):
    """date 오름차순 정렬된 df를 셔플 없이 순서대로 train/val/test로 분할"""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    return train_df, val_df, test_df


def compute_class_weights(df):
    """label 컬럼 기준 클래스별 빈도 역수로 class weight tensor 계산 (CrossEntropyLoss(weight=...)용)"""
    num_classes = len(LABEL_MAP)
    counts = df["label"].value_counts().reindex(range(num_classes), fill_value=0)

    weights = torch.zeros(num_classes, dtype=torch.float)
    for cls, cnt in counts.items():
        weights[cls] = 1.0 / cnt if cnt > 0 else 0.0
    return weights


class NewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        title = str(row["title"]) if pd.notna(row["title"]) else ""
        summary = str(row["summary"]) if pd.notna(row["summary"]) else ""

        encoded = self.tokenizer(
            title,
            summary,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            # skt/kobert-base-v1 토크나이저는 XLNetTokenizer 기반이라
            # return_token_type_ids를 명시하지 않으면 token_type_ids를 아예 반환하지 않음
            return_token_type_ids=True,
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            # 같은 이유로 XLNet 방식 세그먼트 id(0/1/2)가 나오는데,
            # 실제 모델(BertConfig)의 type_vocab_size는 2라서 2가 들어오면 임베딩 인덱스 에러가 남 -> 1로 clamp
            "token_type_ids": encoded["token_type_ids"].squeeze(0).clamp(max=1),
            "label": torch.tensor(row["label"], dtype=torch.long),
        }


if __name__ == "__main__":
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"⚠️ {MODEL_NAME} 로드 실패 (sentencepiece 관련 에러 추정: {e})")
        MODEL_NAME = "monologg/kobert"
        print(f"➡️ {MODEL_NAME}로 대체합니다.")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    df = load_labeled_news()
    print(f"📊 라벨링된 뉴스 총 {len(df)}건")

    train_df, val_df, test_df = split_by_ratio(df)
    print(f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    class_weights = compute_class_weights(train_df)
    print("class weights:", class_weights)

    if len(train_df) > 0:
        dataset = NewsDataset(train_df, tokenizer)
        loader = DataLoader(dataset, batch_size=8, shuffle=False)
        batch = next(iter(loader))
        for key, value in batch.items():
            print(f"{key}: {value.shape}")
    else:
        print("⚠️ train 데이터가 없어 배치 shape 확인을 건너뜁니다.")