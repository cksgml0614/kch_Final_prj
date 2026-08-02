import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from db_manager import get_db_connection

MODEL_NAME = "skt/kobert-base-v1"
LABEL_MAP = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}


def load_labeled_news(ticker=None):
    """sentiment_score가 채워진 뉴스를 (ticker, date, title, summary, sentiment_score, label)로 반환"""
    conn = get_db_connection()
    columns = ["ticker", "date", "title", "summary", "sentiment_score"]
    if not conn:
        return pd.DataFrame(columns=columns + ["label"])

    try:
        query = """
                SELECT ticker, date, title, summary, sentiment_score
                FROM daily_news
                WHERE sentiment_score IS NOT NULL
                """
        params = []
        if ticker is not None:
            query += " AND ticker = %s"
            params.append(ticker)
        query += " ORDER BY date ASC"

        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]

        df = pd.DataFrame(rows, columns=colnames)
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