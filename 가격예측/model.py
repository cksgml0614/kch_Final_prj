# model.py — 운영 변동성 모델 PooledLSTMRegressor(종목 임베딩을 붙인 다종목 pooled LSTM)를 정의한다.
# 파이프라인 단계: 학습·예측에서 쓰는 모델 정의. 주요 진입: PooledLSTMRegressor, MODEL_CLASS_NAME.
# 배경: 결정기록.md의 "모델 클래스 이름과 계열 문자열은 바꾸지 않는다"

import torch
import torch.nn as nn

# 체크포인트 meta·model_registry·MLflow에 그대로 기록되는 계열·클래스 이름(바꾸지 말 것).
MODEL_FAMILY = "lstm"
MODEL_CLASS_NAME = "PooledLSTMRegressor"


class PooledLSTMRegressor(nn.Module):
    """
    설명: 종목 임베딩을 매 타임스텝 피처에 붙여 LSTM에 넣고, 마지막 은닉 상태로 다음 날 변동성을 회귀한다.
    주의 사항: 클래스 이름은 체크포인트 meta의 model_class와 같아야 저장된 모델을 다시 불러올 수 있다.
    입력 타입->반환 타입: forward(torch.Tensor(B,L,F), torch.LongTensor(B)) -> torch.Tensor(B)
    """

    def __init__(self, n_features, lookback, num_stocks, embedding_dim=8,
                 hidden_size=32, num_layers=2, dropout=0.3):
        """
        설명: 종목 임베딩, 단방향 LSTM, dropout, 선형 head를 만든다.
        주의 사항: lookback은 쓰지 않지만 체크포인트 model_kwargs 호환을 위해 받는다. 층이 1개면 LSTM dropout은 0.
        입력 타입->반환 타입: (int, int, int, int, int, int, float) -> None
        """
        super().__init__()
        self.stock_embedding = nn.Embedding(num_stocks, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=n_features + embedding_dim, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True, bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x, ticker_ids):
        """
        설명: (B, L, F) 시퀀스와 종목 id로 다음 날 변동성 예측값 B개를 돌려준다.
        주의 사항: x는 FeatureScaler로 표준화된 값이어야 하고 ticker_ids는 학습 때의 ticker_to_id 번호여야 한다.
        입력 타입->반환 타입: (torch.Tensor, torch.LongTensor) -> torch.Tensor
        """
        # x: (B, L, F), ticker_ids: (B,) 정수 텐서
        _, L, _ = x.shape
        emb = self.stock_embedding(ticker_ids)             # (B, E)
        emb = emb.unsqueeze(1).expand(-1, L, -1)            # (B, L, E) — 매 타임스텝에 같은 종목 임베딩
        h = torch.cat([x, emb], dim=-1)                     # (B, L, F+E)
        h, _ = self.lstm(h)                                 # (B, L, hidden)
        last = h[:, -1, :]
        return self.head(self.dropout(last)).squeeze(-1)
