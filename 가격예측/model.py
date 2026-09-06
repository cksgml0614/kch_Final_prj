# model.py — Task T-1 체크포인트 4
# 최소 구성 Transformer 회귀 모델 (TASK_T 확정: encoder 1~2layer, d_model 32~64, head 2~4,
# dropout 0.2~0.3). train 1,002시퀀스 규모 대비 과적합 방지를 위해 하한값 위주로 시작.
#
# 2026-09-06: PooledTransformerRegressor 추가 — 다종목 pooled 모델(Task T 다종목 확장, 별도
# 실험 트랙). TransformerRegressor는 값 변경 없음.

import torch
import torch.nn as nn


class TransformerRegressor(nn.Module):
    def __init__(self, n_features, lookback, d_model=32, nhead=2, num_layers=2,
                 dim_feedforward=64, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, lookback, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, L, F)
        h = self.input_proj(x) + self.pos_embedding
        h = self.encoder(h)          # (B, L, d_model)
        last = h[:, -1, :]           # 시퀀스 마지막 시점(=target일의 t-1) 표현만 사용
        return self.head(self.dropout(last)).squeeze(-1)

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class PooledTransformerRegressor(nn.Module):
    """다종목 pooled 모델(2026-09-06, Task T 다종목 확장 — 별도 실험 트랙). TransformerRegressor와
    구조는 같되 종목 임베딩(nn.Embedding)을 추가해, 모델 하나로 여러 종목을 함께 학습하면서도
    종목별로 다른 예측을 낼 수 있게 한다.

    결합 방식: 종목 임베딩을 매 타임스텝의 피처 벡터에 concat해 인코더 입력으로 준다(출력
    직전에만 붙이는 방식도 검토했으나 채택하지 않음) — 인코더 자체가 "어느 종목인지"를 알고
    시간적 패턴을 처리할 수 있어야 종목마다 다른 lag 구조/반응성을 표현할 수 있다고 판단했다.
    출력 직전에만 붙이면 인코더는 종목과 무관하게 동일한 인코딩만 하고 종목 정보는 최종 출력
    보정에만 쓰이게 되어 표현력이 떨어진다.

    기존 TransformerRegressor는 단일 종목 파이프라인(가격예측_공통.py/가격예측_일일수집.py)이
    계속 참조하므로 값을 전혀 바꾸지 않았다 — 이 클래스는 완전히 별개이며, forward() 시그니처가
    다르다(ticker_ids 추가 인자) — 그래서 train_common.py의 학습 루프도 pooled 전용 버전
    (train_pooled_transformer 등)을 별도로 둔다."""

    def __init__(self, n_features, lookback, num_stocks, embedding_dim=8,
                 d_model=32, nhead=2, num_layers=2, dim_feedforward=64, dropout=0.3):
        super().__init__()
        self.stock_embedding = nn.Embedding(num_stocks, embedding_dim)
        self.input_proj = nn.Linear(n_features + embedding_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, lookback, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, ticker_ids):
        # x: (B, L, F), ticker_ids: (B,) 정수 텐서
        B, L, _ = x.shape
        emb = self.stock_embedding(ticker_ids)             # (B, E)
        emb = emb.unsqueeze(1).expand(-1, L, -1)            # (B, L, E) — 매 타임스텝에 동일 종목 임베딩 broadcast
        h = torch.cat([x, emb], dim=-1)                     # (B, L, F+E)
        h = self.input_proj(h) + self.pos_embedding
        h = self.encoder(h)
        last = h[:, -1, :]
        return self.head(self.dropout(last)).squeeze(-1)

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
