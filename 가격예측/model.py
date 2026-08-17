# model.py — Task T-1 체크포인트 4
# 최소 구성 Transformer 회귀 모델 (TASK_T 확정: encoder 1~2layer, d_model 32~64, head 2~4,
# dropout 0.2~0.3). train 1,002시퀀스 규모 대비 과적합 방지를 위해 하한값 위주로 시작.

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
