# model.py — 운영 변동성 모델 PooledLSTMRegressor(2026-09-27 추가, 2026-09-28부터 운영).
# 클래스 이름은 체크포인트 meta.json의 model_class·model_registry.model_class 문자열과 같아야 한다(바꾸지 말 것).
# 2026-10-17 정리: TransformerRegressor·PooledTransformerRegressor를 지웠다(git 태그 pre-cleanup 시점에 있음).

import torch
import torch.nn as nn

# 모델 계열 이름(2026-10-17 LSTM 단일화). 두 문자열은 체크포인트 meta.json의 model_class, model_registry.model_family/
# model_class, MLflow 태그 model_family·파라미터 MODEL_ARCHITECTURE/model_class에 그대로 기록된다 — 바꾸지 말 것.
# 2026-09-27 transformer -> lstm 전환(사람 결정), 2026-10-17 Transformer 학습 코드·MODEL_ARCHITECTURE 환경변수 삭제.
MODEL_FAMILY = "lstm"
MODEL_CLASS_NAME = "PooledLSTMRegressor"


class PooledLSTMRegressor(nn.Module):
    """다종목 pooled LSTM 모델(2026-09-27, 지도교수님 피드백 — "수치형 시계열은 수치형답게
    다루는 모델" 검토 트랙). 당시 PooledTransformerRegressor(2026-10-17 삭제)와 아키텍처만 다르고 나머지 조건(피처,
    split, 학습 루프)은 동일하게 두어 "아키텍처만 바꿨을 때의 효과"를 순수하게 비교하기 위한
    클래스다.

    종목 임베딩은 (당시 PooledTransformerRegressor와 같은 방식으로) 매 타임스텝 피처 벡터에
    concat한다 — 순환층 자체가 종목을 알고 시간 패턴을 처리해야
    종목마다 다른 반응성을 표현할 수 있다. Transformer의 input_proj/pos_embedding에 해당하는
    층은 두지 않는다 — LSTM은 입력 가중치를 자체적으로 갖고 순서 정보도 순환 구조로 처리하므로
    위치 임베딩이 필요 없다. 출력은 Transformer와 같이 마지막 타임스텝 은닉 상태 -> dropout ->
    선형 head.

    forward(x, ticker_ids) — store.predict_and_save_for_ticker와 train_common.run_pooled_epoch가 이 인터페이스로
    호출한다. 단방향으로 시작(양방향은 다음 단계 선택지)."""

    def __init__(self, n_features, lookback, num_stocks, embedding_dim=8,
                 hidden_size=32, num_layers=2, dropout=0.3):
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
        # x: (B, L, F), ticker_ids: (B,) 정수 텐서
        _, L, _ = x.shape
        emb = self.stock_embedding(ticker_ids)             # (B, E)
        emb = emb.unsqueeze(1).expand(-1, L, -1)            # (B, L, E) — 매 타임스텝에 동일 종목 임베딩 broadcast
        h = torch.cat([x, emb], dim=-1)                     # (B, L, F+E)
        h, _ = self.lstm(h)                                 # (B, L, hidden)
        last = h[:, -1, :]
        return self.head(self.dropout(last)).squeeze(-1)
