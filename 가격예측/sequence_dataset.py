# sequence_dataset.py — Task T-1 체크포인트 4
# merge된 (feature+target) 데이터프레임을 lookback window 시퀀스(X)와 타깃(y)으로 변환한다.
#
# 윈도우 구성: normal 레짐 전체(정상 레짐, 스트레스 제외)를 날짜순 연속 구간으로 보고
# lookback개의 연속 행을 하나의 시퀀스로 묶는다. 각 행의 피처 컬럼은 이미 체크포인트 1/2에서
# t-1 이하로 정렬되어 있으므로, target일 t의 시퀀스가 row t 자신(피처는 t-1값)까지 포함해도
# 미래 누수가 아니다. 이 방식은 앞쪽 (lookback-1)개 target만 윈도우 부족으로 못 만들고,
# train/val/test 경계를 넘어 과거(train 쪽) 데이터를 val/test 시퀀스의 히스토리로 쓰는 것은
# "이미 확정된 과거 정보 재사용"이라 표준적인 시계열 분할 관행이며 누수가 아니다.

import numpy as np
import pandas as pd


def build_sequences(df, feature_cols, lookback):
    """df: 날짜 오름차순 정렬된 DataFrame (feature_cols + 'target').
    Returns X (N, lookback, n_features), y (N,), dates (N,) — 앞쪽 (lookback-1)행은 제외됨."""
    values = df[feature_cols].values
    targets = df["target"].values
    dates = df.index.values

    n = len(df)
    X, y, seq_dates = [], [], []
    for i in range(lookback - 1, n):
        X.append(values[i - lookback + 1 : i + 1])
        y.append(targets[i])
        seq_dates.append(dates[i])

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), np.asarray(seq_dates)


def split_sequences_by_date(X, y, dates, train_end, val_end):
    """dates(시퀀스의 target일) 기준으로 train/val/test로 나눈다.
    train_end, val_end: 각 구간의 마지막 target 날짜 (pd.Timestamp 또는 datetime64 비교 가능한 값)."""
    dates64 = dates.astype("datetime64[ns]")
    train_end64 = pd.Timestamp(train_end).to_datetime64()
    val_end64 = pd.Timestamp(val_end).to_datetime64()
    train_mask = dates64 <= train_end64
    val_mask = (dates64 > train_end64) & (dates64 <= val_end64)
    test_mask = dates64 > val_end64

    splits = {}
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        splits[name] = (X[mask], y[mask], dates[mask])
    return splits


class FeatureScaler:
    """train 시퀀스에만 fit하고 val/test에는 transform만 적용 (leak-safe).
    (N, L, F) 형태를 (N*L, F)로 펼쳐서 표준화 후 다시 원래 shape으로 되돌린다."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # 상수 컬럼 방어
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
