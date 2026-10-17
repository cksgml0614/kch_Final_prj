# sequence_dataset.py — 피처+target 데이터프레임을 lookback 시퀀스로 바꾸고, 날짜로 분할하고, 표준화한다.
# 파이프라인 단계: 시퀀스. 주요 진입: build_sequences, split_sequences_by_date, FeatureScaler.
# 피처는 이미 t-1 이하로 정렬돼 있어 target일 t의 시퀀스가 행 t까지 포함해도 누수가 아니다.

import numpy as np
import pandas as pd


def build_sequences(df, feature_cols, lookback):
    """
    설명: 날짜순 DataFrame을 연속 lookback행 단위 시퀀스 X, 마지막 행의 target y, 그 날짜로 바꾼다.
    주의 사항: df는 날짜 오름차순이고 'target' 열이 있어야 한다. 앞쪽 lookback-1행은 시퀀스를 만들지 못한다.
    입력 타입->반환 타입: (pd.DataFrame, list[str], int) -> (np.ndarray, np.ndarray, np.ndarray)
    """
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
    """
    설명: 시퀀스 target일을 기준으로 train(<=train_end)/val(<=val_end)/test로 나눈 dict를 돌려준다.
    주의 사항: train_end·val_end는 각 구간의 마지막 target 날짜다.
    입력 타입->반환 타입: (np.ndarray, np.ndarray, np.ndarray, pd.Timestamp, pd.Timestamp) -> dict
    """
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
    """
    설명: (N, L, F) 시퀀스를 피처별 평균·표준편차로 표준화한다.
    주의 사항: train에만 fit하고 val·test·서빙에는 transform만 쓴다(서빙은 저장해 둔 값을 load).
    입력 타입->반환 타입: fit(np.ndarray) -> FeatureScaler, transform(np.ndarray) -> np.ndarray
    """

    def __init__(self):
        """
        설명: fit 전 상태(mean_·std_ = None)로 만든다.
        주의 사항: 없음.
        입력 타입->반환 타입: () -> None
        """
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """
        설명: X를 (N*L, F)로 펼쳐 피처별 평균·표준편차를 구한다.
        주의 사항: 표준편차가 0인 피처는 1.0으로 바꿔 나눗셈을 막는다.
        입력 타입->반환 타입: (np.ndarray) -> FeatureScaler
        """
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # 상수 컬럼 방어
        return self

    def transform(self, X):
        """
        설명: fit한 평균·표준편차로 X를 표준화한다.
        주의 사항: fit 또는 load 뒤에만 부를 수 있다.
        입력 타입->반환 타입: (np.ndarray) -> np.ndarray
        """
        return (X - self.mean_) / self.std_

    def save(self, path):
        """
        설명: mean_·std_를 npz 파일 하나로 저장한다.
        주의 사항: 파일 쓰기. fit 전이면 RuntimeError. 서빙은 이 값을 그대로 다시 써야 한다(새로 fit 금지).
        입력 타입->반환 타입: (str) -> None
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("fit되지 않은 FeatureScaler는 저장할 수 없음")
        np.savez(path, mean=self.mean_, std=self.std_)

    @classmethod
    def load(cls, path):
        """
        설명: save로 저장한 npz에서 mean_·std_를 읽어 새 FeatureScaler를 만든다.
        주의 사항: 파일 읽기.
        입력 타입->반환 타입: (str) -> FeatureScaler
        """
        data = np.load(path)
        obj = cls()
        obj.mean_ = data["mean"]
        obj.std_ = data["std"]
        return obj
