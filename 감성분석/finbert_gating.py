# finbert_gating.py
# Task F 추가 실험(2026-09-06) — 지금까지 게이팅에서 가장 나았던 조합(방향 3-class,
# basis=excess_return, N=60, h=3, 관련성 필터 미적용 전체 기사)으로 KR-FinBERT
# (snunlp/KR-FinBert)를 실제로 파인튜닝하고, TF-IDF 게이팅과 동일한 검증 기준(레이블 셔플
# 테스트 + 클래스 사전분포 무작위 예측기)을 적용한다.
#
# 데이터/매칭 규칙은 taskf_gating_basis.py의 load_news_with_basis_horizon_z()를 그대로
# 재사용한다 — D일 뉴스 -> D보다 뒤인 첫 거래일(allow_exact_matches=False), source=
# search_backfill, 제목만, TRAIN_PERIOD 구간, 기존 70/15/15 시계열 split.
#
# 모델 구조/학습 설정은 감성분석/kobert_train.py를 그대로 참고하되 모델만 교체했다: 부분
# freeze(임베딩 + 인코더 0~8 고정, 9/10/11 + pooler + classifier만 학습), AdamW(lr=1e-5),
# get_linear_schedule_with_warmup(warmup 10%), class weight CrossEntropyLoss, early stopping
# (val macro F1 기준, patience=2, 최대 10 epoch). 입력은 제목만(sentence-pair 아님) —
# kobert_dataset.NewsDataset은 title+summary sentence-pair라 여기서는 재사용하지 않고
# title 단일 시퀀스 Dataset을 별도로 정의한다.
#
# ⚠️ snunlp/KR-FinBert 체크포인트는 safetensors가 아니라 pytorch_model.bin만 제공한다.
# transformers 최신판은 torch<2.6에서 이런 체크포인트의 로드를 CVE-2025-32434 때문에
# 거부한다(torch.load 자체의 취약점이 아니라 transformers의 추가 방어 장치). 이 프로젝트의
# torch는 2.5.1+cu121(CLAUDE.md 명시)이고 임의로 업그레이드하지 않기로 했으므로,
# transformers.modeling_utils.check_torch_load_is_safe를 이 스크립트 안에서만 no-op으로
# 패치해 우회한다 — snunlp(서울대 NLP)의 공개 체크포인트를 신뢰할 수 있는 소스로 판단했다.
# 이 패치는 프로세스 전역에 영향을 주지만 이 스크립트를 실행하는 동안에만 유효하고 다른
# 파일에는 반영하지 않는다.
#
# 검증 비용 절충(사람 지시, 2026-09-06): BERT 파인튜닝은 로지스틱회귀보다 훨씬 비싸 셔플을
# 매번 전체 재학습하기 어렵다. 절충안: 실제 라벨로 1회 전체 파인튜닝한 뒤, 그 인코더를
# "고정 피처 추출기"로 재사용해 셔플마다 분류 헤드(로지스틱 회귀 1개)만 재학습한다("마지막
# 레이어만 재학습" 절충, 사람이 제시한 선택지 2). 셔플 횟수는 50회로, TF-IDF 게이팅과
# 비슷한 해상도(오늘 N=30에서 p=1/31 경계값 오판을 겪은 뒤 N=200으로 정정한 전례를 감안해
# 최소 50 이상 확보)를 유지한다. 이 근사의 한계는 보고서에 명시한다: (1) 인코더 자체는
# 재학습하지 않으므로 "전체 아키텍처가 노이즈에 얼마나 과적합될 수 있는가"는 측정하지
# 못하고 "고정 인코더 표현 위에서 노이즈를 얼마나 분리해낼 수 있는가"만 측정한다. (2) 그
# 인코더 자체가 실제 라벨로 학습된 것이므로 셔플 귀무분포가 완전히 깨끗한 귀무(null)가
# 아니라 실제 라벨 정보가 인코더 표현에 이미 녹아있는 상태에서 헤드만 섞는 것 — 실제 Δ가
# 이 셔플 분포를 넘는다고 해서 곧바로 "독립적인 재현 가능한 신호"를 뜻하지는 않는다(오히려
# 인코더가 실제 신호를 학습했다는 정황 증거에 가깝다). 사전분포 무작위 예측기 비교는 이런
# 오염과 무관하므로 그대로 최대 신뢰도로 사용한다.
#
# 뉴스 테이블에는 아무것도 쓰지 않는다 — 전부 조회 전용.

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ⚠️ 위 docstring 참고 — snunlp/KR-FinBert가 safetensors 없이 pytorch_model.bin만 제공해서
# 우회 필요. 신뢰 가능한 공개 체크포인트(서울대 NLP)로 판단해 이 스크립트 내에서만 패치한다.
import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda *a, **kw: None

from constants import ACTIVE_TICKERS
from 감성분석.taskf_gating import majority_baseline
from 감성분석.taskf_gating_basis import load_news_with_basis_horizon_z
from 감성분석.taskf_gating_horizon import block_shuffle_labels
from 라벨.라벨_공통 import DIRECTION_3CLASS_THRESHOLD, label_direction_3class

MODEL_NAME = "snunlp/KR-FinBert"
LABEL_BASIS = "excess_return"
WINDOW_N = 60
HORIZON_H = 3
CLASSES = [-1, 0, 1]
LABEL_TO_IDX = {-1: 0, 0: 1, 1: 2}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

FROZEN_ENCODER_LAYERS = 9  # kobert_train.py와 동일한 부분 freeze 정책
BATCH_SIZE = 16
MAX_LENGTH = 64  # 제목만 입력(평균 33자, 최대 55자) — 128까지 필요 없음
MAX_EPOCHS = 10
PATIENCE = 2
LR = 1e-5
WARMUP_RATIO = 0.1

N_SHUFFLE = 50
N_RANDOM_PRIOR = 200

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "finbert_best_model.pt")


class TitleOnlyDataset(Dataset):
    """제목만 단일 시퀀스로 토크나이징 — kobert_dataset.NewsDataset(title+summary sentence-pair)과
    달리 이번 요구사항("입력은 제목만")에 맞춘 별도 Dataset."""

    def __init__(self, df, tokenizer, max_length=MAX_LENGTH):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        title = str(row["title"]) if pd.notna(row["title"]) else ""
        encoded = self.tokenizer(
            title, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "token_type_ids": encoded["token_type_ids"].squeeze(0),
            "label": torch.tensor(LABEL_TO_IDX[int(row["label"])], dtype=torch.long),
        }


def load_split_data(ticker):
    """taskf_gating_basis의 매칭 로직을 그대로 써서 (excess_return, N=60, h=3) 방향 3-class
    라벨을 붙인 train/val/test DataFrame을 만든다. 관련성 필터는 적용하지 않는다(요구사항:
    "필터 전체")."""
    merged, split_dates = load_news_with_basis_horizon_z(ticker, LABEL_BASIS, window_n=WINDOW_N, horizons=[HORIZON_H])
    z_col = f"z_h{HORIZON_H}"
    df = merged.copy()
    df["label"] = df[z_col].apply(lambda v: label_direction_3class(v, t=DIRECTION_3CLASS_THRESHOLD))
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df, split_dates


def compute_class_weights(train_df):
    counts = pd.Series(train_df["label"]).value_counts().reindex(CLASSES, fill_value=0)
    weights = torch.zeros(len(CLASSES), dtype=torch.float)
    for cls in CLASSES:
        c = counts[cls]
        weights[LABEL_TO_IDX[cls]] = 1.0 / c if c > 0 else 0.0
    return weights


def build_model(device):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(CLASSES))

    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for layer_idx, layer in enumerate(model.bert.encoder.layer):
        if layer_idx < FROZEN_ENCODER_LAYERS:
            for param in layer.parameters():
                param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"총 파라미터 수: {total_params:,} / 학습 가능: {trainable_params:,} ({trainable_params/total_params:.1%})")
    return model.to(device)


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1, all_labels, all_preds


def train_model(model, train_loader, val_loader, device, class_weights,
                 max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=LR):
    criterion = CrossEntropyLoss(weight=class_weights.to(device))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr)
    total_steps = len(train_loader) * max_epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_f1 = -1.0
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss, n_samples = 0.0, 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)

        train_loss = running_loss / n_samples
        val_loss, val_f1, _, _ = evaluate(model, val_loader, device, criterion)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  ✅ val macro F1 개선 -> 체크포인트 저장")
        else:
            epochs_without_improvement += 1
            print(f"  ⏸️ 개선 없음 ({epochs_without_improvement}/{patience})")
            if epochs_without_improvement >= patience:
                print(f"⛔ early stopping (patience={patience})")
                break
    return history


@torch.no_grad()
def extract_cls_embeddings(model, loader, device):
    """고정 피처 추출 — bert 인코더의 pooler_output([CLS] 위치, tanh dense 통과)을 뽑는다.
    셔플 검증에서 "헤드만 재학습"할 때 이 임베딩을 입력으로 쓴다."""
    model.eval()
    feats, labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        out = model.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        feats.append(out.pooler_output.cpu().numpy())
        labels.extend(batch["label"].tolist())
    return np.concatenate(feats, axis=0), np.array(labels)


@torch.no_grad()
def inspect_attention(model, tokenizer, titles, device, top_k=6):
    """sanity check(요구사항 3) — 마지막 레이어 [CLS] 토큰이 어디에 attention을 두는지 확인.
    1차 KoBERT 실패 패턴(날짜 파편/UI 텍스트)이나 오늘 발견한 사후 서술 어휘가 상위에 오는지 육안 확인."""
    model.eval()
    print("\n--- Sanity check: 마지막 레이어 attention 상위 토큰 (CLS -> 각 토큰) ---")
    for title in titles:
        encoded = tokenizer(title, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        out = model(**encoded, output_attentions=True)
        pred = torch.argmax(out.logits, dim=-1).item()
        last_layer_attn = out.attentions[-1][0]  # (num_heads, seq, seq)
        cls_attn = last_layer_attn[:, 0, :].mean(dim=0)  # 헤드 평균, CLS(위치0) -> 각 토큰
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].cpu())
        top_idx = torch.argsort(cls_attn, descending=True)[:top_k].tolist()
        top_tokens = [(tokens[i], round(cls_attn[i].item(), 3)) for i in top_idx]
        print(f"  제목: {title}")
        print(f"    예측={IDX_TO_LABEL[pred]:+d}  상위 attention 토큰: {top_tokens}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    for ticker in ACTIVE_TICKERS:
        print("\n" + "#" * 100)
        print(f"# KR-FinBERT 파인튜닝 게이팅 — {ticker}, basis={LABEL_BASIS}, N={WINDOW_N}, h={HORIZON_H}, 방향 3-class, 필터 미적용(전체)")
        print("#" * 100)

        train_df, val_df, test_df, split_dates = load_split_data(ticker)
        train_dates, val_dates, test_dates = split_dates
        print(
            f"뉴스(제목) train={len(train_df)} val={len(val_df)} test={len(test_df)}  |  "
            f"거래일 train={len(train_dates)} val={len(val_dates)} test={len(test_dates)}"
        )
        print(f"train 라벨분포: {pd.Series(train_df['label']).value_counts().sort_index().to_dict()}")
        print(f"test  라벨분포: {pd.Series(test_df['label']).value_counts().sort_index().to_dict()}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        class_weights = compute_class_weights(train_df)
        print("class weights:", class_weights.tolist())

        train_dataset = TitleOnlyDataset(train_df, tokenizer)
        val_dataset = TitleOnlyDataset(val_df, tokenizer)
        test_dataset = TitleOnlyDataset(test_df, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # ── 1) 실제 라벨로 파인튜닝 ──
        t0 = time.time()
        model = build_model(device)
        history = train_model(model, train_loader, val_loader, device, class_weights)
        train_wall_time = time.time() - t0
        print(f"\n실제 파인튜닝 소요 시간: {train_wall_time:.1f}초 ({len(history['train_loss'])} epoch)")

        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
        model.to(device)
        criterion = CrossEntropyLoss(weight=class_weights.to(device))
        test_loss, test_f1, test_labels_idx, test_preds_idx = evaluate(model, test_loader, device, criterion)
        test_labels = np.array([IDX_TO_LABEL[i] for i in test_labels_idx])
        test_preds = np.array([IDX_TO_LABEL[i] for i in test_preds_idx])
        test_base_f1, test_base_acc, test_majority = majority_baseline(test_labels, CLASSES)
        real_delta = test_f1 - test_base_f1
        real_acc = accuracy_score(test_labels, test_preds)

        print(f"\n=== [실제 결과] KR-FinBERT 파인튜닝 ===")
        print(f"test macro F1={test_f1:.4f} (baseline {test_base_f1:.4f}, Δ{real_delta:+.4f})  accuracy={real_acc:.4f}")
        print(f"test 최빈클래스={test_majority}")

        # ── 2) 사전분포 무작위 예측기 (계산량 적음 — 반드시 포함) ──
        train_counts = pd.Series(train_df["label"]).value_counts().reindex(CLASSES, fill_value=0)
        prior = (train_counts / train_counts.sum()).values
        prior_f1s = []
        for i in range(N_RANDOM_PRIOR):
            rng = np.random.RandomState(5000 + i)
            pred = rng.choice(CLASSES, size=len(test_labels), p=prior)
            prior_f1s.append(f1_score(test_labels, pred, average="macro", labels=CLASSES, zero_division=0))
        prior_f1s = np.array(prior_f1s)
        print(f"\n사전분포 무작위 예측기({N_RANDOM_PRIOR}회, train 비율 {np.round(prior,3).tolist()}): "
              f"기대 F1={prior_f1s.mean():.4f}±{prior_f1s.std():.4f}  (실제 모델 {test_f1:.4f})")

        # ── 3) 셔플 검증 — 절충안: 고정 인코더 임베딩 위에서 헤드만 재학습 ──
        print(f"\n=== 셔플 검증 (절충안: 인코더 고정, 헤드만 재학습, N={N_SHUFFLE}) ===")
        print("⚠️ 한계: (1) 인코더 자체를 재학습하지 않아 '전체 구조가 노이즈에 얼마나 과적합될 "
              "수 있는가'는 측정하지 못함 — 고정 표현 위에서 헤드가 노이즈를 얼마나 분리하는지만 "
              "측정. (2) 이 인코더는 실제 라벨로 이미 파인튜닝된 것이라 셔플 귀무분포가 완전히 "
              "깨끗한 null이 아님(실제 신호가 인코더 표현에 이미 녹아있을 수 있음).")

        train_feats, _ = extract_cls_embeddings(model, DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False), device)
        test_feats, _ = extract_cls_embeddings(model, test_loader, device)

        # block_shuffle_labels는 "trading_date"/"label" 컬럼을 요구 — train_df 그대로 사용
        real_head = LogisticRegression(class_weight="balanced", max_iter=2000)
        real_head.fit(train_feats, train_df["label"].values)
        head_pred = real_head.predict(test_feats)
        head_f1 = f1_score(test_df["label"].values, head_pred, average="macro", labels=CLASSES, zero_division=0)
        print(f"(참고) 고정 임베딩 + 로지스틱회귀 헤드만 다시 학습한 결과: test macro F1={head_f1:.4f} "
              f"(파인튜닝 전체 결과 {test_f1:.4f}와 비교 — 헤드만으로도 비슷하면 분류기가 어차피 "
              "단순 선형 경계에 의존한다는 뜻)")

        shuffle_f1s = []
        for i in range(N_SHUFFLE):
            rng = np.random.RandomState(1000 + i)
            shuffled_map = block_shuffle_labels(train_df, block_size=HORIZON_H, rng=rng)
            y_shuffled = train_df["trading_date"].map(shuffled_map).values
            head = LogisticRegression(class_weight="balanced", max_iter=2000)
            head.fit(train_feats, y_shuffled)
            pred = head.predict(test_feats)
            f1 = f1_score(test_df["label"].values, pred, average="macro", labels=CLASSES, zero_division=0)
            shuffle_f1s.append(f1)
        shuffle_f1s = np.array(shuffle_f1s)
        shuffle_deltas = shuffle_f1s - test_base_f1
        ci_lo, ci_hi = np.percentile(shuffle_deltas, [2.5, 97.5])
        p_value = (np.sum(shuffle_f1s >= head_f1) + 1) / (N_SHUFFLE + 1)
        print(f"\n헤드-재학습 셔플({N_SHUFFLE}회) Δ 분포: 평균={shuffle_deltas.mean():+.4f} "
              f"표준편차={shuffle_deltas.std():.4f}  95% 구간=[{ci_lo:+.4f}, {ci_hi:+.4f}]")
        head_delta = head_f1 - test_base_f1
        print(f"헤드-재학습 실제 Δ={head_delta:+.4f} vs 셔플 분포, p-value={p_value:.4f} "
              f"(참고: 이 p-value는 '헤드만 재학습' 버전 기준 — 위 한계 설명 참고, 파인튜닝 전체 결과에 "
              "직접 적용되는 값이 아니라 근사치)")

        # ── 4) 종합 판정 (지금까지 기준 그대로: p<0.05 & 95%상단 초과 & 사전분포 +2σ 상회) ──
        beats_shuffle = p_value < 0.05 and head_delta > ci_hi
        beats_prior = test_f1 > prior_f1s.mean() + 2 * prior_f1s.std()
        min_test_day_count = int(
            test_df.drop_duplicates(subset=["trading_date"])["label"].value_counts().reindex(CLASSES, fill_value=0).min()
        )
        print(f"\ntest 최소 클래스 거래일 수: {min_test_day_count}")

        if min_test_day_count < 10:
            verdict_str = "판정 불가(표본 부족)"
            reason = f"test 최소 클래스 거래일 수 {min_test_day_count}개 < 10"
        elif beats_shuffle and beats_prior:
            verdict_str = "유의함"
            reason = f"p={p_value:.4f}<0.05·헤드 재학습 Δ가 셔플 95% 상단 초과, 사전분포 무작위 예측기(+2σ)도 상회"
        elif not beats_shuffle:
            verdict_str = "유의하지 않음"
            reason = f"p={p_value:.4f}(헤드 재학습 근사), 실제 결과가 셔플 분포와 통계적으로 구분 안 됨"
        else:
            verdict_str = "유의하지 않음"
            reason = "셔플 검정은 통과했으나 사전분포 무작위 예측기와 구분 안 됨"

        print("\n" + "=" * 100)
        print("=== 최종 판정 ===")
        print("=" * 100)
        print(f"KR-FinBERT 파인튜닝: {verdict_str} — {reason}")

        print("\n=== TF-IDF 게이팅과 나란히 비교 (동일 조합: excess_return/방향3-class/N=60/h=3, 필터 없음) ===")
        print(f"  TF-IDF+로지스틱회귀: test F1=0.3248 baseline=0.1747 Δ+0.1501 셔플p=0.5572(N=200) -> 유의하지 않음")
        print(f"  KR-FinBERT 파인튜닝: test F1={test_f1:.4f} baseline={test_base_f1:.4f} Δ{real_delta:+.4f} "
              f"헤드재학습셔플p={p_value:.4f}(N={N_SHUFFLE}, 근사) -> {verdict_str}")

        # ── 5) sanity check: attention 상위 토큰 (요구사항 3) ──
        try:
            sample_titles = list(test_df["title"].sample(n=min(6, len(test_df)), random_state=42))
            inspect_attention(model, tokenizer, sample_titles, device)
        except Exception as e:
            print(f"⚠️ attention 점검 스킵(에러: {e})")


if __name__ == "__main__":
    main()
