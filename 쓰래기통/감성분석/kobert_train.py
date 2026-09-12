import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from 감성분석.kobert_dataset import (
    MODEL_NAME,
    LABEL_MAP,
    load_labeled_news,
    split_by_ratio,
    compute_class_weights,
    NewsDataset,
)

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
HISTORY_PLOT_PATH = os.path.join(CHECKPOINT_DIR, "training_history.png")

FROZEN_ENCODER_LAYERS = 9  # 인덱스 0~8 고정, 9/10/11 + pooler + classifier만 학습
BATCH_SIZE = 16  # RTX 4070(12GB) 기준 값. VRAM 부족하면 8로 낮추고, 여유 있으면 32까지 올려도 됨
MAX_LENGTH = 128
MAX_EPOCHS = 10
PATIENCE = 2
LR = 1e-5
WARMUP_RATIO = 0.1


def build_model(device):
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABEL_MAP))

    for param in model.bert.embeddings.parameters():
        param.requires_grad = False

    for layer_idx, layer in enumerate(model.bert.encoder.layer):
        if layer_idx < FROZEN_ENCODER_LAYERS:
            for param in layer.parameters():
                param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"총 파라미터 수: {total_params:,}")
    print(f"학습 가능한 파라미터 수: {trainable_params:,} ({trainable_params / total_params:.1%})")

    return model.to(device)


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).logits
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1, all_labels, all_preds


def train(model, train_loader, val_loader, device, class_weights,
          max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=LR):
    criterion = CrossEntropyLoss(weight=class_weights.to(device))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr)

    total_steps = len(train_loader) * max_epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).logits
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
            print(f"  ✅ val macro F1 개선 -> 체크포인트 저장 ({BEST_MODEL_PATH})")
        else:
            epochs_without_improvement += 1
            print(f"  ⏸️ 개선 없음 ({epochs_without_improvement}/{patience})")
            if epochs_without_improvement >= patience:
                print(f"⛔ early stopping (patience={patience})")
                break

    return history


def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, history["train_loss"], label="train_loss", marker="o")
    ax1.plot(epochs, history["val_loss"], label="val_loss", marker="o")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(epochs, history["val_f1"], label="val_macro_f1", marker="o", color="green")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("macro F1")
    ax2.set_title("Validation Macro F1")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(HISTORY_PLOT_PATH)
    plt.close(fig)
    print(f"📈 학습 곡선 저장: {HISTORY_PLOT_PATH}")


def run_test_evaluation(model, test_loader, device, class_weights):
    criterion = CrossEntropyLoss(weight=class_weights.to(device))
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)

    test_loss, test_f1, labels, preds = evaluate(model, test_loader, device, criterion)
    print(f"\n[TEST] loss={test_loss:.4f} macro_f1={test_f1:.4f}")

    print("class index -> 실제 라벨(-2~2):", INV_LABEL_MAP)
    # test set에 5개 클래스가 전부 등장하지 않을 수도 있으므로 labels를 명시해서
    # classification_report가 target_names 길이 불일치로 죽는 것을 방지
    class_indices = list(range(len(LABEL_MAP)))
    target_names = [f"class {i} (score={INV_LABEL_MAP[i]})" for i in class_indices]
    print(classification_report(labels, preds, labels=class_indices, target_names=target_names, zero_division=0))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    df = load_labeled_news()
    train_df, val_df, test_df = split_by_ratio(df)
    print(f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    class_weights = compute_class_weights(train_df)
    print("class weights:", class_weights)

    train_dataset = NewsDataset(train_df, tokenizer, max_length=MAX_LENGTH)
    val_dataset = NewsDataset(val_df, tokenizer, max_length=MAX_LENGTH)
    test_dataset = NewsDataset(test_df, tokenizer, max_length=MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(device)

    history = train(model, train_loader, val_loader, device, class_weights)
    plot_history(history)

    run_test_evaluation(model, test_loader, device, class_weights)