"""
Fine-tune HyeBERT for Armenian CSO government criticism classification.
Unfreezes last 2 transformer layers + adds classification head.
Compares against frozen-embedding baseline.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "aking11/hyebert"
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5
FREEZE_LAYERS = 10  # freeze first 10 of 12 encoder layers (unfreeze last 2)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Load data ───────────────────────────────────────────────────────────
df = pd.read_csv("/Users/albertananyan/Downloads/cso_classification_project/data/coded_corpus_combined.csv")
texts = df["full_text"].astype(str).tolist()
labels = df["crit_armenian_human"].astype(int).values
print(f"Loaded {len(texts)} documents ({sum(labels)} critical, {len(labels)-sum(labels)} non-critical)")

# ── Tokenizer ───────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class CSODataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding=True,
                                   max_length=max_length, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx]
        }

# ── 5-fold CV for both methods ──────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ft_preds_all = np.zeros(len(labels))
fe_preds_all = np.zeros(len(labels))

for fold, (train_idx, val_idx) in enumerate(cv.split(texts, labels)):
    print(f"\n{'='*60}")
    print(f"FOLD {fold+1}/5")
    print(f"{'='*60}")

    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    # ── METHOD 1: Fine-tuned BERT ────────────────────────────────────
    print("\n[Fine-tuning HyeBERT]")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # Freeze first N encoder layers
    for name, param in model.named_parameters():
        if "encoder.layer" in name:
            layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
            if layer_num < FREEZE_LAYERS:
                param.requires_grad = False
        elif "embeddings" in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    model.to(DEVICE)

    train_dataset = CSODataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = CSODataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01
    )

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE)
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{EPOCHS} - loss: {avg_loss:.4f}")

    # Evaluate fine-tuned model
    model.eval()
    fold_preds = []
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE)
            )
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            fold_preds.extend(preds)
    ft_preds_all[val_idx] = fold_preds
    ft_acc = accuracy_score(val_labels, fold_preds)
    ft_f1 = f1_score(val_labels, fold_preds)
    print(f"  Fine-tuned fold acc: {ft_acc:.3f}, F1: {ft_f1:.3f}")

    # ── METHOD 2: Frozen embeddings + LogReg (baseline) ──────────────
    print("\n[Frozen embeddings + LogReg]")
    bert_base = AutoModel.from_pretrained(MODEL_NAME)
    bert_base.eval()
    bert_base.to(DEVICE)

    def get_embeddings(text_list, batch_size=16):
        embeddings = []
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                             max_length=MAX_LENGTH, padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = bert_base(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls)
            if (i // batch_size) % 5 == 0:
                print(f"  Embedding batch {i//batch_size + 1}/{(len(text_list)-1)//batch_size + 1}")
        return np.vstack(embeddings)

    X_train = get_embeddings(train_texts)
    X_val = get_embeddings(val_texts)

    lr_model = LogisticRegression(max_iter=2000, random_state=42)
    lr_model.fit(X_train, train_labels)
    fe_fold_preds = lr_model.predict(X_val)
    fe_preds_all[val_idx] = fe_fold_preds
    fe_acc = accuracy_score(val_labels, fe_fold_preds)
    fe_f1 = f1_score(val_labels, fe_fold_preds)
    print(f"  Frozen+LR fold acc: {fe_acc:.3f}, F1: {fe_f1:.3f}")

    # Clean up GPU memory
    del model, bert_base
    torch.mps.empty_cache() if DEVICE.type == "mps" else None

# ── Final comparison ────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL 5-FOLD CV RESULTS")
print("="*60)

for name, preds in [("Fine-tuned HyeBERT", ft_preds_all), ("Frozen + LogReg", fe_preds_all)]:
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    print(f"\n{name}:")
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(classification_report(labels, preds, target_names=["Non-critical", "Critical"]))

# ── Save the improvement delta ──────────────────────────────────────────
ft_acc = accuracy_score(labels, ft_preds_all)
fe_acc = accuracy_score(labels, fe_preds_all)
ft_f1 = f1_score(labels, ft_preds_all)
fe_f1 = f1_score(labels, fe_preds_all)
print(f"\nImprovement: Accuracy +{(ft_acc-fe_acc)*100:.1f}pp, F1 +{(ft_f1-fe_f1)*100:.1f}pp")

# If fine-tuning is better, train final model on all data for corpus classification
if ft_f1 > fe_f1:
    print("\n" + "="*60)
    print("Fine-tuning wins! Training final model on ALL labeled data...")
    print("="*60)

    final_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    for name, param in final_model.named_parameters():
        if "encoder.layer" in name:
            layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
            if layer_num < FREEZE_LAYERS:
                param.requires_grad = False
        elif "embeddings" in name:
            param.requires_grad = False

    final_model.to(DEVICE)
    full_dataset = CSODataset(texts, labels.tolist(), tokenizer, MAX_LENGTH)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(
        [p for p in final_model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01
    )

    for epoch in range(EPOCHS):
        final_model.train()
        total_loss = 0
        for batch in full_loader:
            optimizer.zero_grad()
            outputs = final_model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE)
            )
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            optimizer.step()
            total_loss += outputs.loss.item()
        print(f"  Final model epoch {epoch+1}/{EPOCHS} - loss: {total_loss/len(full_loader):.4f}")

    # Save the fine-tuned model
    save_path = "/Users/albertananyan/Downloads/bert_output/hyebert_finetuned"
    final_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"  Saved fine-tuned model to {save_path}")
else:
    print("\nFrozen embeddings performed better or equal. No fine-tuning advantage.")
