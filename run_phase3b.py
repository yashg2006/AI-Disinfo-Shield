"""
Phase 3B -- ClaimSeg DeBERTa-v3-base
Manual PyTorch training loop (no Trainer API / no accelerate dependency)
Binary claim segmentation: 1=factual claim, 0=non-claim
"""
import os, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, classification_report

os.makedirs("phase3_models/claimseg_deberta", exist_ok=True)

LOCAL_MODEL_PATH = "models/deberta-v3-base"
DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------
# STEP 3B-1: Load and prepare data
# ---------------------------------------------------------------
print("\nSTEP 3B-1: Loading data...")
train_df = pd.read_csv("phase1_output/claims_train.csv")
val_df   = pd.read_csv("phase1_output/claims_val.csv")
print(f"  Train rows: {len(train_df)}  |  Val rows: {len(val_df)}")

MAX_TRAIN, MAX_VAL = 4000, 800
np.random.seed(42)
if len(train_df) > MAX_TRAIN:
    train_df = train_df.sample(MAX_TRAIN, random_state=42).reset_index(drop=True)
if len(val_df) > MAX_VAL:
    val_df = val_df.sample(MAX_VAL, random_state=42).reset_index(drop=True)
print(f"  Sampled -> train: {len(train_df)}  |  val: {len(val_df)}")

def build_positives(df):
    return df["text"].dropna().astype(str).tolist(), [1] * len(df)

def build_negatives(df):
    opinion_starts = ("I ", "We ", "You ", "This is", "It is")
    texts, labels = [], []
    for text in df["text"].dropna().astype(str):
        for sent in text.split(". "):
            sent = sent.strip()
            if not sent:
                continue
            if len(sent.split()) < 8 or any(sent.startswith(op) for op in opinion_starts):
                texts.append(sent)
                labels.append(0)
    return texts, labels

def balance(pos_t, pos_l, neg_t, neg_l):
    if len(neg_t) == 0:
        neg_t = ["No comment.", "Opinion only.", "Unclear statement."] * (len(pos_t) // 3 + 1)
        neg_l = [0] * len(neg_t)
    while len(neg_t) < len(pos_t):
        neg_t = neg_t + neg_t
        neg_l = neg_l + neg_l
    neg_t, neg_l = neg_t[:len(pos_t)], neg_l[:len(pos_t)]
    combined = list(zip(pos_t + neg_t, pos_l + neg_l))
    np.random.seed(42)
    np.random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)

train_texts, train_labels = balance(*build_positives(train_df), *build_negatives(train_df))
val_texts,   val_labels   = balance(*build_positives(val_df),   *build_negatives(val_df))
train_texts = [t[:512] for t in train_texts]
val_texts   = [t[:512] for t in val_texts]

print(f"  Final dataset -> train: {len(train_texts)}  |  val: {len(val_texts)}")
print(f"  Train labels: 0={train_labels.count(0)}  1={train_labels.count(1)}")
print(f"  Val   labels: 0={val_labels.count(0)}    1={val_labels.count(1)}")

# ---------------------------------------------------------------
# STEP 3B-2: Tokenise
# ---------------------------------------------------------------
print("\nSTEP 3B-2: Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)

print("  Tokenising train set...")
train_enc = tokenizer(train_texts, max_length=256, truncation=True, padding="max_length")
print("  Tokenising val set...")
val_enc   = tokenizer(val_texts,   max_length=256, truncation=True, padding="max_length")

# ---------------------------------------------------------------
# STEP 3B-3: PyTorch Dataset
# ---------------------------------------------------------------
class ClaimDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = ClaimDataset(train_enc, train_labels)
val_dataset   = ClaimDataset(val_enc,   val_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False)

# ---------------------------------------------------------------
# STEP 3B-4: Manual training loop
# ---------------------------------------------------------------
print("\nSTEP 3B-4: Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    LOCAL_MODEL_PATH, num_labels=2
)
model = model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
loss_fn   = nn.CrossEntropyLoss()

NUM_EPOCHS  = 3
best_f1     = -1.0
best_state  = None

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(DEVICE)
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="binary")
    avg_loss = total_loss / len(loader)
    return avg_loss, acc, f1

print(f"\nStarting training ({NUM_EPOCHS} epochs on {DEVICE})...")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0.0
    for step, batch in enumerate(train_loader):
        labels = batch.pop("labels").to(DEVICE)
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_train_loss += loss.item()
        if (step + 1) % 50 == 0:
            print(f"  Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_loader)
    val_loss, val_acc, val_f1 = evaluate(model, val_loader)
    print(f"\n  [Epoch {epoch+1}/{NUM_EPOCHS}]  train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1    = val_f1
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  >> New best model saved (F1={best_f1:.4f})")

# ---------------------------------------------------------------
# STEP 3B-5: Evaluate best model and save
# ---------------------------------------------------------------
print("\nSTEP 3B-5: Loading best model and final evaluation...")
if best_state:
    model.load_state_dict(best_state)

val_loss, val_acc, val_f1 = evaluate(model, val_loader)
print(f"  Val Accuracy : {val_acc:.4f}")
print(f"  Val F1       : {val_f1:.4f}")

# Full classification report on val set
model.eval()
all_preds, all_labels_list = [], []
with torch.no_grad():
    for batch in val_loader:
        labels = batch.pop("labels").to(DEVICE)
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}
        preds  = torch.argmax(model(**inputs).logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels_list.extend(labels.cpu().numpy())
print("\nClassification Report:")
print(classification_report(all_labels_list, all_preds, target_names=["NON-CLAIM", "CLAIM"]))

model.save_pretrained("phase3_models/claimseg_deberta")
tokenizer.save_pretrained("phase3_models/claimseg_deberta")
print("ClaimSeg model saved.")

# ---------------------------------------------------------------
# Inference on 5 example sentences
# ---------------------------------------------------------------
print("\n--- Inference on example sentences ---")
examples = [
    "Russia launched missiles at Kyiv on September 14 2025.",
    "The situation is very complicated.",
    "Ukraine received 50 Leopard tanks from Germany in 2023.",
    "I think this is very unfair to everyone.",
    "Israel conducted airstrikes on 3 locations in Gaza.",
]
model.eval()
for sent in examples:
    enc = tokenizer(sent[:512], max_length=256, truncation=True,
                    padding="max_length", return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        pred = torch.argmax(model(**enc).logits, dim=-1).item()
    label = "CLAIM (1)" if pred == 1 else "NON-CLAIM (0)"
    print(f"  [{label}]  \"{sent}\"")

print("\n=== Phase 3B complete! ===")
