import os
import re
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report

# For saving vectorizer locally
import joblib


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensures more reproducible results (may slightly reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def show_class_distribution(name, labels):
    total = len(labels)
    spam = int(np.sum(labels))
    ham = total - spam
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Total : {total}")
    print(f"Ham  : {ham} ({ham/total*100:.2f}%)")
    print(f"Spam : {spam} ({spam/total*100:.2f}%)")


def to_tensor(x_sparse, y_series):
    # Note: converting TF-IDF sparse matrix to dense can be memory heavy.
    # This is fine for this dataset size but beware for larger datasets.
    x = torch.tensor(x_sparse.toarray(), dtype=torch.float32)
    y = torch.tensor(y_series.values, dtype=torch.long)
    return x, y


class SpamNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x, return_hidden: bool = False):
        hidden = self.relu(self.fc1(x))
        output = self.fc2(hidden)
        if return_hidden:
            return output, hidden
        return output


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    logits = model(X)
    preds = torch.argmax(logits, dim=1)
    acc = accuracy_score(y.cpu(), preds.cpu())
    f1 = f1_score(y.cpu(), preds.cpu())
    return acc, f1, preds


def train_model(model, train_loader, X_val, y_val, optimizer, criterion, device, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        avg_loss = running_loss / len(train_loader.dataset)

        val_acc, val_f1, _ = evaluate(model, X_val, y_val)
        print(f"Epoch {epoch+1}/{epochs} | Loss {avg_loss:.4f} | Val Acc {val_acc:.4f} | Val F1 {val_f1:.4f}")


def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # -------------------------
    # 1) Load EMAIL dataset
    # -------------------------
    email_dataset = load_dataset("iamthearafatkhan/email-spam-dataset")["train"]
    df_email = pd.DataFrame(email_dataset)

    df_email["clean_text"] = df_email["text"].apply(clean_text)
    X_email = df_email["clean_text"]
    y_email = df_email["label"]

    show_class_distribution("EMAIL DATASET (FULL)", y_email)

    X_train_email, X_temp_email, y_train_email, y_temp_email = train_test_split(
        X_email, y_email, test_size=0.4, stratify=y_email, random_state=42
    )

    X_val_email, X_test_email, y_val_email, y_test_email = train_test_split(
        X_temp_email, y_temp_email, test_size=0.5, stratify=y_temp_email, random_state=42
    )

    show_class_distribution("EMAIL TRAIN", y_train_email)
    show_class_distribution("EMAIL VALIDATION", y_val_email)
    show_class_distribution("EMAIL TEST", y_test_email)

    # -------------------------
    # 2) TF-IDF
    # -------------------------
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train_email)
    X_val_tfidf = vectorizer.transform(X_val_email)
    X_test_tfidf = vectorizer.transform(X_test_email)

    X_train_tensor, y_train_tensor = to_tensor(X_train_tfidf, y_train_email)
    X_val_tensor, y_val_tensor = to_tensor(X_val_tfidf, y_val_email)
    X_test_tensor, y_test_tensor = to_tensor(X_test_tfidf, y_test_email)

    # Move val/test to device once (training batches are moved inside the loop)
    X_val_tensor = X_val_tensor.to(device)
    y_val_tensor = y_val_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)

    # -------------------------
    # 3) Model + training setup
    # -------------------------
    model = SpamNN(X_train_tensor.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 256
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # -------------------------
    # 4) Train
    # -------------------------
    start = time.time()
    train_model(
        model=model,
        train_loader=train_loader,
        X_val=X_val_tensor,
        y_val=y_val_tensor,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=10,
    )
    print("Training time:", round(time.time() - start, 2), "s")

    # -------------------------
    # 5) Evaluate on EMAIL test
    # -------------------------
    test_acc, test_f1, preds_email = evaluate(model, X_test_tensor, y_test_tensor)

    print("\n=== BASELINE – EMAILS ===")
    print("Accuracy:", test_acc)
    print("F1-score:", test_f1)
    print(classification_report(y_test_tensor.cpu(), preds_email.cpu()))

    # -------------------------
    # 6) Load SMS dataset (Domain B) and evaluate
    # -------------------------
    sms_dataset = load_dataset("ucirvine/sms_spam")["train"]
    df_sms = pd.DataFrame(sms_dataset)

    # Some versions of this dataset use different column names.
    # Try common ones:
    if "sms" in df_sms.columns:
        text_col = "sms"
    else:
        raise ValueError(f"Couldn't find text column in SMS dataset. Columns: {df_sms.columns.tolist()}")

    df_sms["clean_text"] = df_sms[text_col].apply(clean_text)
    X_sms = df_sms["clean_text"]
    y_sms = df_sms["label"]

    show_class_distribution("SMS DATASET (FULL)", y_sms)

    X_sms_pool, X_sms_test, y_sms_pool, y_sms_test = train_test_split(
        X_sms, y_sms, test_size=0.2, stratify=y_sms, random_state=42
    )

    show_class_distribution("SMS TEST", y_sms_test)
    show_class_distribution("SMS ACTIVE LEARNING POOL", y_sms_pool)

    X_sms_test_tfidf = vectorizer.transform(X_sms_test)
    X_sms_test_tensor = torch.tensor(X_sms_test_tfidf.toarray(), dtype=torch.float32).to(device)
    y_sms_test_tensor = torch.tensor(y_sms_test.values, dtype=torch.long).to(device)

    sms_acc, sms_f1, preds_sms = evaluate(model, X_sms_test_tensor, y_sms_test_tensor)

    print("\n=== BASELINE – SMS (DOMAIN B) ===")
    print("Accuracy:", sms_acc)
    print("F1-score:", sms_f1)
    print(classification_report(y_sms_test_tensor.cpu(), preds_sms.cpu()))

    # Hidden layer
    with torch.no_grad():
        _, hidden_sms = model(X_sms_test_tensor, return_hidden=True)
    print("Hidden layer shape:", tuple(hidden_sms.shape))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", n_params)

   

if __name__ == "__main__":
    main()
