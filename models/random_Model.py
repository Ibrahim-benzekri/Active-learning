# Filename suggestion (IMPORTANT): retrain_email_plus_sms_logs.py
# Do NOT name this file random.py (it breaks imports).

import os
import re
import time
import random
import math
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from datasets import load_dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# -----------------------------
# Repro / preprocessing
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# Logging + CI
# -----------------------------
def log_txt(path: str, line: str = ""):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def mean_ci95_t(values):
    """
    95% CI with Student t (small n). For n=3 => df=2 => tcrit ~ 4.303
    returns: mean, ci_lo, ci_hi
    """
    vals = np.array(values, dtype=float)
    n = len(vals)
    mean = float(vals.mean())
    if n < 2:
        return mean, mean, mean
    sd = float(vals.std(ddof=1))
    se = sd / math.sqrt(n)
    tcrit = 4.303 if n == 3 else 1.96
    half = tcrit * se
    return mean, mean - half, mean + half


# -----------------------------
# Model
# -----------------------------
class SpamNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def build_model(input_dim: int, device: str):
    return SpamNN(input_dim).to(device)


# -----------------------------
# Data helpers
# -----------------------------
def make_balanced_loader(X_cpu: torch.Tensor, y_cpu: torch.Tensor, batch_size=256):
    """
    WeightedRandomSampler to mitigate class imbalance.
    X_cpu and y_cpu must be CPU tensors.
    """
    y_np = y_cpu.numpy()
    class_counts = np.bincount(y_np, minlength=2)  # [ham, spam]
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[y_np]

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    ds = TensorDataset(X_cpu, y_cpu)
    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    return loader, class_counts

def train_epochs(model, optimizer, criterion, train_loader, device, epochs=8):
    model.train()
    start = time.time()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    return time.time() - start

@torch.no_grad()
def eval_on_sms_test(model, X_test_device, y_test_device):
    model.eval()
    logits = model(X_test_device)
    preds = torch.argmax(logits, dim=1)

    y_true = y_test_device.detach().cpu().numpy()
    y_pred = preds.detach().cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    f1b = f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    return {
        "sms_test_accuracy": acc,
        "sms_test_precision_spam": prec,
        "sms_test_recall_spam": rec,
        "sms_test_f1_spam": f1b,
        "sms_test_f1_macro": f1_macro,
        "sms_test_f1_weighted": f1_weighted,
        "report": report,
    }


# -----------------------------
# Main experiment
# -----------------------------
def main():
    # Experiment config
    repeats = 3          # repeat selection-training-evaluation 3 times
    iters = 5            # ratios: 20%,40%,60%,80%,100%
    sms_test_size = 0.2  # fixed SMS test
    epochs = 8
    lr = 1e-3
    batch_size = 256

    base_seed = 42

    set_seed(base_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    os.makedirs("artifacts", exist_ok=True)
    logs_path = os.path.join("artifacts", "retrain_email_plus_sms_repeats_logs.txt")

    # fresh log file each run (optional). Comment out if you want append across runs.
    with open(logs_path, "w", encoding="utf-8") as f:
        f.write("")

    log_txt(logs_path, "===== Retrain from scratch: EMAIL(train) + SMS_subset(train) =====")
    log_txt(logs_path, f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_txt(logs_path, f"device={device} | repeats={repeats} | iters={iters} | epochs={epochs} | lr={lr} | batch_size={batch_size}")
    log_txt(logs_path, "TF-IDF is fit on EMAIL train only (stable across repeats).")
    log_txt(logs_path, "Evaluation is on SMS test only (fixed split, 20%).")
    log_txt(logs_path, "Sampler: WeightedRandomSampler (class balancing on combined train set).")
    log_txt(logs_path, "label_mapping: 0=ham, 1=spam")
    log_txt(logs_path, "")

    # -------------------------
    # Load datasets once (stable splits for fairness)
    # -------------------------
    print("Loading EMAIL dataset...")
    email_dataset = load_dataset("iamthearafatkhan/email-spam-dataset")["train"]
    df_email = pd.DataFrame(email_dataset)
    df_email["clean_text"] = df_email["text"].apply(clean_text)
    X_email = df_email["clean_text"]
    y_email = df_email["label"]

    # Domain A train fixed for all repeats
    X_email_train, X_email_tmp, y_email_train, y_email_tmp = train_test_split(
        X_email, y_email, test_size=0.4, stratify=y_email, random_state=base_seed
    )
    # not used further, but keep deterministic split
    _X_email_val, _X_email_test, _y_email_val, _y_email_test = train_test_split(
        X_email_tmp, y_email_tmp, test_size=0.5, stratify=y_email_tmp, random_state=base_seed
    )

    print("Loading SMS dataset...")
    sms_dataset = load_dataset("ucirvine/sms_spam")["train"]
    df_sms = pd.DataFrame(sms_dataset)
    if "sms" in df_sms.columns:
        text_col = "sms"
    elif "text" in df_sms.columns:
        text_col = "text"
    elif "message" in df_sms.columns:
        text_col = "message"
    else:
        raise ValueError(f"SMS text column not found. Columns: {df_sms.columns.tolist()}")

    df_sms["clean_text"] = df_sms[text_col].apply(clean_text)
    X_sms = df_sms["clean_text"]
    y_sms = df_sms["label"]

    # fixed SMS split for all repeats
    X_sms_train, X_sms_test, y_sms_train, y_sms_test = train_test_split(
        X_sms, y_sms, test_size=sms_test_size, stratify=y_sms, random_state=base_seed
    )

    # -------------------------
    # Fit TF-IDF once on EMAIL train only
    # -------------------------
    print("Fitting TF-IDF on EMAIL train...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_email_train_tfidf = vectorizer.fit_transform(X_email_train)

    X_email_train_tensor = torch.tensor(X_email_train_tfidf.toarray(), dtype=torch.float32)  # CPU
    y_email_train_tensor = torch.tensor(y_email_train.values, dtype=torch.long)              # CPU
    input_dim = X_email_train_tensor.shape[1]

    # SMS test tensors to device once
    X_sms_test_tfidf = vectorizer.transform(X_sms_test)
    X_sms_test_tensor = torch.tensor(X_sms_test_tfidf.toarray(), dtype=torch.float32).to(device)
    y_sms_test_tensor = torch.tensor(y_sms_test.values, dtype=torch.long).to(device)

    # Prepare SMS train arrays (for chunking)
    X_sms_train_arr = X_sms_train.reset_index(drop=True)
    y_sms_train_arr = y_sms_train.reset_index(drop=True)

    # Keep F1(spam) for summary (iteration -> list of 3 values)
    all_f1_spam = {it: [] for it in range(1, iters + 1)}
    all_acc = {it: [] for it in range(1, iters + 1)}
    all_prec = {it: [] for it in range(1, iters + 1)}
    all_rec = {it: [] for it in range(1, iters + 1)}

    # -------------------------
    # Repeats
    # -------------------------
    for rep in range(1, repeats + 1):
        rep_seed = base_seed + rep
        set_seed(rep_seed)

        log_txt(logs_path, f"==================== REPEAT {rep}/{repeats} | seed={rep_seed} ====================")

        # 5 stratified chunks (20% each) with a rep-dependent random_state
        skf = StratifiedKFold(n_splits=iters, shuffle=True, random_state=rep_seed)
        chunks = []
        for _, idx_chunk in skf.split(X_sms_train_arr, y_sms_train_arr):
            chunks.append(idx_chunk)

        used_indices = []

        for it in range(1, iters + 1):
            # cumulative subset: 20%,40%,...,100%
            used_indices.extend(chunks[it - 1])
            used_sorted = np.array(sorted(set(used_indices)))

            X_sms_it = X_sms_train_arr.iloc[used_sorted]
            y_sms_it = y_sms_train_arr.iloc[used_sorted]

            # vectorize SMS subset (CPU)
            X_sms_it_tfidf = vectorizer.transform(X_sms_it)
            X_sms_it_tensor = torch.tensor(X_sms_it_tfidf.toarray(), dtype=torch.float32)  # CPU
            y_sms_it_tensor = torch.tensor(y_sms_it.values, dtype=torch.long)              # CPU

            # combined training set (CPU)
            X_comb = torch.cat([X_email_train_tensor, X_sms_it_tensor], dim=0)
            y_comb = torch.cat([y_email_train_tensor, y_sms_it_tensor], dim=0)

            train_loader, class_counts = make_balanced_loader(X_comb, y_comb, batch_size=batch_size)

            # retrain from scratch at each iteration
            set_seed(rep_seed)  # same init within a repeat for all iterations (optional but consistent)
            model = build_model(input_dim=input_dim, device=device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            msg_head = (
                f"[rep {rep}/{repeats} | iter {it}/{iters} | SMS_frac={it/iters:.1f}] "
                f"TRAIN: EMAIL={len(y_email_train_tensor)} + SMS_subset={len(y_sms_it_tensor)} "
                f"=> total={len(y_comb)} | train_counts ham/spam={class_counts.tolist()}"
            )
            print(msg_head)
            log_txt(logs_path, msg_head)

            train_time = train_epochs(model, optimizer, criterion, train_loader, device=device, epochs=epochs)

            metrics = eval_on_sms_test(model, X_sms_test_tensor, y_sms_test_tensor)

            # store for summary
            all_f1_spam[it].append(metrics["sms_test_f1_spam"])
            all_acc[it].append(metrics["sms_test_accuracy"])
            all_prec[it].append(metrics["sms_test_precision_spam"])
            all_rec[it].append(metrics["sms_test_recall_spam"])

            # log block
            log_txt(
                logs_path,
                f"  train_time_s={train_time:.2f} | "
                f"SMS_TEST acc={metrics['sms_test_accuracy']:.4f} "
                f"prec_spam={metrics['sms_test_precision_spam']:.4f} "
                f"recall_spam={metrics['sms_test_recall_spam']:.4f} "
                f"f1_spam={metrics['sms_test_f1_spam']:.4f} "
                f"f1_macro={metrics['sms_test_f1_macro']:.4f} "
                f"f1_weighted={metrics['sms_test_f1_weighted']:.4f}"
            )
            log_txt(logs_path, "  Classification report:")
            for line in metrics["report"].splitlines():
                log_txt(logs_path, "    " + line)
            log_txt(logs_path, "")  # blank line

    # -------------------------
    # Summary (mean + CI95 over 3 repeats)
    # -------------------------
    log_txt(logs_path, "==================== SUMMARY over repeats (mean + 95% CI) ====================")
    log_txt(logs_path, "Metrics computed on SMS TEST only. CI95 uses Student t (n=3 => t=4.303).")
    log_txt(logs_path, "")

    for it in range(1, iters + 1):
        mean_f1, lo_f1, hi_f1 = mean_ci95_t(all_f1_spam[it])
        mean_acc, lo_acc, hi_acc = mean_ci95_t(all_acc[it])
        mean_prec, lo_prec, hi_prec = mean_ci95_t(all_prec[it])
        mean_rec, lo_rec, hi_rec = mean_ci95_t(all_rec[it])

        log_txt(logs_path, f"iter {it}/{iters} | SMS_frac={it/iters:.1f}")
        log_txt(
            logs_path,
            f"  ACC mean={mean_acc:.4f}  CI95=[{lo_acc:.4f}, {hi_acc:.4f}]   runs={['%.4f' % v for v in all_acc[it]]}"
        )
        log_txt(
            logs_path,
            f"  PREC(spam) mean={mean_prec:.4f} CI95=[{lo_prec:.4f}, {hi_prec:.4f}] runs={['%.4f' % v for v in all_prec[it]]}"
        )
        log_txt(
            logs_path,
            f"  REC(spam) mean={mean_rec:.4f}  CI95=[{lo_rec:.4f}, {hi_rec:.4f}]   runs={['%.4f' % v for v in all_rec[it]]}"
        )
        log_txt(
            logs_path,
            f"  F1(spam) mean={mean_f1:.4f}   CI95=[{lo_f1:.4f}, {hi_f1:.4f}]   runs={['%.4f' % v for v in all_f1_spam[it]]}"
        )
        log_txt(logs_path, "")

    log_txt(logs_path, f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nDone. Logs saved to:", logs_path)


if __name__ == "__main__":
    main()
