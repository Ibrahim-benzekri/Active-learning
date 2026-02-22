# Active Learning (Hybrid) | Sequential: KMeans (diversity) -> Entropy (uncertainty)
# Retrain from scratch each iteration | 5 iterations x 3 repeats | logs to TXT

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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.cluster import KMeans


# -----------------------------
# Repro / preprocessing
# -----------------------------
def set_seed(seed: int):
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
# Model (returns hidden layer)
# -----------------------------
class SpamNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x, return_hidden: bool = False):
        hidden = self.relu(self.fc1(x))
        out = self.fc2(hidden)
        if return_hidden:
            return out, hidden
        return out

def build_model(input_dim: int, device: str) -> nn.Module:
    return SpamNN(input_dim).to(device)


# -----------------------------
# Training helpers
# -----------------------------
def make_balanced_loader(X_cpu: torch.Tensor, y_cpu: torch.Tensor, batch_size=256):
    y_np = y_cpu.numpy()
    class_counts = np.bincount(y_np, minlength=2)
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
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    return acc, prec, rec, f1b, report


# -----------------------------
# Hidden + Entropy
# -----------------------------
@torch.no_grad()
def hidden_embeddings(model, X_cpu, device, batch_size=2048) -> np.ndarray:
    model.eval()
    n = X_cpu.shape[0]
    out = np.empty((n, 128), dtype=np.float32)
    pos = 0
    for start in range(0, n, batch_size):
        xb = X_cpu[start:start + batch_size].to(device)
        _, h = model(xb, return_hidden=True)
        h = h.detach().cpu().numpy()
        out[pos:pos + h.shape[0]] = h
        pos += h.shape[0]
    return out

@torch.no_grad()
def entropy_scores(model, X_cpu, device, batch_size=2048) -> np.ndarray:
    model.eval()
    n = X_cpu.shape[0]
    out = np.empty(n, dtype=np.float32)
    eps = 1e-12
    pos = 0
    for start in range(0, n, batch_size):
        xb = X_cpu[start:start + batch_size].to(device)
        probs = torch.softmax(model(xb), dim=1)
        ent = -(probs * torch.log(probs + eps)).sum(dim=1)
        ent = ent.detach().cpu().numpy()
        out[pos:pos + ent.shape[0]] = ent
        pos += ent.shape[0]
    return out


def kmeans_then_entropy_select(model, X_pool_cpu, pool_indices, k, device, rep_seed, n_clusters=50):
    """
    diversity-first:
      1) KMeans on hidden embeddings of whole pool (n_clusters fixed)
      2) inside each cluster, pick highest entropy
      3) repeat round-robin until k samples selected
    """
    if k <= 0 or X_pool_cpu.shape[0] == 0:
        return np.array([], dtype=int)

    N = X_pool_cpu.shape[0]
    if N <= k:
        return np.array(pool_indices, dtype=int)

    # 1) embeddings + KMeans
    emb = hidden_embeddings(model, X_pool_cpu, device=device)
    K = int(min(n_clusters, N))
    km = KMeans(n_clusters=K, random_state=rep_seed, n_init=10)
    labels = km.fit_predict(emb)

    # 2) entropy on whole pool
    ent = entropy_scores(model, X_pool_cpu, device=device)
    rng = np.random.default_rng(rep_seed)
    ent = ent + rng.normal(0, 1e-12, size=ent.shape)

    # group indices by cluster
    clusters = []
    for c in range(K):
        idxs = np.where(labels == c)[0]
        if idxs.size == 0:
            continue
        # sort cluster members by entropy desc
        idxs_sorted = idxs[np.argsort(-ent[idxs])]
        clusters.append(list(idxs_sorted))

    # 3) round-robin pick top entropy from each cluster
    selected_local = []
    while len(selected_local) < k and len(clusters) > 0:
        progressed = False
        for cl in clusters:
            if len(selected_local) >= k:
                break
            if cl:
                selected_local.append(cl.pop(0))
                progressed = True
        # remove empty clusters
        clusters = [cl for cl in clusters if len(cl) > 0]
        if not progressed:
            break

    selected_local = np.array(selected_local[:k], dtype=int)
    return np.array(pool_indices, dtype=int)[selected_local]


# -----------------------------
# Main experiment
# -----------------------------
def main():
    repeats = 3
    iters = 5
    sms_test_size = 0.2

    epochs = 8
    lr = 1e-3
    batch_size = 256

    # diversity-first parameter
    N_CLUSTERS = 50

    base_seed = 42
    set_seed(base_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    os.makedirs("artifacts", exist_ok=True)
    logs_path = os.path.join("artifacts", "al_kmeans_entropy_sequential_retrain_logs.txt")
    with open(logs_path, "w", encoding="utf-8") as f:
        f.write("")

    log_txt(logs_path, "===== Active Learning (Hybrid Sequential) | KMeans -> Entropy =====")
    log_txt(logs_path, f"device={device} | repeats={repeats} | iters={iters} | n_clusters={N_CLUSTERS}")
    log_txt(logs_path, "")

    # Load EMAIL
    email_dataset = load_dataset("iamthearafatkhan/email-spam-dataset")["train"]
    df_email = pd.DataFrame(email_dataset)
    df_email["clean_text"] = df_email["text"].apply(clean_text)
    X_email = df_email["clean_text"]
    y_email = df_email["label"]

    X_email_train, X_email_tmp, y_email_train, y_email_tmp = train_test_split(
        X_email, y_email, test_size=0.4, stratify=y_email, random_state=base_seed
    )
    _X_email_val, _X_email_test, _y_email_val, _y_email_test = train_test_split(
        X_email_tmp, y_email_tmp, test_size=0.5, stratify=y_email_tmp, random_state=base_seed
    )

    # Load SMS
    sms_dataset = load_dataset("ucirvine/sms_spam")["train"]
    df_sms = pd.DataFrame(sms_dataset)
    text_col = "sms" if "sms" in df_sms.columns else ("text" if "text" in df_sms.columns else "message")
    df_sms["clean_text"] = df_sms[text_col].apply(clean_text)

    X_sms = df_sms["clean_text"]
    y_sms = df_sms["label"]

    X_sms_train, X_sms_test, y_sms_train, y_sms_test = train_test_split(
        X_sms, y_sms, test_size=sms_test_size, stratify=y_sms, random_state=base_seed
    )

    # TF-IDF fit on EMAIL train only
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_email_train_tfidf = vectorizer.fit_transform(X_email_train)

    X_email_train_tensor = torch.tensor(X_email_train_tfidf.toarray(), dtype=torch.float32)
    y_email_train_tensor = torch.tensor(y_email_train.values, dtype=torch.long)
    input_dim = X_email_train_tensor.shape[1]

    # SMS train pool tensors
    X_sms_train_arr = X_sms_train.reset_index(drop=True)
    y_sms_train_arr = y_sms_train.reset_index(drop=True)

    X_sms_train_tfidf = vectorizer.transform(X_sms_train_arr)
    X_sms_train_tensor = torch.tensor(X_sms_train_tfidf.toarray(), dtype=torch.float32)
    y_sms_train_tensor = torch.tensor(y_sms_train_arr.values, dtype=torch.long)
    sms_indices = np.arange(len(y_sms_train_tensor))

    # SMS test tensors on device
    X_sms_test_tfidf = vectorizer.transform(X_sms_test)
    X_sms_test_tensor = torch.tensor(X_sms_test_tfidf.toarray(), dtype=torch.float32).to(device)
    y_sms_test_tensor = torch.tensor(y_sms_test.values, dtype=torch.long).to(device)

    # Targets
    N_sms_train = len(y_sms_train_tensor)
    targets = [int(round(N_sms_train * (i / iters))) for i in range(1, iters + 1)]
    targets[-1] = N_sms_train

    all_acc = {it: [] for it in range(1, iters + 1)}
    all_prec = {it: [] for it in range(1, iters + 1)}
    all_rec = {it: [] for it in range(1, iters + 1)}
    all_f1 = {it: [] for it in range(1, iters + 1)}

    for rep in range(1, repeats + 1):
        rep_seed = base_seed + rep
        set_seed(rep_seed)
        log_txt(logs_path, f"==================== REPEAT {rep}/{repeats} | seed={rep_seed} ====================")

        labeled_set = set()
        unlabeled_set = set(sms_indices.tolist())

        for it in range(1, iters + 1):
            target = targets[it - 1]
            need = max(0, target - len(labeled_set))

            # scoring model training
            if len(labeled_set) == 0:
                X_train_comb = X_email_train_tensor
                y_train_comb = y_email_train_tensor
            else:
                idx = list(labeled_set)
                X_train_comb = torch.cat([X_email_train_tensor, X_sms_train_tensor[idx]], dim=0)
                y_train_comb = torch.cat([y_email_train_tensor, y_sms_train_tensor[idx]], dim=0)

            loader, _ = make_balanced_loader(X_train_comb, y_train_comb, batch_size=batch_size)

            scoring_model = build_model(input_dim, device)
            opt = optim.Adam(scoring_model.parameters(), lr=lr)
            crit = nn.CrossEntropyLoss()
            _ = train_epochs(scoring_model, opt, crit, loader, device, epochs=epochs)

            # selection: KMeans -> entropy
            if need > 0 and len(unlabeled_set) > 0:
                pool_list = np.array(sorted(unlabeled_set), dtype=int)
                X_pool_cpu = X_sms_train_tensor[pool_list]

                selected = kmeans_then_entropy_select(
                    scoring_model, X_pool_cpu, pool_list,
                    k=min(need, len(pool_list)),
                    device=device, rep_seed=rep_seed + it,
                    n_clusters=N_CLUSTERS
                )
                for s in selected.tolist():
                    labeled_set.add(int(s))
                    unlabeled_set.discard(int(s))

            # final retrain from scratch
            if len(labeled_set) == 0:
                X_train_final = X_email_train_tensor
                y_train_final = y_email_train_tensor
            else:
                idx2 = list(labeled_set)
                X_train_final = torch.cat([X_email_train_tensor, X_sms_train_tensor[idx2]], dim=0)
                y_train_final = torch.cat([y_email_train_tensor, y_sms_train_tensor[idx2]], dim=0)

            final_loader, _ = make_balanced_loader(X_train_final, y_train_final, batch_size=batch_size)

            model = build_model(input_dim, device)
            opt2 = optim.Adam(model.parameters(), lr=lr)
            crit2 = nn.CrossEntropyLoss()
            _ = train_epochs(model, opt2, crit2, final_loader, device, epochs=epochs)

            acc, prec, rec, f1b, rep_txt = eval_on_sms_test(model, X_sms_test_tensor, y_sms_test_tensor)

            all_acc[it].append(acc)
            all_prec[it].append(prec)
            all_rec[it].append(rec)
            all_f1[it].append(f1b)

            log_txt(logs_path, f"iter {it}/5 | target={target}/{N_sms_train}")
            log_txt(logs_path, f"  SMS_TEST acc={acc:.4f} prec_spam={prec:.4f} rec_spam={rec:.4f} f1_spam={f1b:.4f}")
            log_txt(logs_path, "")

    log_txt(logs_path, "==================== SUMMARY (mean + 95% CI) ====================")
    for it in range(1, iters + 1):
        m_acc, lo_acc, hi_acc = mean_ci95_t(all_acc[it])
        m_f1, lo_f1, hi_f1 = mean_ci95_t(all_f1[it])
        log_txt(logs_path, f"iter {it}/5")
        log_txt(logs_path, f"  ACC mean={m_acc:.4f} CI95=[{lo_acc:.4f},{hi_acc:.4f}] runs={['%.4f'%v for v in all_acc[it]]}")
        log_txt(logs_path, f"  F1(spam) mean={m_f1:.4f} CI95=[{lo_f1:.4f},{hi_f1:.4f}] runs={['%.4f'%v for v in all_f1[it]]}")
        log_txt(logs_path, "")

    print("Done. Logs saved to:", logs_path)


if __name__ == "__main__":
    main()
