# Active Learning (Hybrid) | Sequential: Entropy -> KMeans (diversity)
# Retrain from scratch each iteration | 5 iterations x 3 repeats | logs to TXT
# Do NOT name this file random.py

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
# (0) Optional: avoid KMeans MKL warning on Windows
# Put this BEFORE importing sklearn in your real script.
# If you still see warning, set OMP_NUM_THREADS=1 in your OS env.
# -----------------------------
# import os
# os.environ["OMP_NUM_THREADS"] = "1"


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
    """
    95% CI with Student t (small n). For n=3 => df=2 => tcrit ~ 4.303.
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
    """
    WeightedRandomSampler to mitigate class imbalance IN TRAINING.
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
# (1) Uncertainty: Entropy scores
# -----------------------------
@torch.no_grad()
def entropy_scores(model, X_pool_cpu, device, batch_size=2048):
    """
    Returns entropy per sample in pool: higher => more uncertain.
    """
    model.eval()
    n = X_pool_cpu.shape[0]
    out = np.empty(n, dtype=np.float32)
    eps = 1e-12

    pos = 0
    for start in range(0, n, batch_size):
        xb = X_pool_cpu[start:start + batch_size].to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        ent = -(probs * torch.log(probs + eps)).sum(dim=1)  # [B]
        ent = ent.detach().cpu().numpy()
        out[pos:pos + ent.shape[0]] = ent
        pos += ent.shape[0]

    return out


# -----------------------------
# (2) Diversity inside top-M: KMeans on hidden embeddings
# -----------------------------
@torch.no_grad()
def hidden_embeddings(model, X_cpu, device, batch_size=2048) -> np.ndarray:
    """
    Compute hidden embeddings for X (CPU tensor).
    Returns numpy array [N, 128].
    """
    model.eval()
    n = X_cpu.shape[0]
    out = np.empty((n, 128), dtype=np.float32)

    pos = 0
    for start in range(0, n, batch_size):
        xb = X_cpu[start:start + batch_size].to(device)
        _, h = model(xb, return_hidden=True)  # [B,128]
        h = h.detach().cpu().numpy()
        out[pos:pos + h.shape[0]] = h
        pos += h.shape[0]

    return out

def pick_nearest_to_centroids(emb: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    For each cluster, pick the point closest to its centroid.
    Returns local indices selected (size <= n_clusters).
    """
    selected_local = []
    for c in range(centers.shape[0]):
        idxs = np.where(labels == c)[0]
        if idxs.size == 0:
            continue
        cluster_emb = emb[idxs]
        center = centers[c]
        d2 = ((cluster_emb - center) ** 2).sum(axis=1)
        best = idxs[int(np.argmin(d2))]
        selected_local.append(best)
    return np.array(selected_local, dtype=int)

def entropy_then_kmeans_select(
    model,
    X_pool_cpu,
    pool_indices,
    k,
    device,
    rep_seed,
    topM_factor=5
):
    """
    Sequential hybrid:
      1) compute entropy on full pool
      2) keep top-M most uncertain (M = min(N_pool, topM_factor * k))
      3) run KMeans with n_clusters = k on embeddings of top-M
      4) select nearest-to-centroid points => k diverse + uncertain samples
    """
    if k <= 0 or X_pool_cpu.shape[0] == 0:
        return np.array([], dtype=int)

    N = X_pool_cpu.shape[0]
    M = int(min(N, max(k, topM_factor * k)))  # ensure M>=k

    # ---- (1) Entropy over full pool
    ent = entropy_scores(model, X_pool_cpu, device=device)

    # ---- (2) Keep top-M uncertain
    rng = np.random.default_rng(rep_seed)
    ent = ent + rng.normal(0, 1e-12, size=ent.shape)  # tie-break
    topM_local = np.argsort(-ent)[:M]  # descending entropy

    X_topM_cpu = X_pool_cpu[topM_local]
    pool_topM_indices = np.array(pool_indices, dtype=int)[topM_local]

    # ---- If M == k, just return these top-k uncertain
    if M == k:
        return pool_topM_indices

    # ---- (3) KMeans on hidden embeddings of top-M
    emb = hidden_embeddings(model, X_topM_cpu, device=device)  # [M,128]

    # If too few points or k too large
    if emb.shape[0] <= k:
        return pool_topM_indices

    km = KMeans(n_clusters=k, random_state=rep_seed, n_init=10)
    labels = km.fit_predict(emb)
    centers = km.cluster_centers_

    # ---- (4) Pick closest points to centroids
    selected_local = pick_nearest_to_centroids(emb, labels, centers)
    selected = pool_topM_indices[selected_local]

    # Fill if <k (rare)
    if selected.shape[0] < k:
        remaining = np.setdiff1d(pool_topM_indices, selected, assume_unique=False)
        need = k - selected.shape[0]
        if remaining.size > 0:
            extra = rng.choice(remaining, size=min(need, remaining.size), replace=False)
            selected = np.concatenate([selected, extra])

    return selected[:k]


# -----------------------------
# Main experiment
# -----------------------------
def main():
    # ---- Config ----
    repeats = 3
    iters = 5
    sms_test_size = 0.2

    epochs = 8
    lr = 1e-3
    batch_size = 256

    topM_factor = 5  # hybrid: keep top (5*k) most uncertain before KMeans

    base_seed = 42
    set_seed(base_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    os.makedirs("artifacts", exist_ok=True)
    logs_path = os.path.join("artifacts", "al_entropy_kmeans_sequential_retrain_logs.txt")

    # reset logs
    with open(logs_path, "w", encoding="utf-8") as f:
        f.write("")

    log_txt(logs_path, "===== Active Learning (Hybrid Sequential) | Entropy -> KMeans (hidden) =====")
    log_txt(logs_path, f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_txt(logs_path, f"device={device} | repeats={repeats} | iters={iters}")
    log_txt(logs_path, f"selection: topM_factor={topM_factor} (keep top M=topM_factor*k uncertain, then KMeans to pick k diverse)")
    log_txt(logs_path, f"training: epochs={epochs} lr={lr} batch_size={batch_size} sampler=WeightedRandomSampler")
    log_txt(logs_path, f"evaluation: SMS test only (fixed split={sms_test_size})")
    log_txt(logs_path, "")

    # ---- Load EMAIL ----
    print("Loading EMAIL dataset...")
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

    # ---- Load SMS ----
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

    X_sms_train, X_sms_test, y_sms_train, y_sms_test = train_test_split(
        X_sms, y_sms, test_size=sms_test_size, stratify=y_sms, random_state=base_seed
    )

    # ---- TF-IDF on EMAIL train only ----
    print("Fitting TF-IDF on EMAIL train only...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_email_train_tfidf = vectorizer.fit_transform(X_email_train)

    X_email_train_tensor = torch.tensor(X_email_train_tfidf.toarray(), dtype=torch.float32)  # CPU
    y_email_train_tensor = torch.tensor(y_email_train.values, dtype=torch.long)              # CPU
    input_dim = X_email_train_tensor.shape[1]

    # ---- SMS train pool tensors (CPU) ----
    X_sms_train_arr = X_sms_train.reset_index(drop=True)
    y_sms_train_arr = y_sms_train.reset_index(drop=True)

    X_sms_train_tfidf = vectorizer.transform(X_sms_train_arr)
    X_sms_train_tensor = torch.tensor(X_sms_train_tfidf.toarray(), dtype=torch.float32)  # CPU
    y_sms_train_tensor = torch.tensor(y_sms_train_arr.values, dtype=torch.long)         # CPU
    sms_indices = np.arange(len(y_sms_train_tensor))

    # ---- SMS test tensors (device) ----
    X_sms_test_tfidf = vectorizer.transform(X_sms_test)
    X_sms_test_tensor = torch.tensor(X_sms_test_tfidf.toarray(), dtype=torch.float32).to(device)
    y_sms_test_tensor = torch.tensor(y_sms_test.values, dtype=torch.long).to(device)

    # ---- Iteration targets (20%,40%,...,100%) ----
    N_sms_train = len(y_sms_train_tensor)
    targets = [int(round(N_sms_train * (i / iters))) for i in range(1, iters + 1)]
    targets[-1] = N_sms_train

    # ---- Summary storage over repeats ----
    all_f1_spam = {it: [] for it in range(1, iters + 1)}
    all_acc = {it: [] for it in range(1, iters + 1)}
    all_prec = {it: [] for it in range(1, iters + 1)}
    all_rec = {it: [] for it in range(1, iters + 1)}

    # =========================
    # REPEATS
    # =========================
    for rep in range(1, repeats + 1):
        rep_seed = base_seed + rep
        set_seed(rep_seed)
        log_txt(logs_path, f"==================== REPEAT {rep}/{repeats} | seed={rep_seed} ====================")

        labeled_set = set()
        unlabeled_set = set(sms_indices.tolist())

        for it in range(1, iters + 1):
            target = targets[it - 1]
            need = max(0, target - len(labeled_set))

            # 1) Scoring model train from scratch on EMAIL + current labeled SMS
            if len(labeled_set) == 0:
                X_train_comb = X_email_train_tensor
                y_train_comb = y_email_train_tensor
            else:
                idx = list(labeled_set)
                X_train_comb = torch.cat([X_email_train_tensor, X_sms_train_tensor[idx]], dim=0)
                y_train_comb = torch.cat([y_email_train_tensor, y_sms_train_tensor[idx]], dim=0)

            loader, counts = make_balanced_loader(X_train_comb, y_train_comb, batch_size=batch_size)

            set_seed(rep_seed)
            scoring_model = build_model(input_dim=input_dim, device=device)
            optimizer = optim.Adam(scoring_model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            head = (
                f"[rep {rep}/{repeats} | iter {it}/{iters} | target_SMS={target}/{N_sms_train}] "
                f"SCORING TRAIN: EMAIL={len(y_email_train_tensor)} + labeled_SMS={len(labeled_set)} "
                f"=> total={len(y_train_comb)} | train_counts ham/spam={counts.tolist()}"
            )
            print(head)
            log_txt(logs_path, head)

            t_scoring = train_epochs(scoring_model, optimizer, criterion, loader, device, epochs=epochs)

            # 2) Sequential selection: Entropy -> KMeans on top-M
            if need > 0 and len(unlabeled_set) > 0:
                pool_list = np.array(sorted(unlabeled_set), dtype=int)
                X_pool_cpu = X_sms_train_tensor[pool_list]

                selected = entropy_then_kmeans_select(
                    model=scoring_model,
                    X_pool_cpu=X_pool_cpu,
                    pool_indices=pool_list,
                    k=min(need, len(pool_list)),
                    device=device,
                    rep_seed=rep_seed + it,
                    topM_factor=topM_factor
                )

                for s in selected.tolist():
                    labeled_set.add(int(s))
                    unlabeled_set.discard(int(s))

                log_txt(
                    logs_path,
                    f"  selection(hybrid): need={need}, selected={len(selected)}, labeled_now={len(labeled_set)} | "
                    f"topM_factor={topM_factor}"
                )
            else:
                log_txt(logs_path, f"  selection(hybrid): need={need}, selected=0, labeled_now={len(labeled_set)}")

            # 3) Final retrain from scratch on EMAIL + labeled after selection
            if len(labeled_set) == 0:
                X_train_final = X_email_train_tensor
                y_train_final = y_email_train_tensor
            else:
                idx2 = list(labeled_set)
                X_train_final = torch.cat([X_email_train_tensor, X_sms_train_tensor[idx2]], dim=0)
                y_train_final = torch.cat([y_email_train_tensor, y_sms_train_tensor[idx2]], dim=0)

            final_loader, _ = make_balanced_loader(X_train_final, y_train_final, batch_size=batch_size)

            set_seed(rep_seed)
            model = build_model(input_dim=input_dim, device=device)
            optimizer2 = optim.Adam(model.parameters(), lr=lr)
            criterion2 = nn.CrossEntropyLoss()

            t_final = train_epochs(model, optimizer2, criterion2, final_loader, device, epochs=epochs)

            # 4) Evaluate on SMS test
            metrics = eval_on_sms_test(model, X_sms_test_tensor, y_sms_test_tensor)

            all_acc[it].append(metrics["sms_test_accuracy"])
            all_prec[it].append(metrics["sms_test_precision_spam"])
            all_rec[it].append(metrics["sms_test_recall_spam"])
            all_f1_spam[it].append(metrics["sms_test_f1_spam"])

            log_txt(logs_path, f"  scoring_train_time_s={t_scoring:.2f} | final_train_time_s={t_final:.2f}")
            log_txt(
                logs_path,
                f"  SMS_TEST acc={metrics['sms_test_accuracy']:.4f} "
                f"prec_spam={metrics['sms_test_precision_spam']:.4f} "
                f"recall_spam={metrics['sms_test_recall_spam']:.4f} "
                f"f1_spam={metrics['sms_test_f1_spam']:.4f} "
                f"f1_macro={metrics['sms_test_f1_macro']:.4f} "
                f"f1_weighted={metrics['sms_test_f1_weighted']:.4f}"
            )
            log_txt(logs_path, "  Classification report:")
            for line in metrics["report"].splitlines():
                log_txt(logs_path, "    " + line)
            log_txt(logs_path, "")

    # =========================
    # SUMMARY
    # =========================
    log_txt(logs_path, "==================== SUMMARY over repeats (mean + 95% CI) ====================")
    log_txt(logs_path, "Metrics computed on SMS TEST only. CI95 uses Student t (n=3 => t=4.303).")
    log_txt(logs_path, "")

    for it in range(1, iters + 1):
        mean_f1, lo_f1, hi_f1 = mean_ci95_t(all_f1_spam[it])
        mean_acc, lo_acc, hi_acc = mean_ci95_t(all_acc[it])
        mean_prec, lo_prec, hi_prec = mean_ci95_t(all_prec[it])
        mean_rec, lo_rec, hi_rec = mean_ci95_t(all_rec[it])

        log_txt(logs_path, f"iter {it}/{iters} | labeled_SMS_target={targets[it-1]}/{N_sms_train} ({it/iters:.1f})")
        log_txt(logs_path, f"  ACC mean={mean_acc:.4f}  CI95=[{lo_acc:.4f}, {hi_acc:.4f}]  runs={['%.4f' % v for v in all_acc[it]]}")
        log_txt(logs_path, f"  PREC(spam) mean={mean_prec:.4f} CI95=[{lo_prec:.4f}, {hi_prec:.4f}] runs={['%.4f' % v for v in all_prec[it]]}")
        log_txt(logs_path, f"  REC(spam) mean={mean_rec:.4f}  CI95=[{lo_rec:.4f}, {hi_rec:.4f}]  runs={['%.4f' % v for v in all_rec[it]]}")
        log_txt(logs_path, f"  F1(spam) mean={mean_f1:.4f}   CI95=[{lo_f1:.4f}, {hi_f1:.4f}]  runs={['%.4f' % v for v in all_f1_spam[it]]}")
        log_txt(logs_path, "")

    log_txt(logs_path, f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nDone. Logs saved to:", logs_path)


if __name__ == "__main__":
    main()
