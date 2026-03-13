#!/usr/bin/env python3
"""
HW08-09 execution script - runs all experiments and saves artifacts.
"""
import os
import json
import random
import csv
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms

# Paths
ARTIFACTS = os.path.join(os.path.dirname(__file__), 'artifacts')
FIGURES = os.path.join(ARTIFACTS, 'figures')
os.makedirs(FIGURES, exist_ok=True)

SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 15
HIDDEN_DIMS = (256, 128)
NUM_CLASSES = 10
INPUT_DIM = 28 * 28

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def accuracy_from_logits(logits, y_true):
    preds = torch.argmax(logits, dim=1)
    return (preds == y_true).float().mean().item()

class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=(256, 128), num_classes=10,
                 dropout_p=0.0, use_batchnorm=False):
        super().__init__()
        layers = [nn.Flatten()]
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.best_state = None
        self.counter = 0

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, y)
        n += 1
    return total_loss / n, total_acc / n

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, y)
        n += 1
    return total_loss / n, total_acc / n

def fit(model, train_loader, val_loader, optimizer, criterion, device, epochs,
        early_stopping=None, verbose=True):
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        if verbose:
            print(f"epoch {epoch:02d}/{epochs} | train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f}")
        if early_stopping is not None:
            if early_stopping.step(va_acc, model):
                early_stopping.restore_best(model)
                break
    return history

def plot_history(history, title="", savepath=None):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="train_loss")
    axes[0].plot(epochs, history["val_loss"], label="val_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"{title} - loss")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(epochs, history["train_acc"], label="train_acc")
    axes[1].plot(epochs, history["val_acc"], label="val_acc")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title(f"{title} - accuracy")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # KMNIST: сервер codh.rois.ac.jp часто недоступен (таймауты), используем FashionMNIST как fallback
    use_fashion = os.environ.get("USE_FASHION_MNIST", "").lower() in ("1", "true", "yes")
    if use_fashion:
        train_full = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
        dataset_name = "FashionMNIST"
        print("USE_FASHION_MNIST=1 — используем FashionMNIST")
    else:
        try:
            train_full = torchvision.datasets.KMNIST(root="./data", train=True, download=True, transform=transform)
            test_ds = torchvision.datasets.KMNIST(root="./data", train=False, download=True, transform=transform)
            dataset_name = "KMNIST"
        except (RuntimeError, OSError):
            train_full = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
            test_ds = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
            dataset_name = "FashionMNIST"
            print("KMNIST недоступен (таймаут/ошибка), используем FashionMNIST")

    val_ratio = 0.2
    val_size = int(len(train_full) * val_ratio)
    train_size = len(train_full) - val_size
    gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(train_full, [train_size, val_size], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    runs = []

    # E1: base
    print("\n=== E1: base ===")
    set_seed(SEED)
    model = MLP(INPUT_DIM, HIDDEN_DIMS, NUM_CLASSES, dropout_p=0.0, use_batchnorm=False).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    h1 = fit(model, train_loader, val_loader, opt, criterion, device, MAX_EPOCHS, None)
    best_idx = np.argmax(h1["val_acc"])
    runs.append({
        "experiment_id": "E1", "dataset": dataset_name, "seed": SEED,
        "model_summary": "2 hidden (256,128), ReLU, no dropout, no BN",
        "optimizer": "Adam", "lr": 1e-3, "momentum": 0, "weight_decay": 0,
        "epochs_trained": len(h1["train_loss"]), "best_val_accuracy": h1["val_acc"][best_idx],
        "best_val_loss": h1["val_loss"][best_idx]
    })

    # E2: Dropout
    print("\n=== E2: Dropout ===")
    set_seed(SEED)
    model = MLP(INPUT_DIM, HIDDEN_DIMS, NUM_CLASSES, dropout_p=0.3, use_batchnorm=False).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    h2 = fit(model, train_loader, val_loader, opt, criterion, device, MAX_EPOCHS, None)
    best_idx = np.argmax(h2["val_acc"])
    runs.append({
        "experiment_id": "E2", "dataset": dataset_name, "seed": SEED,
        "model_summary": "2 hidden (256,128), ReLU, Dropout(0.3), no BN",
        "optimizer": "Adam", "lr": 1e-3, "momentum": 0, "weight_decay": 0,
        "epochs_trained": len(h2["train_loss"]), "best_val_accuracy": h2["val_acc"][best_idx],
        "best_val_loss": h2["val_loss"][best_idx]
    })

    # E3: BatchNorm
    print("\n=== E3: BatchNorm ===")
    set_seed(SEED)
    model = MLP(INPUT_DIM, HIDDEN_DIMS, NUM_CLASSES, dropout_p=0.0, use_batchnorm=True).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    h3 = fit(model, train_loader, val_loader, opt, criterion, device, MAX_EPOCHS, None)
    best_idx = np.argmax(h3["val_acc"])
    runs.append({
        "experiment_id": "E3", "dataset": dataset_name, "seed": SEED,
        "model_summary": "2 hidden (256,128), ReLU, BatchNorm, no dropout",
        "optimizer": "Adam", "lr": 1e-3, "momentum": 0, "weight_decay": 0,
        "epochs_trained": len(h3["train_loss"]), "best_val_accuracy": h3["val_acc"][best_idx],
        "best_val_loss": h3["val_loss"][best_idx]
    })

    # Choose best from E2/E3 for E4
    val_e2, val_e3 = runs[1]["best_val_accuracy"], runs[2]["best_val_accuracy"]
    use_dropout = val_e2 >= val_e3

    # E4: EarlyStopping
    print("\n=== E4: EarlyStopping (best from E2/E3) ===")
    set_seed(SEED)
    model = MLP(INPUT_DIM, HIDDEN_DIMS, NUM_CLASSES, dropout_p=0.3 if use_dropout else 0.0,
                use_batchnorm=not use_dropout).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    es = EarlyStopping(patience=4)
    h4 = fit(model, train_loader, val_loader, opt, criterion, device, MAX_EPOCHS, early_stopping=es)
    best_idx = np.argmax(h4["val_acc"])
    runs.append({
        "experiment_id": "E4", "dataset": dataset_name, "seed": SEED,
        "model_summary": f"2 hidden, EarlyStopping, {'Dropout' if use_dropout else 'BatchNorm'}",
        "optimizer": "Adam", "lr": 1e-3, "momentum": 0, "weight_decay": 0,
        "epochs_trained": len(h4["train_loss"]), "best_val_accuracy": h4["val_acc"][best_idx],
        "best_val_loss": h4["val_loss"][best_idx]
    })

    torch.save(model.state_dict(), os.path.join(ARTIFACTS, 'best_model.pt'))
    plot_history(h4, "E4 best", os.path.join(FIGURES, 'curves_best.png'))

    best_config = {
        "dataset": dataset_name, "seed": SEED, "hidden_dims": list(HIDDEN_DIMS),
        "dropout": 0.3 if use_dropout else 0.0, "use_batchnorm": not use_dropout,
        "optimizer": "Adam", "lr": 1e-3, "best_val_accuracy": runs[-1]["best_val_accuracy"],
    }
    with open(os.path.join(ARTIFACTS, 'best_config.json'), 'w') as f:
        json.dump(best_config, f, indent=2)

    # O1: LR too large
    print("\n=== O1: LR too large ===")
    set_seed(SEED)
    model_o1 = MLP(INPUT_DIM, HIDDEN_DIMS, NUM_CLASSES, dropout_p=0.3 if use_dropout else 0.0,
                   use_batchnorm=not use_dropout).to(device)
    opt_o1 = optim.Adam(model_o1.parameters(), lr=1e-1)
    ho1 = fit(model_o1, train_loader, val_loader, opt_o1, criterion, device, 6, None)
    runs.append({
        "experiment_id": "O1", "dataset": dataset_name, "seed": SEED,
        "model_summary": "same as E4", "optimizer": "Adam", "lr": 1e-1, "momentum": 0, "weight_decay": 0,
        "epochs_trained": 6, "best_val_accuracy": max(ho1["val_acc"]), "best_val_loss": min(ho1["val_loss"])
    })

    # O2: LR too small
    print("\n=== O2: LR too small ===")
    set_seed(SEED)
    model_o2 = MLP(INPUT_DIM, HIDDEN_DIMS, NUM_CLASSES, dropout_p=0.3 if use_dropout else 0.0,
                   use_batchnorm=not use_dropout).to(device)
    opt_o2 = optim.Adam(model_o2.parameters(), lr=1e-5)
    ho2 = fit(model_o2, train_loader, val_loader, opt_o2, criterion, device, 6, None)
    runs.append({
        "experiment_id": "O2", "dataset": dataset_name, "seed": SEED,
        "model_summary": "same as E4", "optimizer": "Adam", "lr": 1e-5, "momentum": 0, "weight_decay": 0,
        "epochs_trained": 6, "best_val_accuracy": max(ho2["val_acc"]), "best_val_loss": min(ho2["val_loss"])
    })

    # O3: SGD + momentum + weight decay
    print("\n=== O3: SGD + momentum + weight decay ===")
    set_seed(SEED)
    model_o3 = MLP(INPUT_DIM, HIDDEN_DIMS, NUM_CLASSES, dropout_p=0.3 if use_dropout else 0.0,
                   use_batchnorm=not use_dropout).to(device)
    opt_o3 = optim.SGD(model_o3.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    ho3 = fit(model_o3, train_loader, val_loader, opt_o3, criterion, device, 12, None)
    runs.append({
        "experiment_id": "O3", "dataset": dataset_name, "seed": SEED,
        "model_summary": "same as E4", "optimizer": "SGD", "lr": 1e-2, "momentum": 0.9, "weight_decay": 1e-4,
        "epochs_trained": 12, "best_val_accuracy": max(ho3["val_acc"]), "best_val_loss": min(ho3["val_loss"])
    })

    # Save runs.csv
    with open(os.path.join(ARTIFACTS, 'runs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=runs[0].keys())
        w.writeheader()
        w.writerows(runs)

    # Plot LR extremes
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs_o1 = np.arange(1, len(ho1["val_loss"]) + 1)
    epochs_o2 = np.arange(1, len(ho2["val_loss"]) + 1)
    axes[0].plot(epochs_o1, ho1["val_loss"], 'r-', label="O1 val loss (lr=1e-1)")
    axes[0].plot(epochs_o2, ho2["val_loss"], 'b-', label="O2 val loss (lr=1e-5)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("val_loss")
    axes[0].set_title("LR extremes: val_loss")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(epochs_o1, ho1["val_acc"], 'r-', label="O1 val acc (lr=1e-1)")
    axes[1].plot(epochs_o2, ho2["val_acc"], 'b-', label="O2 val acc (lr=1e-5)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val_acc")
    axes[1].set_title("LR extremes: val_acc")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, 'curves_lr_extremes.png'), dpi=100, bbox_inches='tight')
    plt.close()

    # Final test eval for E4
    model.load_state_dict(torch.load(os.path.join(ARTIFACTS, 'best_model.pt'), map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal test accuracy (E4): {test_acc:.4f}")
    best_config["test_accuracy"] = test_acc
    with open(os.path.join(ARTIFACTS, 'best_config.json'), 'w') as f:
        json.dump(best_config, f, indent=2)

    print("\nDone. Artifacts saved to", ARTIFACTS)

if __name__ == "__main__":
    main()
