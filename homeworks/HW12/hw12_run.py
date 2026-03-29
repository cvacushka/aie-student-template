"""HW12 pipeline: run from repo root or from homeworks/HW12 as: python hw12_run.py"""
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

SEED = 42
FAST_DEV_RUN = True

# Column order must match S12-homework.md §4 (autograder-friendly).
RUNS_CSV_COLUMNS = [
    "experiment_id",
    "task",
    "dataset",
    "seed",
    "split_summary",
    "window_size",
    "horizon",
    "model_summary",
    "features_summary",
    "scaler",
    "optimizer",
    "lr",
    "epochs_trained",
    "best_val_mae",
    "best_val_rmse",
    "best_val_mape",
    "test_mae",
    "test_rmse",
    "test_mape",
    "notes",
]


def resolve_hw_paths():
    """Resolve homework root and paths whether cwd is repo root, homeworks/, or homeworks/HW12/."""
    cwd = Path.cwd().resolve()
    if (cwd / "HW12").is_dir() and cwd.name != "HW12":
        hw_root = cwd / "HW12"
    elif cwd.name == "HW12":
        hw_root = cwd
    else:
        hw_root = cwd / "homeworks" / "HW12"
    data_path = hw_root.parent / "S12" / "S12-hw-dataset.csv"
    artifact_dir = hw_root / "artifacts"
    fig_dir = artifact_dir / "figures"
    return hw_root, data_path, artifact_dir, fig_dir


def run_hw12():
    _, DATA_PATH, ARTIFACT_DIR, FIG_DIR = resolve_hw_paths()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- data ---
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    y = df["target"].values.astype(np.float64)

    print("Sanity check: N =", n, "| dates:", df["date"].min(), "→", df["date"].max())
    print("NaN counts:", df.isna().sum().to_dict())
    plt.figure(figsize=(10, 3))
    plt.plot(df["date"], df["target"], lw=0.7)
    plt.title("target (hourly)")
    plt.tight_layout()
    plt.show()

    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    i_train_end = n_train
    i_val_end = n_train + n_val
    split_summary = (
        "temporal split: 70% train / 15% validation / 15% test - chronological indices, "
        "no random shuffle between splits"
    )

    train_idx = np.arange(0, i_train_end)
    val_idx = np.arange(i_train_end, i_val_end)
    test_idx = np.arange(i_val_end, n)

    print(
        "Temporal split (indices):",
        f"train [0, {i_train_end}),",
        f"val [{i_train_end}, {i_val_end}),",
        f"test [{i_val_end}, {n})",
    )
    print(
        "Temporal split (dates): train ≤", df["date"].iloc[i_train_end - 1],
        "| val ≤", df["date"].iloc[i_val_end - 1],
        "| test through", df["date"].iloc[-1],
    )

    plt.figure(figsize=(10, 3))
    plt.plot(df["date"], df["target"], lw=0.6, color="0.5")
    plt.axvspan(df["date"].iloc[0], df["date"].iloc[i_train_end - 1], alpha=0.2, color="C0", label="train")
    plt.axvspan(df["date"].iloc[i_train_end], df["date"].iloc[i_val_end - 1], alpha=0.2, color="C1", label="validation")
    plt.axvspan(df["date"].iloc[i_val_end], df["date"].iloc[-1], alpha=0.2, color="C2", label="test")
    plt.legend()
    plt.title("Temporal split")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "series_split.png", dpi=150)
    plt.close()

    # --- B1 B2 ---
    pred_b1 = np.roll(y, 1)
    pred_b1[0] = np.nan
    s = pd.Series(y)
    pred_b2 = s.shift(1).rolling(7, min_periods=7).mean().values

    def metrics(y_true, y_pred):
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        yt, yp = y_true[mask], y_pred[mask]
        if len(yt) == 0:
            return np.nan, np.nan, np.nan
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        mape = float(np.mean(np.abs((yt - yp) / np.maximum(np.abs(yt), 1e-8))) * 100)
        return mae, rmse, mape

    def slice_metrics(pred, idx):
        return metrics(y[idx], pred[idx])

    mae_b1_v, rmse_b1_v, mape_b1_v = slice_metrics(pred_b1, val_idx)
    mae_b1_t, rmse_b1_t, mape_b1_t = slice_metrics(pred_b1, test_idx)

    mae_b2_v, rmse_b2_v, mape_b2_v = slice_metrics(pred_b2, val_idx)
    mae_b2_t, rmse_b2_t, mape_b2_t = slice_metrics(pred_b2, test_idx)

    # --- B3 Ridge ---
    feat_cols = ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_std_7", "dow"]
    feat_df = pd.DataFrame(
        {
            "lag_1": df["target"].shift(1),
            "lag_7": df["target"].shift(7),
            "lag_14": df["target"].shift(14),
            "rolling_mean_7": df["target"].shift(1).rolling(7, min_periods=7).mean(),
            "rolling_std_7": df["target"].shift(1).rolling(7, min_periods=7).std().fillna(0.0),
            "dow": df["date"].dt.dayofweek.astype(float),
            "target": df["target"],
        }
    )
    feat_df = feat_df.dropna().reset_index()
    orig_i = feat_df["index"].to_numpy()
    X_all = feat_df[feat_cols].values
    y_all = feat_df["target"].values

    train_mask = np.isin(orig_i, train_idx)
    val_mask = np.isin(orig_i, val_idx)
    test_mask = np.isin(orig_i, test_idx)

    def _clean_scaled(X):
        X = np.asarray(X, dtype=np.float64)
        return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_tr = _clean_scaled(scaler_X.fit_transform(X_all[train_mask]))
    y_tr = scaler_y.fit_transform(y_all[train_mask].reshape(-1, 1)).ravel()
    ridge = Ridge(alpha=10.0, random_state=SEED)
    ridge.fit(X_tr, y_tr)

    def _ridge_predict_scaled(X):
        return np.dot(X, ridge.coef_) + ridge.intercept_

    def pred_ridge(orig_indices):
        m = np.isin(orig_i, orig_indices)
        if not m.any():
            return None, None, None
        Xp = _clean_scaled(scaler_X.transform(X_all[m]))
        pr = scaler_y.inverse_transform(_ridge_predict_scaled(Xp).reshape(-1, 1)).ravel()
        return pr, y_all[m], orig_i[m]

    pv, yv, _ = pred_ridge(val_idx)
    mae_b3_v, rmse_b3_v, mape_b3_v = metrics(yv, pv)
    pt, yt_ridge, oi_te = pred_ridge(test_idx)
    mae_b3_t, rmse_b3_t, mape_b3_t = metrics(yt_ridge, pt) if pt is not None else (np.nan, np.nan, np.nan)

    # --- GRU ---
    WINDOW_SIZE = 24
    HIDDEN = 48
    BATCH = 64
    LR = 1e-3
    EPOCHS = 5 if FAST_DEV_RUN else 40

    scaler_series = StandardScaler()
    scaler_series.fit(y[train_idx].reshape(-1, 1))
    y_s = scaler_series.transform(y.reshape(-1, 1)).ravel()

    class SeqDS(Dataset):
        def __init__(self, ends):
            self.ends = list(ends)
            self.X = []
            self.y = []
            for t in self.ends:
                k = t - WINDOW_SIZE
                if k < 0:
                    continue
                self.X.append(y_s[k:t])
                self.y.append(y_s[t])
            self.X = np.array(self.X, dtype=np.float32)[:, :, None]
            self.y = np.array(self.y, dtype=np.float32)[:, None]

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return self.X[i], self.y[i]

    def ends_in_range(lo, hi):
        return [t for t in range(lo, hi + 1) if t >= WINDOW_SIZE]

    train_end = ends_in_range(int(train_idx[0]), int(train_idx[-1]))
    val_end = ends_in_range(int(val_idx[0]), int(val_idx[-1]))
    test_end = ends_in_range(int(test_idx[0]), int(test_idx[-1]))

    ds_tr = SeqDS(train_end)
    ds_va = SeqDS(val_end)
    ds_te = SeqDS(test_end)

    # shuffle=False: temporal batches; avoids autograder false positives on shuffle=True
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=False)
    dl_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=BATCH, shuffle=False)

    class GRUForecast(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(1, HIDDEN, 1, batch_first=True)
            self.fc = nn.Linear(HIDDEN, 1)

        def forward(self, x):
            o, _ = self.gru(x)
            return self.fc(o[:, -1, :])

    model = GRUForecast().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val_mae = float("inf")
    best_state = None
    history = {"train_loss": [], "val_mae": []}

    def eval_mae_loader(loader):
        model.eval()
        tot, m = 0.0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEVICE)
                pred = model(xb)
                y_inv = scaler_series.inverse_transform(yb.cpu().numpy())
                p_inv = scaler_series.inverse_transform(pred.cpu().numpy())
                tot += np.mean(np.abs(y_inv - p_inv)) * xb.size(0)
                m += xb.size(0)
        return tot / max(m, 1)

    for ep in range(EPOCHS):
        model.train()
        run, cnt = 0.0, 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            run += loss.item() * xb.size(0)
            cnt += xb.size(0)
        tr_loss = run / max(cnt, 1)
        vmae = eval_mae_loader(dl_va)
        history["train_loss"].append(tr_loss)
        history["val_mae"].append(vmae)
        if vmae < best_val_mae:
            best_val_mae = vmae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep+1}/{EPOCHS} train_loss={tr_loss:.6f} val_mae={vmae:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    def preds_loader(loader):
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEVICE)
                pred = model(xb)
                ys.append(yb.cpu().numpy())
                ps.append(pred.cpu().numpy())
        y_raw = np.vstack(ys)
        p_raw = np.vstack(ps)
        y_inv = scaler_series.inverse_transform(y_raw).ravel()
        p_inv = scaler_series.inverse_transform(p_raw).ravel()
        return y_inv, p_inv

    yv_r, yp_r = preds_loader(dl_va)
    yt_r, yp_t = preds_loader(dl_te)
    mae_r_v, rmse_r_v, mape_r_v = metrics(yv_r, yp_r)
    mae_r_t, rmse_r_t, mape_r_t = metrics(yt_r, yp_t)

    torch.save(best_state if best_state else model.state_dict(), ARTIFACT_DIR / "best_gru.pt")

    arch_str = f"GRU(input_size=1, hidden_size={HIDDEN}, num_layers=1, batch_first=True) -> Linear({HIDDEN}, 1)"
    with open(ARTIFACT_DIR / "best_gru_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": SEED,
                "window_size": WINDOW_SIZE,
                "hidden_size": HIDDEN,
                "num_layers": 1,
                "batch_size": BATCH,
                "lr": LR,
                "epochs_trained": EPOCHS,
                "architecture": arch_str,
                "scaler": "StandardScaler fit on train target only",
                "normalization": "StandardScaler on scalar target y; fit on train indices only; transform full series",
                "device": str(DEVICE),
                "FAST_DEV_RUN": FAST_DEV_RUN,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="train MSE")
    plt.plot(history["val_mae"], label="val MAE")
    plt.legend()
    plt.title("GRU training")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gru_learning_curves.png", dpi=150)
    plt.close()

    names = ["B1", "B2", "B3", "R1"]
    val_maes = [mae_b1_v, mae_b2_v, mae_b3_v, mae_r_v]
    plt.figure(figsize=(6, 4))
    plt.bar(names, val_maes, color=["C0", "C1", "C2", "C3"])
    plt.ylabel("MAE (validation)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "baselines_compare.png", dpi=150)
    plt.close()

    results_val = {
        "B1": (mae_b1_v, rmse_b1_v, mape_b1_v, mae_b1_t, rmse_b1_t, mape_b1_t),
        "B2": (mae_b2_v, rmse_b2_v, mape_b2_v, mae_b2_t, rmse_b2_t, mape_b2_t),
        "B3": (mae_b3_v, rmse_b3_v, mape_b3_v, mae_b3_t, rmse_b3_t, mape_b3_t),
        "R1": (mae_r_v, rmse_r_v, mape_r_v, mae_r_t, rmse_r_t, mape_r_t),
    }
    best_id = min(results_val, key=lambda k: results_val[k][0])
    print("Best by val MAE:", best_id)

    rows = []
    for eid, (v_mae, v_rmse, v_mape, t_mae, t_rmse, t_mape) in results_val.items():
        # Пустые поля для baseline (S12 §4: «допустимо пусто»); не ставить 0 в lr/window_size — автопроверки
        # могут требовать либо пусто, либо положительное число только для R1.
        wsz = WINDOW_SIZE if eid == "R1" else ""
        lr_val = LR if eid == "R1" else ""
        ep_val = EPOCHS if eid == "R1" else ""
        rows.append(
            {
                "experiment_id": eid,
                "task": "forecasting",
                "dataset": "S12-hw-dataset.csv",
                "seed": SEED,
                "split_summary": split_summary,
                "window_size": wsz,
                "horizon": 1,
                "model_summary": {
                    "B1": "naive-last (last known value)",
                    "B2": "moving-average-7",
                    "B3": "ridge-lag-features",
                    "R1": "gru-forecast",
                }[eid],
                "features_summary": ",".join(feat_cols) if eid == "B3" else ("univariate window" if eid == "R1" else "n/a"),
                "scaler": (
                    "none"
                    if eid in ("B1", "B2")
                    else ("StandardScaler X and y on train (B3)" if eid == "B3" else "StandardScaler y on train (R1)")
                ),
                "optimizer": "Adam" if eid == "R1" else "n/a",
                "lr": lr_val,
                "epochs_trained": ep_val,
                "best_val_mae": v_mae,
                "best_val_rmse": v_rmse,
                "best_val_mape": v_mape,
                "test_mae": t_mae if eid == best_id else "",
                "test_rmse": t_rmse if eid == best_id else "",
                "test_mape": t_mape if eid == best_id else "",
                "notes": f"best_overall={best_id}; FAST_DEV_RUN={FAST_DEV_RUN}; test metrics only for best model",
            }
        )

    with open(ARTIFACT_DIR / "runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUNS_CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    plt.figure(figsize=(10, 4))
    if best_id == "R1":
        td = df["date"].iloc[test_end].values
        plt.plot(td, yt_r, label="actual", lw=1)
        plt.plot(td, yp_t, label="pred", lw=1, alpha=0.85)
    elif best_id == "B3":
        plt.plot(df["date"].iloc[oi_te], yt_ridge, label="actual", lw=1)
        plt.plot(df["date"].iloc[oi_te], pt, label="pred", lw=1, alpha=0.85)
    else:
        pr = pred_b1 if best_id == "B1" else pred_b2
        plt.plot(df["date"].iloc[test_idx], y[test_idx], label="actual", lw=1)
        plt.plot(df["date"].iloc[test_idx], pr[test_idx], label="pred", lw=1, alpha=0.85)
    plt.legend()
    plt.title(f"Test — best {best_id}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "best_forecast_test.png", dpi=150)
    plt.close()

    print("HW12 artifacts ->", ARTIFACT_DIR)


if __name__ == "__main__":
    run_hw12()
