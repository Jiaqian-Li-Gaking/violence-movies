import os
from pathlib import Path
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

# ----------------------------
# Paths
# ----------------------------
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    # Jupyter notebook fallback
    BASE_DIR = Path.cwd().parent

base_path = BASE_DIR / "data" / "processed" / "violence-detection-dataset"

folders = ["high-level violence_frames", "low-level violence_frames", "non-violence_frames"]
labels = {"high-level violence_frames": 0, "low-level violence_frames": 1, "non-violence_frames": 2}

# ----------------------------
# Data loading (matches your logic)
# - read first `frame_count` images per sample folder
# - pad with zeros if fewer than frame_count
# - normalize to [0,1]
# Returns:
#   data: (N, T, H, W, C)
#   targets: (N,)
# ----------------------------
def load_data(folders, labels, base_path, frame_count=10, img_size=(64, 64)):
    data, targets = [], []

    for folder in folders:
        class_dir = base_path / folder
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing folder: {class_dir}")

        # Each subfolder is assumed to be one sample (sequence of frames)
        sample_dirs = [p for p in class_dir.iterdir() if p.is_dir()]
        sample_dirs.sort()

        for sample_dir in sample_dirs:
            frame_files = sorted([p for p in sample_dir.iterdir()
                                  if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

            frames = []
            for fp in frame_files[:frame_count]:
                img = cv2.imread(str(fp))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
                frames.append(img.astype(np.float32))

            if len(frames) == 0:
                # If a sample folder is empty/broken, skip it (safer than crashing)
                continue

            if len(frames) < frame_count:
                pad = [np.zeros_like(frames[0], dtype=np.float32)] * (frame_count - len(frames))
                frames.extend(pad)

            data.append(np.stack(frames, axis=0))        # (T,H,W,C)
            targets.append(labels[folder])

    data = np.stack(data, axis=0) / 255.0               # (N,T,H,W,C), float32
    targets = np.array(targets, dtype=np.int64)
    return data.astype(np.float32), targets


# ----------------------------
# PyTorch Dataset
# Converts (T,H,W,C) -> (C,T,H,W) for Conv3D
# ----------------------------
class VideoFramesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]  # (T,H,W,C)
        x = np.transpose(x, (3, 0, 1, 2))  # (C,T,H,W)
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)


# ----------------------------
# Model: Conv3D blocks + LSTM + FC
# ----------------------------
class Conv3D_LSTM(nn.Module):
    def __init__(self, num_classes=3, conv_filters=32, lstm_units=64, dropout_rate=0.5):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels=3, out_channels=conv_filters, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv3d(conv_filters, conv_filters, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv3d(conv_filters, conv_filters, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # LSTM is created lazily once we know feature dimension after convs
        self.lstm_units = lstm_units
        self.lstm = None

        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(lstm_units, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def _ensure_lstm(self, feature_dim, device):
        if self.lstm is None:
            self.lstm = nn.LSTM(
                input_size=feature_dim,
                hidden_size=self.lstm_units,
                batch_first=True
            ).to(device)

    def forward(self, x):
        # x: (N,C,T,H,W)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # x: (N, F, T', H', W')
        n, f, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()    # (N, T', F, H', W')
        x = x.view(n, t, f * h * w)                  # (N, T', feature_dim)

        self._ensure_lstm(x.size(-1), x.device)
        out, (hn, cn) = self.lstm(x)                 # out: (N,T',lstm_units)
        x = out[:, -1, :]                            # last time step

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


# ----------------------------
# Training utilities
# ----------------------------
def train_one_fold(model, train_loader, val_loader, device, epochs=50, lr=1e-4, patience=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # validate
        model.eval()
        val_losses = []
        all_probs = []
        all_true = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())

                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                all_probs.append(probs)
                all_true.append(yb.detach().cpu().numpy())

        val_loss = float(np.mean(val_losses))
        y_true = np.concatenate(all_true)
        y_prob = np.concatenate(all_probs)
        y_pred = np.argmax(y_prob, axis=1)

        f1 = f1_score(y_true, y_pred, average="macro")
        val_acc = (y_pred == y_true).mean()


        # -------- PRINT --------
        print(
            f"Epoch [{epoch:02d}/{epochs}] | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_acc:.4f} | "
            f"F1: {f1:.4f}"
        )


        # multi-class AUC (one-vs-rest). If a fold is missing a class, handle safely.
        try:
            auc_ovr = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except ValueError:
            auc_ovr = float("nan")

        # early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_loss, f1, auc_ovr





def main():
    # ===== everything from "data, targets = load_data(...)" onwards =====
    data, targets = load_data(folders, labels, base_path, frame_count=10, img_size=(64, 64))
    print("Data shape:", data.shape)       # (N,10,64,64,3)
    print("Targets shape:", targets.shape) # (N,)

    # ----------------------------
    # Grid + 10-fold CV
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    input_shape = (10, 64, 64, 3)  # (T,H,W,C) just for reference
    num_classes = 3
    conv_filters_list = [32, 64, 128]
    lstm_units_list = [64, 128, 256]

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    results = {}

    for conv_filters in conv_filters_list:
        for lstm_units in lstm_units_list:
            fold_f1 = []
            fold_auc = []
            fold_vloss = []
            print(f"\n=== Training: conv={conv_filters}, lstm={lstm_units} ===")

            for fold_id, (train_idx, val_idx) in enumerate(kf.split(data, targets), 1):
                print(f"\n--- Fold {fold_id}/10 ---")
                X_train, y_train = data[train_idx], targets[train_idx]
                X_val, y_val = data[val_idx], targets[val_idx]

                train_ds = VideoFramesDataset(X_train, y_train)
                val_ds = VideoFramesDataset(X_val, y_val)

                train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
                val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

                model = Conv3D_LSTM(
                    num_classes=num_classes,
                    conv_filters=conv_filters,
                    lstm_units=lstm_units,
                    dropout_rate=0.5
                ).to(device)

                vloss, f1, auc_ovr = train_one_fold(
                    model, train_loader, val_loader,
                    device=device, epochs=50, lr=1e-4, patience=5
                )

                fold_vloss.append(vloss)
                fold_f1.append(f1)
                fold_auc.append(auc_ovr)

            key = f"conv_{conv_filters}_lstm_{lstm_units}"
            results[key] = {
                "mean_val_loss": float(np.nanmean(fold_vloss)),
                "mean_f1_macro": float(np.nanmean(fold_f1)),
                "mean_auc_ovr": float(np.nanmean(fold_auc)),
            }


            # Train a final model on all data (optional) and save weights
            print("Train a final model on all data and save weights")
            full_ds = VideoFramesDataset(data, targets)
            full_loader = DataLoader(full_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)

            final_model = Conv3D_LSTM(num_classes, conv_filters, lstm_units, dropout_rate=0.5).to(device)
            # quick full training (no CV) just to produce a saved model
            _ = train_one_fold(final_model, full_loader, full_loader, device=device, epochs=50, lr=1e-4, patience=5)

            save_dir = BASE_DIR / "train_weight"
            save_dir.mkdir(parents=True, exist_ok=True)

            save_path = save_dir / f"violence_detection_model_conv_{conv_filters}_lstm_{lstm_units}.pt"
            torch.save(final_model.state_dict(), save_path)

            print(f"\n[{key}] saved -> {save_path}")
            print("  mean F1(macro):", results[key]["mean_f1_macro"])
            print("  mean AUC(ovr): ", results[key]["mean_auc_ovr"])
            print("  mean val loss: ", results[key]["mean_val_loss"])

    print("\nAll results:")
    for k, v in results.items():
        print(k, v)



if __name__ == "__main__":
    # needed on Windows when using multiple workers
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
