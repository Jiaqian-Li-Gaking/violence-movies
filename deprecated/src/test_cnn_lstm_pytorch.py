import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# reuse Conv3D_LSTM from above
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

        self.lstm_units = lstm_units
        # After 3 pools with input (T,H,W) = (10,64,64):
        # (T,H,W) -> (1,8,8), so feature_dim = conv_filters * 8 * 8
        feature_dim = conv_filters * 8 * 8
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=self.lstm_units,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(lstm_units, 256)
        self.fc2 = nn.Linear(256, num_classes)

    # def _ensure_lstm(self, feature_dim, device):
    #     if self.lstm is None:
    #         self.lstm = nn.LSTM(
    #             input_size=feature_dim,
    #             hidden_size=self.lstm_units,
    #             batch_first=True
    #         ).to(device)

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

        # Move time to axis 1 and flatten spatial dims
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (N, T', F, H', W')
        x = x.view(n, t, f * h * w)                # (N, T', feature_dim)

        out, _ = self.lstm(x)                      # (N, T', lstm_units)
        x = out[:, -1, :]                          # last time step


        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

def preprocess_video_to_tensor(video_path, frame_count=10, frame_size=(64, 64)):
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while len(frames) < frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
        frames.append(frame.astype(np.float32))
    cap.release()

    if len(frames) == 0:
        frames = [np.zeros((frame_size[1], frame_size[0], 3), dtype=np.float32)] * frame_count
    if len(frames) < frame_count:
        frames.extend([np.zeros_like(frames[0])] * (frame_count - len(frames)))

    x = np.stack(frames, axis=0) / 255.0      # (T,H,W,C)
    x = np.transpose(x, (3, 0, 1, 2))         # (C,T,H,W)
    return x.astype(np.float32)

def load_test_videos(folder_path: Path, frame_count=10, frame_size=(64, 64)):
    video_files = [p for p in folder_path.iterdir() if p.suffix.lower() in [".mp4", ".avi"]]
    X_list, names = [], []
    for vp in sorted(video_files):
        X_list.append(preprocess_video_to_tensor(vp, frame_count, frame_size))
        names.append(vp.name)
    X = np.stack(X_list, axis=0) if X_list else np.zeros((0,3,frame_count,frame_size[1],frame_size[0]), dtype=np.float32)
    return X, names

def main():
    try:
        BASE_DIR = Path(__file__).resolve().parent.parent
    except NameError:
        BASE_DIR = Path.cwd().parent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # test_video_folder = BASE_DIR / "data" / "raw" / "Real Life Violence Dataset" / "Violence"
    # test_video_folder = BASE_DIR / "data" / "processed" / "violence-detection-dataset" / "low-level violence" / "cam2"
    test_video_folder = BASE_DIR / "data" / "processed" / "violence-detection-dataset" / "non-violence" / "cam2"
    video_files = sorted([p for p in test_video_folder.iterdir() if p.suffix.lower() in [".mp4", ".avi"]])

    model = Conv3D_LSTM(num_classes=3, conv_filters=64, lstm_units=64, dropout_rate=0.5).to(device)
    weight_path = BASE_DIR / "train_weight" / "violence_detection_model_conv_64_lstm_64.pt"
    state = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    class_names = ["high-level violence", "low-level violence", "non-violence"]

    batch_size = 4  # try 1,2,4; reduce if still OOM
    frame_count = 10
    frame_size = (64, 64)

    preds = []

    with torch.inference_mode():
        for i in range(0, len(video_files), batch_size):
            batch_paths = video_files[i:i+batch_size]

            # build one small batch on CPU
            batch_np = np.stack(
                [preprocess_video_to_tensor(p, frame_count, frame_size) for p in batch_paths],
                axis=0
            )  # (B,3,T,H,W)

            xb = torch.from_numpy(batch_np).to(device, non_blocking=True)

            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()

            for pth, c in zip(batch_paths, pred):
                print(f"Video: {pth.name}, Predicted Class: {class_names[c]}")

            # free GPU tensor ASAP
            del xb, logits
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
