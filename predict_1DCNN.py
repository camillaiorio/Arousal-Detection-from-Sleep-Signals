import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.nn as nn

class ArousalCNN(nn.Module):
    def __init__(self, input_channels):
        super(ArousalCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.net(x)
        return self.classifier(x).squeeze(1)


class ArousalDataset(Dataset):
    def __init__(self, folder, window_size=600, step=60):
        self.samples = []
        self.window_size = window_size
        self.step = step

        files = sorted(glob.glob(os.path.join(folder, "*_p_signal.npy")))[50:]
        for file_idx, signal_file in enumerate(tqdm(files, desc="[Dataset Load]")):
            base = signal_file.replace("_p_signal.npy", "")
            signal = np.load(base + "_p_signal.npy", mmap_mode='r')
            arousal = np.load(base + "_arousals.npy", mmap_mode='r')

            seq, seq_len, f = signal.shape
            signal = signal.reshape(-1, f)[::200, :]
            arousal = arousal.reshape(-1, arousal.shape[-1])[::200, :]

            padding = np.zeros((self.window_size - 1, f))
            signal = np.concatenate((padding, signal))
            signal = torch.tensor(signal).unfold(0, self.window_size, self.step).numpy()
            label = arousal.max(-1)
            label = label[self.window_size - 1:]  # allineamento con padding
            label = label[::self.step]  # downsampling come nelle 2D CNN

            for i in range(len(label)):
                self.samples.append((file_idx, i, signal[i], label[i]))

            # min_len = min(len(signal), len(label))
            # for i in range(min_len):
            #     self.samples.append((file_idx, i, signal[i], label[i]))

        print(f"Loaded {len(self.samples)} samples from {folder}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, _, sig, label = self.samples[idx]
        signal = torch.tensor(sig, dtype=torch.float32)  # shape: (channels, time)
        return signal, torch.tensor(label, dtype=torch.float32)

# ===== PREDICT FUNCTION =====

def predict(model, loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="[Predict]"):
            x = x.to(device)
            preds = model(x).softmax(-1).cpu().numpy()
            all_preds.extend(preds[:, 1])
            all_labels.extend(y.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)

if __name__ == '__main__':
    batch_size = 512
    window_size = 600
    step = 60
    data_path = r'D:\TESI\records'
    checkpoint_path = 'model_1DCNN9.pt'
    model_name = 'model_1DCNN'
    os.makedirs(f'predictions/{model_name}', exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("[INFO] Preparazione dataset...")

    dataset = ArousalDataset(data_path, window_size=600, step=60)
    sample_input, _ = dataset[0]
    input_channels = sample_input.shape[0]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = ArousalCNN(input_channels).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"[INFO] Modello caricato da {checkpoint_path}")

    predictions, labels = predict(model, loader, device)

    np.savetxt(f'predictions/{model_name}/predictions.txt', predictions, fmt='%.4f')
    np.savetxt(f'predictions/{model_name}/labels.txt', labels, fmt='%.0f')
    print(f"[INFO] Predizioni salvate in predictions/{model_name}/")