import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import torch.nn.functional as F

class ArousalDataset(Dataset):
    def __init__(self, folder, window_size=600, step=600):
        self.samples = []
        self.window_size = window_size

        files = sorted(glob.glob(os.path.join(folder, "*_p_signal.npy")))[:1]
        for signal_file in files:
            base = signal_file.replace("_p_signal.npy", "")
            signal = np.load(base + "_p_signal.npy",mmap_mode='r')  # shape: (N, window, channels)
            arousal = np.load(base + "_arousals.npy",mmap_mode='r')  # shape: (N, window, arousal_types)

            seq, seq_len, f = signal.shape
            seq, seq_len, c = arousal.shape

            signal = signal.reshape(-1, f)[::200,:]
            arousal = arousal.reshape(-1, c)[::200,:]

            padding = np.zeros((self.window_size-1, f))
            signal = np.concatenate((padding, signal))
            signal = torch.tensor(signal).unfold(0, self.window_size, 1).numpy()

            label = arousal.max(-1)

            for s, l in zip(signal, label):
                self.samples.append((s, l))


        print(f"Loaded {len(self.samples)} samples from {folder}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        signal, label = self.samples[idx]
        signal = torch.tensor(signal, dtype=torch.float32)  # shape: (channels, time)
        if idx < 3:
            print(f"[DEBUG] Campione {idx} - Signal shape: {signal.shape}, Label: {label}")

        return signal, torch.tensor(label, dtype=torch.float32)
import torch.nn as nn
import torch.nn.functional as F

class ArousalCNN(nn.Module): #TODO aumentare la profondità
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


import torch
from sklearn.metrics import accuracy_score, roc_auc_score


def train(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"\n[INFO] === Inizio epoca {epoch + 1}/{epochs} ===")
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            print(f"[TRAIN] Input shape: {x.shape}, Target shape: {y.shape}")
            optimizer.zero_grad()
            pred = model(x)
            print(f"[TRAIN] Output shape: {pred.shape}")
            loss = criterion(pred, y.long())
            print(f"[TRAIN] Loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        mean_loss = np.mean(losses)
        print(f"[INFO] === Fine epoca {epoch + 1} - Loss medio: {mean_loss:.4f} ===")
        acc = evaluate(model, val_loader, device=device)
        print(f"[INFO] Accuracy su validation set: {acc:.4f}")


def evaluate(model, loader, device='cuda'):
    print("[INFO] Valutazione in corso...")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for (x, y) in loader:
            x = x.to(device)
            preds = model(x).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_classes = all_preds.argmax(-1)
    acc = accuracy_score(all_labels, pred_classes)
    auc = roc_auc_score(all_labels, all_preds[:,1])
    print(f"Accuracy: {acc:.4f}, ROC AUC: {auc:.4f}")
    return acc

if __name__ == '__main__':
    # Parametri
    batch_size = 32
    window_size = 600  # 60s * 10Hz
    data_path = r'D:\TESI\records'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("[INFO] Preparazione dataset...")
    # Dataset
    dataset = ArousalDataset(data_path, window_size)

    sample_input, _ = dataset[0]
    input_channels = sample_input.shape[0]
    print(input_channels)
    exit(1)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    #torch.save([train_set, val_set], r'D:\TESI\preprocessed_ds.pt')
    #print('preprocessed')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # Inizializza modello



    model = ArousalCNN(input_channels)

    # Train
    #print("[INFO] Inizio addestramento...")
    train(model, train_loader, val_loader, epochs=10, device=device)
