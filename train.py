import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import pywt
import torch.nn.functional as F
import tqdm
import torchvision.transforms.v2 as T
import matplotlib.pyplot as plt
import random

class ArousalDataset(Dataset):
    def __init__(self, folder, wavelet_folder, window_size=600, step=600):
        self.samples = []
        self.window_size = window_size
        self.wavelet_folder = wavelet_folder

        files = sorted(glob.glob(os.path.join(folder, "*_p_signal.npy")))[:10]
        for file_idx, signal_file in enumerate(tqdm.tqdm(files)):
            base = signal_file.replace("_p_signal.npy", "")
            arousal = np.load(base + "_arousals.npy", mmap_mode='r')  # shape: (N, window, arousal_types)
            arousal = arousal.reshape(-1, arousal.shape[-1])[::200, :]
            padding = np.zeros((window_size - 1, arousal.shape[-1]))
            arousal = np.concatenate((padding, arousal))
            label = arousal.max(-1)

            # stima numero di finestre
            n_windows = len(label) - window_size + 1
            for i in range(n_windows):
                self.samples.append((file_idx, i, label[i + window_size - 1]))

        print(f"Loaded {len(self.samples)} samples from {folder}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, window_idx, label = self.samples[idx]
        cwt_file = os.path.join(self.wavelet_folder, f"{file_idx}_{window_idx}_cwt.npy")
        cwtmatr = np.load(cwt_file)
        signal = torch.tensor(cwtmatr, dtype=torch.float32).permute((0, 2, 1))  # (wavelet_channels, channels, time)
        return signal, torch.tensor(label, dtype=torch.float32)

import torch.nn as nn
import torch.nn.functional as F

import torch
import torchvision.transforms as T
def add_noise(x, std=0.05):
    noise = torch.randn_like(x) * std
    return x + noise
class ArousalCNN(nn.Module):
    def __init__(self, input_channels):
        super(ArousalCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.net(x)
        return self.classifier(x).squeeze(1)

class DeepArousalCNN(nn.Module):
    def __init__(self, input_channels):
        super(DeepArousalCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [B, 32, 300, 6]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [B, 64, 150, 3]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1)  # -> [B, 128, 1, 1]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # [B, 128]
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.net(x)
        return self.classifier(x)



import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    T.GaussianBlur(kernel_size=(3, 3)),
])

def train(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cuda', start_epoch=0, optimizer=None):
    model = model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, epochs):
        print(f"\n[INFO] === Inizio epoca {epoch + 1}/{epochs} ===")
        model.train()
        losses = []
        for x, y in tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
            x = add_noise(x)
            x, y = x.to(device), y.to(device)
            #Data augmentation
            x = torch.stack([transform(sample) for sample in x])
            #print(f"[TRAIN] Input shape: {x.shape}, Target shape: {y.shape}")

            optimizer.zero_grad()
            pred = model(x)
            #print(f"[TRAIN] Output shape: {pred.shape}")

            loss = criterion(pred, y.long())
            print(f"\n[TRAIN] Loss: {loss.item():.4f}")

            preds_class = torch.argmax(pred, dim=1)
            correct = (preds_class == y).sum().item()
            acc = correct / y.size(0)
            print(f"\n[TRAIN] Accuracy: {acc:.4f}")



            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            #  break

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'model_{epoch}.pt')

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
        for (x, y) in tqdm.tqdm(loader):
            x = x.to(device)
            preds = model(x).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    print(all_labels, all_preds)
    pred_classes = all_preds.argmax(-1)
    acc = accuracy_score(all_labels, pred_classes)
    auc = roc_auc_score(all_labels, all_preds[:,1])
    f1 = f1_score(all_labels, pred_classes)
    print(f"Accuracy: {acc:.4f}, ROC AUC: {auc:.4f}, F1: {f1:.4f}")
    return acc


if __name__ == '__main__':
    # Parametri
    batch_size = 512
    window_size = 600  # 60s * 10Hz
    data_path = r'D:\TESI\records'
    wavelet_path = r'D:\TESI\wavelets'
    seed = 0

    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("[INFO] Preparazione dataset...")
    # Dataset
    dataset = ArousalDataset(data_path, wavelet_path, window_size=600)

    sample_input, _ = dataset[0]
    input_channels = sample_input.shape[0]
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    print(train_size, val_size)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    torch.save([train_set, val_set], r'D:\TESI\preprocessed_ds.pt')
    print('preprocessed')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=3)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=3)

    # Inizializza modello
    #model = ArousalCNN(input_channels)
    model = DeepArousalCNN(input_channels)
    model = model.to(device)

    #evaluate(model, val_loader, device)
    #evaluate(model, train_loader, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    checkpoint_path = 'model_7.pt'
    start_epoch = 7
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1  # per riprendere dal punto giusto
        print(f"[INFO] Caricato modello da {checkpoint_path}, riprendo da epoca {start_epoch}")

    else:
        print("[INFO] Nessun checkpoint trovato, si parte da zero.")

    print("[INFO] Inizio addestramento...")
    train(model, train_loader, val_loader, epochs=10, device=device, start_epoch=start_epoch, optimizer=optimizer)

