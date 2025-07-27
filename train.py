import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import pywt
import torch.nn.functional as F
import tqdm
import torchvision.transforms.v2 as T
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import random
from torchvision.models import resnet18, ResNet18_Weights

class ArousalDataset(Dataset):
    def __init__(self, folder, wavelet_folder, window_size=600, step=600):
        self.samples = []
        self.window_size = window_size
        self.wavelet_folder = wavelet_folder

        files = sorted(glob.glob(os.path.join(folder, "*_p_signal.npy")))[:66]
        for file_idx, signal_file in enumerate(tqdm.tqdm(files)):
            base = signal_file.replace("_p_signal.npy", "")
            # Load arousal labels and flatten
            arousal = np.load(base + "_arousals.npy", mmap_mode='r')  # shape: (N, window, arousal_types)
            arousal = arousal.reshape(-1, arousal.shape[-1])[::200, :]
            #padding = np.zeros((window_size - 1, arousal.shape[-1]))
            #arousal = np.concatenate((padding, arousal))
            label = arousal.max(-1)

            ###
            #signal = np.load(base + "_p_signal.npy", mmap_mode='r')  # shape: (N, window, channels)
            #seq, window, channels = signal.shape
            #flat_signal = signal.reshape(-1, channels)[::200, :]
            #padding = np.zeros((600 - 1, channels))  # 600 = window_size (6 seconds at 100 Hz)
            #flat_signal = np.concatenate((padding, flat_signal))
            #unfolded = np.lib.stride_tricks.sliding_window_view(flat_signal, window_shape=(600, channels)).squeeze()
            #unfolded = unfolded[::60]
            ###

            label = label[::60]

            # Compute number of valid windows
            n_windows = len(label)
            for i in range(n_windows):
                self.samples.append((file_idx, i, label[i]))
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

def get_resnet18(input_channels: int, num_classes: int = 2) -> nn.Module:
    # Carico ResNet18 con i pesi di ImageNet
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    #riconfiguriamo conv1
    if input_channels != 3:
        orig = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=orig.out_channels,
            kernel_size=orig.kernel_size,
            stride=orig.stride,
            padding=orig.padding,
            bias=False
        )
    #Sostituisco la testa finale per avere un num_classes-output
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
# Augmentation pipeline
transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    T.GaussianBlur(kernel_size=(3, 3)),
])

def train(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cuda', start_epoch=0, optimizer=None, patience=3):
    writer = SummaryWriter()

    model = model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(start_epoch, epochs):
        print(f"\n[INFO] === Inizio epoca {epoch + 1}/{epochs} ===")
        model.train()
        losses = []
        all_preds = []
        all_labels = []
        #print(model)
        for x, y in tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
            x = add_noise(x)
            x, y = x.to(device), y.to(device)
            x = torch.stack([transform(sample) for sample in x])

            # Ridimensiono per ResNet
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y.long())

            preds_class = torch.argmax(pred, dim=1)
            correct = (preds_class == y).sum().item()
            acc = correct / y.size(0)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            all_preds.extend(preds_class.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        train_loss = np.mean(losses)
        train_acc = accuracy_score(all_labels, all_preds)

        val_acc, val_loss = evaluate(model, val_loader, device=device, return_loss=True)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)

        # Salvataggio modello corrente
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'model_ResNet{epoch}.pt')

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pt')  # Salva il modello migliore
            print(f"[INFO] ➤ Miglioramento val loss ({val_loss:.4f}), modello salvato.")
        else:
            epochs_no_improve += 1
            print(f"[INFO] ➤ Nessun miglioramento ({epochs_no_improve}/{patience})")

            if epochs_no_improve >= patience:
                print(f"[INFO] ➤ Early stopping attivato. Val loss non migliora da {patience} epoche.")
                break

        print(f"[INFO] === Fine epoca {epoch + 1} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} ===")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

    # Grafici
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(rf"D:\TESI\loss_epoch_{epoch}.png")
    plt.close()

    plt.figure()
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(rf"D:\TESI\accuracy_epoch_{epoch}.png")
    plt.close()

    writer.close()


def evaluate(model, loader, device='cuda', return_loss=False):
    print("[INFO] Valutazione in corso...")
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for (x, y) in tqdm.tqdm(loader):
            # Ridimensiono per ResNet
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y.long())
            total_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_classes = all_preds.argmax(-1)
    acc = accuracy_score(all_labels, pred_classes)

    if return_loss:
        return acc, total_loss / len(loader)
    else:
        print(f"Accuracy: {acc:.4f}")
        return acc

if __name__ == '__main__':
    # Parametri
    batch_size = 512
    window_size = 600  # 60s * 10Hz
    data_path = r'K:\TESI\records'
    wavelet_path = r'K:\TESI\wavelets'
    seed = 0

    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("[INFO] Preparazione dataset...")
    # Dataset
    dataset = ArousalDataset(data_path, wavelet_path, window_size=600)
    sample_input, _ = dataset[0]
    print(sample_input.shape)

    # Split into training and validation sets
    train_size = int(0.5 * len(dataset))
    val_size = len(dataset) - train_size

    print(train_size, val_size)
    #train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_set = torch.utils.data.Subset(dataset, range(train_size))
    val_set = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    torch.save([train_set, val_set], r'D:\TESI\preprocessed_ds.pt')
    print('preprocessed')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=3)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=3)

    # Initialize model
    #model = ArousalCNN(input_channels)
    #model = DeepArousalCNN(input_channels)
    model = get_resnet18(input_channels, num_classes=2)
    model = model.to(device)

    #evaluate(model, val_loader, device)
    #evaluate(model, train_loader, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    checkpoint_path = 'model_10.pt'
    start_epoch = 0
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

