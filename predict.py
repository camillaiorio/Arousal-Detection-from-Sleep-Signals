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
from torchvision.models import resnet18, ResNet18_Weights
import random

class ArousalDataset(Dataset):
    def __init__(self, folder, wavelet_folder, window_size=600, step=600, files_ids=[]):
        self.samples = []
        self.window_size = window_size
        self.wavelet_folder = wavelet_folder

        files = sorted(glob.glob(os.path.join(folder, "*_p_signal.npy")))[:10]
        for file_idx, signal_file in enumerate(tqdm.tqdm(files)):
            if file_idx not in files_ids: continue
            base = signal_file.replace("_p_signal.npy", "")
            # Load arousal labels and flatten
            arousal = np.load(base + "_arousals.npy", mmap_mode='r')  # shape: (N, window, arousal_types)
            arousal = arousal.reshape(-1, arousal.shape[-1])[::200, :]
            padding = np.zeros((window_size - 1, arousal.shape[-1]))
            arousal = np.concatenate((padding, arousal))
            label = arousal.max(-1)

            # Compute number of valid windows
            n_windows = len(label) - window_size + 1
            for i in range(n_windows):
                self.samples.append((file_idx, i, label[i + window_size - 1]))

        print(f"Loaded {len(self.samples)} samples from {folder}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, window_idx, label = self.samples[idx]
        cwt_file = os.path.join(self.wavelet_folder, f"{file_idx}_{window_idx}_cwt.npy")
        print(cwt_file)
        cwtmatr = np.load(cwt_file)
        signal = torch.tensor(cwtmatr, dtype=torch.float32).permute((0, 2, 1))  # (wavelet_channels, channels, time)
        return signal, torch.tensor(label, dtype=torch.float32)

class ArousalDataset(Dataset):
    def __init__(self, folder, wavelet_folder, window_size=600, step=600, files_ids=None):
        self.samples = []
        self.window_size = window_size
        self.wavelet_folder = wavelet_folder

        files = sorted(glob.glob(os.path.join(folder, "*_p_signal.npy")))[:100]
        for file_idx, signal_file in enumerate(tqdm.tqdm(files)):
            if files_ids is not None and file_idx not in files_ids: continue
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

def predict(model, loader, device='cuda'):
    print("[INFO] Valutazione in corso...")
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for (x, y) in tqdm.tqdm(loader):
            try:
                x, y = x.to(device), y.to(device)

                # Ridimensiono per ResNet
                #x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

                preds = model(x).softmax(-1)

                all_preds.extend(preds.cpu().numpy()[:,1])
                all_trues.extend(y.cpu().numpy())
            except:
                continue
    return np.array(all_preds).tolist(), np.array(all_trues).tolist()

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
    model_name = 'model_ArousalCNN9'
    os.makedirs('predictions/'+model_name, exist_ok=True)
    checkpoint_path = model_name+'.pt'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("[INFO] Preparazione dataset...")
    files = sorted(glob.glob(os.path.join(data_path, "*_p_signal.npy")))[50:]
    n_files = len(files)

    model = None
    for file_idx in range(n_files):
        print(files[file_idx])
        # Dataset
        dataset = ArousalDataset(data_path, wavelet_path, window_size=600, files_ids=[file_idx])

        sample_input, _ = dataset[0]
        input_channels = sample_input.shape[0]


        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        if model is None:
            model = ArousalCNN(input_channels)
            model = model.to(device)


            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])

        prediction, trues = predict(model, loader, device)
        with open(f'predictions\\{model_name}\\'+files[file_idx].split('\\')[-1]+'_pred.txt', 'w') as f:
            f.write('\n'.join([str(p) for p in prediction]))
        with open(f'predictions\\{model_name}\\'+files[file_idx].split('\\')[-1]+'_true.txt', 'w') as f:
            f.write('\n'.join([str(p) for p in trues]))


