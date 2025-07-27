import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import torchvision.transforms.v2 as T


# --- Copia qui dentro le classi dal tuo train.py ---
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
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.net(x)
        return self.classifier(x)

# --- Dataset ---------------------------------------------------------------
class ArousalDataset(Dataset):
    def __init__(self, records_folder, wavelet_folder, num_files=50):
        self.samples = []
        self.wavelet_folder = wavelet_folder
        self.wavelet_name = os.path.basename(wavelet_folder)
        files = sorted(glob.glob(os.path.join(records_folder, "*_p_signal.npy")))[:66]
        #print(files, records_folder, glob.glob(os.path.join(records_folder, "*_p_signal.npy")))
        targets = []
        for file_idx, signal_file in enumerate(files):
            base = signal_file.replace("_p_signal.npy", "")
            arousal = np.load(base + "_arousals.npy", mmap_mode='r')
            arousal = arousal.reshape(-1, arousal.shape[-1])[::200, :]
            label = arousal.max(-1)[::60]
            for win_idx, l in enumerate(label):
                self.samples.append((file_idx, win_idx, int(l)))
                targets.append(int(l))
        targets = torch.tensor(targets).long()
        class_sample_count = torch.bincount(targets)
        class_weights = 1.0/class_sample_count
        self.weights = class_weights[targets]
        print(f"[{self.wavelet_name}] Loaded {len(self.samples)} windows")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, win_idx, label = self.samples[idx]
        # adesso includiamo anche wavelet_name nel filename
        fn = os.path.join(
            self.wavelet_folder,
            f"{file_idx}_{win_idx}_{self.wavelet_name}_cwt.npy"
        )
        cwt = np.load(fn)
        x = torch.tensor(cwt, dtype=torch.float32).permute(0,2,1)  # (scales, ch, time)
        return x, torch.tensor(label, dtype=torch.long)

# --- Train / Eval ----------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, augment=None):
    model.train()
    losses, preds, labels = [], [], []
    for x,y in loader:
        if augment:
            x = torch.stack([augment(xx) for xx in x])
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    return np.mean(losses), (np.array(preds)==np.array(labels)).mean()

@torch.no_grad()
def eval_model(model, loader, criterion, device):
    model.eval()
    losses, preds, labels = [], [], []
    for x,y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        losses.append(criterion(out, y).item())
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    return np.mean(losses), (np.array(preds)==np.array(labels)).mean()

# --- Main loop -------------------------------------------------------------
if __name__ == "__main__":
    records_folder   = r"k:\TESI\records"
    base_wave_folder = r"C:\Users\Utente\PycharmProjects\TESI\Tesi\wavelets"
    log_root         = "runs"
    ckpt_root        = "checkpoints"

    os.makedirs(log_root, exist_ok=True)
    os.makedirs(ckpt_root, exist_ok=True)

    batch_size = 256
    lr         = 1e-3
    epochs     = 10
    num_files  = 50
    num_workers= 4
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    seed       = 0

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    augment = T.Compose([
        T.RandomHorizontalFlip(0.5),
        T.RandomAffine(degrees=0, translate=(0.1,0.1)),
        T.GaussianBlur((3,3))
    ])

    wavelets = sorted(os.listdir(base_wave_folder))
    models   = {"ArousalCNN": ArousalCNN, "DeepArousalCNN": DeepArousalCNN}

    for wname in wavelets:
        wavelet_folder = os.path.join(base_wave_folder, wname)
        for mname, Mclass in models.items():
            exper_name = f"{wname}_{mname}"
            print(f"\n>>> TRAINING {exper_name} <<<")
            log_dir  = os.path.join(log_root, exper_name)
            ckpt_dir = os.path.join(ckpt_root, exper_name)
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(ckpt_dir, exist_ok=True)

            ds = ArousalDataset(records_folder, wavelet_folder, num_files)
            n = len(ds)
            train_n = int(0.5*n)
            idxs = list(range(n)); random.shuffle(idxs)
            tr_idx, va_idx = idxs[:train_n], idxs[train_n:]
            train_sampler = WeightedRandomSampler(weights=ds.weights[tr_idx], num_samples = train_n, replacement=True)
            tr_loader = DataLoader(torch.utils.data.Subset(ds, tr_idx),
                                   batch_size=batch_size,
                                   num_workers=num_workers, pin_memory=True, sampler = train_sampler)
            va_loader = DataLoader(torch.utils.data.Subset(ds, va_idx),
                                   batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, pin_memory=True)

            sample_x, _ = ds[0]
            model = Mclass(sample_x.shape[0]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            writer    = SummaryWriter(log_dir=log_dir)

            best_val = float('inf')
            for ep in range(1, epochs+1):
                tr_loss, tr_acc = train_one_epoch(model, tr_loader, optimizer, criterion, device, augment)
                va_loss, va_acc = eval_model(model, va_loader, criterion, device)
                print(f"[{exper_name}] Ep{ep} tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f}"
                      f" | va_loss={va_loss:.4f} va_acc={va_acc:.4f}")
                writer.add_scalar("Loss/train", tr_loss, ep)
                writer.add_scalar("Loss/val",   va_loss, ep)
                writer.add_scalar("Acc/train",  tr_acc, ep)
                writer.add_scalar("Acc/val",    va_acc, ep)

                ckpt = os.path.join(ckpt_dir, f"{exper_name}_ep{ep}.pt")
                torch.save({'epoch':ep,'model':model.state_dict(),'opt':optimizer.state_dict()}, ckpt)
                if va_loss < best_val:
                    best_val = va_loss
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{exper_name}_best.pt"))

            writer.close()

    print("\n=== Fine allenamenti! ===")
