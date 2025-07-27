import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import shutil  # <=== IMPORT PER CANCELLARE CARTELLE


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
            nn.AdaptiveMaxPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.classifier(self.net(x))


class DeepArousalCNN(nn.Module):
    def __init__(self, input_channels):
        super(DeepArousalCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.classifier(self.net(x))


class ArousalPredictDataset(Dataset):
    def __init__(self, records_folder, wavelet_folder, file_idx,
                 window_size=600, downsample=200, step=60):
        self.samples = []
        self.wavelet_folder = wavelet_folder
        wavelet_name = os.path.basename(wavelet_folder)

        all_signals = sorted(glob.glob(os.path.join(records_folder, "*_p_signal.npy")))
        sig = all_signals[file_idx]
        base = sig.replace("_p_signal.npy", "")

        # preparo i label
        arousal = np.load(base + "_arousals.npy", mmap_mode='r')
        arousal = arousal.reshape(-1, arousal.shape[-1])[::downsample, :]
        padding = np.zeros((window_size - 1, arousal.shape[-1]))
        arousal = np.concatenate((padding, arousal), axis=0)
        label = arousal.max(-1)[::step]

        # filtro solo le finestre per cui esiste il .npy
        for i, l in enumerate(label):
            fn = os.path.join(
                wavelet_folder,
                f"{file_idx}_{i}_{wavelet_name}_cwt.npy"
            )
            if os.path.exists(fn):
                self.samples.append((file_idx, i, int(l)))

        print(f"[Predict {wavelet_name}] {os.path.basename(sig)} -> {len(self.samples)} windows")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, win_idx, label = self.samples[idx]
        wavelet_name = os.path.basename(self.wavelet_folder)
        fn = os.path.join(
            self.wavelet_folder,
            f"{file_idx}_{win_idx}_{wavelet_name}_cwt.npy"
        )
        cwt = np.load(fn)
        x = torch.tensor(cwt, dtype=torch.float32).permute(0, 2, 1)  # (scales, ch, time)
        return x, torch.tensor(label, dtype=torch.long)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x).softmax(dim=1)[:, 1].cpu().numpy()
        preds.extend(out.tolist())
        trues.extend(y.numpy().tolist())
    return preds, trues


if __name__ == "__main__":
    records_folder = r"k:\TESI\records"
    base_wave_folder = r"C:\Users\Utente\PycharmProjects\TESI\Tesi\wavelets"
    ckpt_root = "checkpoints"
    pred_root = "predictions"
    os.makedirs(pred_root, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 512

    all_signals = sorted(glob.glob(os.path.join(records_folder, "*_p_signal.npy")))[66:99]
    wavelets = sorted([d for d in os.listdir(base_wave_folder)
                       if os.path.isdir(os.path.join(base_wave_folder, d))])
    models_map = {"ArousalCNN": ArousalCNN, "DeepArousalCNN": DeepArousalCNN}

    for w in wavelets:
        wavelet_folder = os.path.join(base_wave_folder, w)
        for mname, Mclass in models_map.items():
            exp_name = f"{w}_{mname}"
            best_ckpt = os.path.join(ckpt_root, exp_name, f"{exp_name}_best.pt")
            if not os.path.exists(best_ckpt):
                print(f"[WARN] Missing checkpoint for {exp_name}, skipping")
                continue

            print(f"\n>>> Predicting with {exp_name} <<<")

            # Cancella eventuali predizioni precedenti
            out_dir = os.path.join(pred_root, exp_name)
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)

            # Carico modello
            ds0 = ArousalPredictDataset(records_folder, wavelet_folder, file_idx=50)
            sample_x, _ = ds0[0]
            model = Mclass(sample_x.shape[0]).to(device)
            state = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(state)

            total_windows = 0

            for relative_idx, sig in enumerate(all_signals):
                true_idx = 66 + relative_idx

                fn0 = os.path.join(wavelet_folder, f"{true_idx}_0_{w}_cwt.npy")
                if not os.path.exists(fn0):
                    continue

                ds = ArousalPredictDataset(records_folder, wavelet_folder, true_idx)
                if len(ds) == 0:
                    continue

                loader = DataLoader(ds, batch_size=batch_size,
                                    shuffle=False, num_workers=4)
                preds, trues = predict(model, loader, device)

                base_name = os.path.basename(sig).replace("_p_signal.npy", "")
                with open(os.path.join(out_dir, base_name + "_pred.txt"), 'w') as f:
                    f.write("\n".join(map(str, preds)))
                with open(os.path.join(out_dir, base_name + "_true.txt"), 'w') as f:
                    f.write("\n".join(map(str, trues)))

                total_windows += len(ds)

            print(f"[INFO] Totale finestre predette per {exp_name}: {total_windows}")

    print("\n=== Tutte le predizioni generate! ===")
