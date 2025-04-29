import numpy as np
import os
import glob
import pywt
import tqdm

data_path = r'D:\TESI\records'
wavelet_dir = r'D:\TESI\wavelets'
os.makedirs(wavelet_dir, exist_ok=True)

wavelet = 'cmor1.5-1.0'
widths = np.geomspace(1, 16, num=10)

# Retrieve all *_p_signal.npy files
signal_files = sorted(glob.glob(os.path.join(data_path, "*_p_signal.npy")))[:10]

# Loop over each signal file
for idx, signal_file in enumerate(tqdm.tqdm(signal_files, desc="Precomputing wavelets")):
    base = signal_file.replace("_p_signal.npy", "")
    signal = np.load(base + "_p_signal.npy", mmap_mode='r')  # shape: (N, window, channels)

    # Downsample and flatten the signal
    seq, window, channels = signal.shape
    flat_signal = signal.reshape(-1, channels)[::200, :]
    # Pad the signal to prepare for windowing
    padding = np.zeros((600 - 1, channels))  # 600 = window_size
    flat_signal = np.concatenate((padding, flat_signal))
    # Extract sliding windows of size (600, channels)
    unfolded = np.lib.stride_tricks.sliding_window_view(flat_signal, window_shape=(600, channels)).squeeze()

    # Compute and save the wavelet transform for each window
    for i, s in enumerate(unfolded):
        cwtmatr, _ = pywt.cwt(s.T, widths, wavelet)
        cwtmatr = np.abs(cwtmatr)  # shape: (wavelet_channels, time, input_channels)
        filename = os.path.join(wavelet_dir, f"{idx}_{i}_cwt.npy")
        np.save(filename, cwtmatr)
