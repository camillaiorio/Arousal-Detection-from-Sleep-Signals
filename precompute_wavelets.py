import numpy as np
import os
import glob
import pywt # PyWavelets library for continuous wavelet transform
import tqdm
import os
import numpy as np
import multiprocessing
from joblib import dump, load

def compute_and_save_wavelet(args):
    i, s, idx, widths, wavelet, wavelet_dir = args
    #cA, cD = pywt.dwt(s.T, wavelet)
    #waveletmatr = np.stack([cA, cD], -1)
    cwtmatr, _ = pywt.cwt(s.T, widths, wavelet)
    cwtmatr = np.abs(cwtmatr)
    filename = os.path.join(wavelet_dir, f"{idx}_{i}_cwt.npy")
    np.save(filename, cwtmatr)
    return filename  # Optional: can be used for logging or debugging

def parallel_wavelet_transform(unfolded, idx, widths, wavelet, wavelet_dir, num_workers=None):
    os.makedirs(wavelet_dir, exist_ok=True)
    args_list = [(i, s, idx, widths, wavelet, wavelet_dir) for i, s in enumerate(unfolded)]

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(compute_and_save_wavelet, args_list)

data_path = r'D:\TESI\records'
wavelet_dir = r'D:\TESI\wavelets'
os.makedirs(wavelet_dir, exist_ok=True)

# Set wavelet function (Complex Morlet with bandwidth=1.5 and center frequency=1.0)
wavelet = 'cmor1.5-1.0'
#wavelet = 'db1'
# Define scales: geometrically spaced values between 1 and 16 (10 values total)
widths = np.geomspace(1, 16, num=10)

# Retrieve all *_p_signal.npy files
signal_files = sorted(glob.glob(os.path.join(data_path, "*_p_signal.npy")))#[:10]
print(signal_files[:100])
exit(1)
if __name__ == '__main__':
    # Loop over each signal file
    for idx, signal_file in enumerate(tqdm.tqdm(signal_files, desc="Precomputing wavelets")):
        base = signal_file.replace("_p_signal.npy", "")
        signal = np.load(base + "_p_signal.npy", mmap_mode='r')  # shape: (N, window, channels)
        print(signal.shape)

        # Downsample and flatten the signal
        seq, window, channels = signal.shape
        flat_signal = signal.reshape(-1, channels)[::200, :]
        #print(flat_signal.shape)
        #exit(1)
        # Pad the signal to prepare for windowing
        padding = np.zeros((600 - 1, channels))  # 600 = window_size (6 seconds at 100 Hz)
        flat_signal = np.concatenate((padding, flat_signal))
        # Extract sliding windows of size (600, channels)
        unfolded = np.lib.stride_tricks.sliding_window_view(flat_signal, window_shape=(600, channels)).squeeze()
        unfolded = unfolded[::60]
        # Compute and save the wavelet transform for each window
        #for i, s in tqdm.tqdm(list(enumerate(unfolded))):
            #cA, cD = pywt.dwt(s.T, wavelet, level=2)
            #waveletmatr = np.stack([cA, cD], -1)
            #cwtmatr, _ = pywt.cwt(s.T, widths, wavelet)
            # cwtmatr = np.abs(cwtmatr)
            #filename = os.path.join(wavelet_dir, f"{idx}_{i}_dwt.npy")
            #np.save(filename, waveletmatr)
            #cwtmatr, _ = pywt.cwt(s.T, widths, wavelet)
            #cwtmatr = np.abs(cwtmatr)  # shape: (wavelet_channels, time, input_channels)
            #l.append(cwtmatr)
            #filename = os.path.join(wavelet_dir, f"{idx}_{i}_cwt.npy")
            #dump(cwtmatr, filename)
            #np.save(filename, cwtmatr)
        #print(len(unfolded))
        parallel_wavelet_transform(unfolded, idx, widths, wavelet, wavelet_dir, num_workers=4)

