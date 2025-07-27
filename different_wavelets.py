import os
import glob
import numpy as np
import pywt
import tqdm
import multiprocessing

def compute_and_save_wavelet(args):
    i, window_data, file_idx, widths, wavelet_name, out_dir = args
    cwtmatr, _ = pywt.cwt(window_data.T, widths, wavelet_name)
    cwtmatr = np.abs(cwtmatr)
    fname = os.path.join(out_dir, f"{file_idx}_{i}_{wavelet_name}_cwt.npy")
    np.save(fname, cwtmatr)
    return fname

def parallel_wavelet_transform(unfolded, file_idx, widths, wavelet_name, out_dir, num_workers):
    os.makedirs(out_dir, exist_ok=True)
    args = [
        (i, seg, file_idx, widths, wavelet_name, out_dir)
        for i, seg in enumerate(unfolded)
    ]
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(compute_and_save_wavelet, args)

def preprocess_signal(signal_path, downsample=200, window_size=600, step=60):
    signal = np.load(signal_path, mmap_mode='r')
    seq, win, ch = signal.shape
    flat = signal.reshape(-1, ch)[::downsample, :]
    pad = np.zeros((window_size - 1, ch))
    flat = np.concatenate((pad, flat), axis=0)
    unfolded = np.lib.stride_tricks.sliding_window_view(
        flat, window_shape=(window_size, ch)
    ).squeeze()[::step]
    return unfolded

def generate_all_wavelets(
    data_folder,
    base_out_folder,
    wavelet_list,
    widths,
    num_files=50,
    num_workers=4
):
    signal_files = sorted(glob.glob(os.path.join(data_folder, "*_p_signal.npy"))) [:num_files]
    for wavelet_name in wavelet_list:
        print(f"\n=== Generating CWT with wavelet: {wavelet_name} ===")
        out_dir = os.path.join(base_out_folder, wavelet_name)
        for file_idx, sig_path in enumerate(tqdm.tqdm(signal_files, desc=wavelet_name)):
            unfolded = preprocess_signal(sig_path,
                                         downsample=200,
                                         window_size=600,
                                         step=60)
            parallel_wavelet_transform(
                unfolded,
                file_idx,
                widths,
                wavelet_name,
                out_dir,
                num_workers
            )

if __name__ == "__main__":
    # Percorso dei dati
    data_folder     = r"K:\TESI\records"
    # Qui sotto: in C:\Users\Utente\PycharmProjects\TESI\Tesi\wavelets
    base_out_folder = r"C:\Users\Utente\PycharmProjects\TESI\Tesi\wavelets"

    wavelet_list = [
        'cmor0.5-1.0', 'cmor1.0-1.0', 'cmor1.5-1.0',
        'mexh',
        'gaus1', 'gaus2', 'gaus4',
        'shan1.5-1.0',
        'fbsp1-1.5-1.0',
    ]

    widths = np.geomspace(1, 16, num=10)

    generate_all_wavelets(
        data_folder,
        base_out_folder,
        wavelet_list,
        widths,
        num_files=100,
        num_workers=4
    )
