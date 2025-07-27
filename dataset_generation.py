import os
import numpy as np
import scipy
import wfdb
from tqdm import tqdm
import multiprocessing
import glob
train_folder = r'C:\Users\Utente\Downloads\training'
test_folder =  r'C:\Users\Utente\Downloads\test'

arousal_types = ['arousal_bruxism', 'arousal_noise', 'arousal_plm',
                 'arousal_rera', 'arousal_snore', 'arousal_spontaneous',
                 'resp_centralapnea', 'resp_cheynestokesbreath', 'resp_hypopnea',
                 'resp_hypoventilation', 'resp_mixedapnea',
                 'resp_obstructiveapnea', 'resp_partialobstructive']
sleep_stage_types = ['W', 'N1', 'N2', 'N3', 'R']

# Create dictionaries to map event names to numerical indices
arousal_types_d = {k: v for v, k in enumerate(arousal_types)}
sleep_stage_types_d = {k: v for v, k in enumerate(sleep_stage_types)}
# a list of indices to all other stages for each sleep stage
other_sleep_stage_types_d = {k: [i for i in range(len(sleep_stage_types)) if i != v] for v, k in
                             enumerate(sleep_stage_types)}

#Extracts signal and labels from a WFDB record and annotation.
def extract_signal(record, annotation, window=60 * 60, freq=10):
    # window: window size in seconds
    # freq: sampling frequency in seconds
    fs = record.fs
    sig_name = record.sig_name
    p_signal = record.p_signal

    # Initialize sleep stage labels and arousal label matrix
    sleep_stages = np.zeros((p_signal.shape[0], len(sleep_stage_types))).astype(int)
    sleep_stages[:, 0] = 1
    arousals = np.zeros((p_signal.shape[0], len(arousal_types))).astype(int)
    for sample, event_type in zip(annotation.sample, annotation.aux_note):
        if event_type in sleep_stage_types:
            # If annotation is a sleep stage:set the corresponding stage to 1
            # from the given sample onwards and all other stages to 0
            sleep_stages[sample:, sleep_stage_types_d[event_type]] = 1
            sleep_stages[sample:, other_sleep_stage_types_d[event_type]] = 0
        else:
            # For arousal and respiratory events:
            # Check whether it's a start ('(' prefix) or end (')' suffix) event
            begin = event_type[0] == '('
            event_type = event_type[1:] if begin else event_type[:-1]
            # Set or unset the arousal label starting from this sample
            arousals[sample:, arousal_types_d[event_type]] = 1 if begin else 0

    # Downsample the signal to target frequency
    p_signal = p_signal[::fs // freq, :]

    arousals = arousals.reshape(-1, fs // freq, len(arousal_types))
    arousals = arousals.max(1)

    sleep_stages = sleep_stages.reshape(-1, fs // freq, len(sleep_stage_types))
    sleep_stages = sleep_stages.max(1)

    # Remove remaining samples that don't fit perfectly into a window
    remaining = p_signal.shape[0] % (window * freq)

    if remaining != 0:
        p_signal = p_signal[:-remaining, :]
        arousals = arousals[:-remaining, :]
        sleep_stages = sleep_stages[:-remaining, :]

    # Reshape all arrays into chunks of shape (num_windows, window_size, num_channels)
    p_signal = p_signal.reshape(-1, window * freq, len(sig_name))
    arousals = arousals.reshape(-1, window * freq, len(arousal_types))
    sleep_stages = sleep_stages.reshape(-1, window * freq, len(sleep_stage_types))
    return p_signal, arousals, sleep_stages


def worker_function(record_name): #Extracts signal, arousals, and sleep stage masks
    record = wfdb.rdrecord(record_name)
    annotation = wfdb.rdann(record_name, "arousal")
    p_signal, arousals, sleep_stages = extract_signal(record, annotation, 60*60, freq=200)
    savename = record_name.split("\\")[-1]
    np.save(f'D:\\TESI\\records\\{savename}_p_signal.npy', p_signal)
    np.save(f'D:\\TESI\\records\\{savename}_arousals.npy', arousals)
    np.save(f'D:\\TESI\\records\\{savename}_sleep_stages.npy', sleep_stages)
    return record_name

def parallel_processing_with_progress(data, processes = multiprocessing.cpu_count()):
    """Parallel execution using Pool.imap_unordered with tqdm for progress tracking."""
    results = []
    with multiprocessing.Pool(processes=processes) as pool:
        for result in tqdm(pool.imap_unordered(worker_function, data), total=len(data), desc="Processing"):
            results.append(result)
    return results


if __name__ == '__main__':
    records = [r.replace('.hea', '') for r in glob.glob('C:\\Users\\Utente\\Downloads\\training\\tr*\*.hea')]
    # Process first 100 records in parallel using 8 processes
    parallel_processing_with_progress(records[:100], 8)