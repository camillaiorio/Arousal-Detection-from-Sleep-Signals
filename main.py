train_folder = r'C:\Users\Utente\Downloads\training'
test_folder =  r'C:\Users\Utente\Downloads\test'

import os
print(os.listdir(train_folder))

patient = "tr12-0244"
#patient = "tr07-0293"
signals_file = patient+'.mat'
arousal_file = patient


import numpy as np
import scipy

print(arousal_file)

import wfdb
ann = wfdb.rdann(os.path.join(train_folder, patient, arousal_file), 'arousal', sampto=300000)
#data = scipy.io.loadmat(os.path.join(train_folder, patient, arousal_file))
print(ann)




exit(0)

data = scipy.io.loadmat(os.path.join(train_folder, patient, signals_file))

print(data['val'])

import matplotlib.pyplot as plt

data['val'] = data['val'][:,::20]
frequency = 10
total_seconds = data['val'].shape[1]/frequency
print(total_seconds)
time = np.linspace(0, total_seconds, data['val'].shape[1])


import pywt
signal = data['val'][-3, 0:15*60*frequency]
time = time[:15*60*frequency]


# perform CWT
wavelet = "cmor1.5-1.0"
# logarithmic scale for scales, as suggested by Torrence & Compo:
widths = np.geomspace(1, 1024, num=100)
sampling_period = np.diff(time).mean()
cwtmatr, freqs = pywt.cwt(signal, widths, wavelet, sampling_period=sampling_period)
# absolute take absolute value of complex result
cwtmatr = np.abs(cwtmatr[:-1, :-1])

# plot result using matplotlib's pcolormesh (image with annoted axes)
fig, axs = plt.subplots(2, 1)
pcm = axs[0].pcolormesh(time, freqs, cwtmatr)
axs[0].set_yscale("log")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Frequency (Hz)")
axs[0].set_title("Continuous Wavelet Transform (Scaleogram)")
fig.colorbar(pcm, ax=axs[0])

# plot fourier transform for comparison
from numpy.fft import rfft, rfftfreq

yf = rfft(signal)
xf = rfftfreq(len(signal), sampling_period)
plt.semilogx(xf, np.abs(yf))
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_title("Fourier Transform")
plt.tight_layout()
plt.show()