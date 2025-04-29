# Arousal Detection from Sleep Signals (Work in Progress)
This project processes polysomnographic (PSG) recordings to classify arousal events during sleep using deep learning models.
**Note**: This project is still under development and subject to changes.
## Project Structure
- **Data Preprocessing**:
  - Extracts and segments signals and annotations from .hea and .arousal files.
  - Saves the preprocessed data (p_signal, arousals, and sleep_stages) as .npy files.
- **Wavelet Transform**:
  - Applies Continuous Wavelet Transform (CWT) to the signals for time-frequency analysis.
  - Saves the wavelet-transformed windows as separate .npy files.
- **Models**:
  - A basic 1D convolutional network for binary classification (arousal vs. no arousal).
  - A basic 2D convolutional network for binary classification.
  - DeepArousalCNN: A deeper 2D CNN with batch normalization and dropout for better performance on wavelet data.

## Requirements
- Python 3.8+
- Libraries: numpy, scipy, wfdb, pywt, torch, tqdm, sklearn, matplotlib, torchvision
