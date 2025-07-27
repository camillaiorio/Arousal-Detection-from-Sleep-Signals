# Arousal Detection from Sleep Signals
This project processes polysomnographic (PSG) recordings to classify arousal events during sleep using deep learning models.
## Project Structure
- **Data Preprocessing**:
  - Extracts and segments signals and annotations from .hea and .arousal files using the WFDB library.
  - Downsamples signals to a target frequency and aligns them with arousal and sleep stage annotations.
  - Segments signals into fixed-size windows and saves arrays (_p_signal.npy, _arousals.npy, _sleep_stages.npy) for each subject.
- **Wavelet Transform**:
  - Applies Continuous Wavelet Transform (CWT) using a complex Morlet wavelet ('cmor1.5-1.0') to extract time-frequency features from the signals.
  - Processes fixed-length windows in parallel and saves each wavelet-transformed window as a .npy file.
- **Models**:
  - 1DCNN: A 1D convolutional neural network trained directly on the raw (downsampled) signal.
  - 2DCNN (ArousalCNN): A compact CNN using precomputed CWT images for input.
  - DeepArousalCNN: A deeper 2D CNN with batch normalization and dropout layers for improved generalization.
  - ResNet18: A transfer learning approach using a modified ResNet-18 pretrained on ImageNet, adapted to handle variable input channels and trained on resized CWT images.
- **Training**:
  - Supports training for both 1D and 2D models, with optional data augmentation (noise, flip, affine, blur).
  - Implements early stopping and logs metrics to TensorBoard.
  - Saves model checkpoints per epoch and the best-performing model separately.
  - Generates plots of loss and accuracy trends during training.
- **Prediction**:
  - Performs batched inference over entire PSG recordings using sliding windows.
  - Saves predicted probabilities and ground truth labels to .txt files for each subject, enabling evaluation and post-processing.
- **Evaluation & Analysis**:
  - Computes standard classification metrics (Accuracy, ROC AUC, F1-score, precision, recall) using sklearn.
  - Supports batch evaluation of multiple models and comparison against DeepSleep outputs.
  - Includes a script for signal-level analysis of .vec outputs (DeepSleep compatibility).

## Requirements
- Python 3.8+
- Libraries:
    - Core: numpy, scipy, wfdb, pywt, tqdm, joblib, glob, os
    - Deep Learning: torch, torchvision, matplotlib, sklearn, tensorboard
## Notes
- All signals are preprocessed into 60-second windows at 10 Hz (600 samples).
- CWT is applied on 6-second subwindows (600 samples at 100 Hz).
- The dataset comes from the 2018 PhysioNet/CinC Challenge on Sleep Arousal Detection.
- Predictions and evaluation are performed at the window level, not per-sample.