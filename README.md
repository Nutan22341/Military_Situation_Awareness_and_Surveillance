# Military Situational Awareness and Surveillance

This repository contains the implementation of machine learning and deep learning pipelines for classifying military-related audio signals such as gunfire, vehicle noise, and aircraft activity. The goal is to build a robust system for enhancing real-time audio surveillance and threat detection in military environments.

## 🧠 Project Overview

With increasing reliance on audio-based surveillance systems in defense applications, accurate and efficient sound classification has become a necessity. This project leverages **MFCC** and **Mel Spectrogram** features, combined with ML and DL models, to classify sounds into predefined military categories.

## 📁 Dataset

- The dataset is based on the **MAD (Military Audio Dataset)**.
- Audio clips were downloaded and segmented from YouTube based on metadata.
- Final dataset: 7,837 labeled audio samples after filtering.
- Classes include: Communication, Gunshot, Footsteps, Shelling, Vechile, Helicopter and Fighter.

## 📊 Preprocessing and Feature Extraction

- **MFCCs**: Extracted with statistical summaries (mean, variance, min, max). Link - https://github.com/kaen2891/military_audio_dataset?tab=readme-ov-file
- **Mel Spectrograms**: Resampled at 22,500 Hz and resized to uniform shape.
- **Label Encoding**: Labels converted to numerical and one-hot encoded form.
- **Standardization**: Applied on features using `StandardScaler`.

## 📈 Exploratory Data Analysis

- Class distribution analyzed across training and test sets.
- Spectrograms visualized to observe time-frequency patterns.
- Average duration and signal characteristics assessed per class.

## 🧪 Models Used

### Trained on MFCC Features
- **ML Models**: Random Forest, Logistic Regression, SVM, K-Nearest Neighbors, XGBoost
- **DL Models**: 
  - ANN (3 dense layers with dropout)
  - CNN (Conv1D + MaxPooling + Dense)
  - RNN (LSTM-based)

### Trained on Mel Spectrogram Features
- **Random Forest**
- **ANN**
- **CNN** (with dropout and one-hot encoding)

## 🏆 Results

### MFCC Feature Results
| Model             | Accuracy | Precision | Recall | F1-Score |
|------------------|----------|-----------|--------|----------|
| XGBoost          | 89.98%   | 90.00%    | 89.98% | 89.97%   |
| Random Forest    | 89.79%   | 89.81%    | 89.79% | 89.79%   |
| CNN              | 88.86%   | 89.00%    | 88.86% | 88.83%   |

### Mel Spectrogram Results
| Model          | Accuracy |
|----------------|----------|
| CNN            | 63.07%   |
| Random Forest  | 45.45%   |
| ANN            | 35.57%   |

> 🔍 **Observation**: MFCC features yielded better performance. Regularization and one-hot encoding helped reduce overfitting in CNNs.

## 🛠️ Technologies Used

- Python
- NumPy, Pandas
- Scikit-learn
- TensorFlow / Keras
- Librosa
- Matplotlib
  
## 🚀 How to Run

1. Clone this repository.
2. Ensure all audio files are placed under `data/MAD dataset/`.
3. Run preprocessing:
   ```bash
   python src/preprocessing.py

python src/train_mfcc_models.py
python src/train_spectrogram_models.py
