# FHRDiff 🚀

## About 📚
This repository contains code for our paper "**FHRDiff: Leveraging Diffusion Models for Conditional Fetal Heart Rate Signal Generation**" presented at BIBM 2024.

## Dataset 📈
The dataset used for this project can be downloaded from the following open-source repository: [CTU-UHB CTG Database.](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/)


## Files 📂

- **`classify.py`**: Classify the original FHR data into normal/abnormal based on the PH value.
- **`spectrogram`**: Generate the PRSA curve and its spectrogram.
- **`preprocess.py`**: Preprocess the original FHR signals, including removing gaps, interpolation, detrending, and segmenting the signals to specific lengths.
- **`train.py`**: Code for training the unconditional and conditional models.
- **`inference.py`**: Code for using the trained model to synthesize FHR signals.


## Getting Started ✅

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FHRDiff.git
