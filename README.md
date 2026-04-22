# Speech-Recognition
End-to-End Speech Recognition with DeepSpeech2 Architecture
# End-to-End Speech Recognition (DeepSpeech2)

This repository contains a modular implementation of a Deep Learning-based Automatic Speech Recognition (ASR) system using **TensorFlow** and **Keras**. The architecture is inspired by **DeepSpeech2**, utilizing a combination of 2D Convolutional layers and Bidirectional Recurrent Neural Networks.

## 🚀 Project Overview
The goal of this project is to transcribe raw audio signals into text. It uses the **LJSpeech-1.1 dataset**, which consists of 13,100 short audio clips of a single speaker.

### Technical Highlights
*   **Feature Extraction:** Converts raw `.wav` files into **Log-Mel Spectrograms** using Short-Time Fourier Transform (STFT).
*   **Acoustic Model:** A hybrid architecture featuring:
    *   **2D CNNs** for spatial and frequency-domain feature extraction.
    *   **5-stack Bidirectional GRUs** (512 units) for temporal sequence modeling.
*   **Loss Function:** Utilizes **CTC (Connectionist Temporal Classification) Loss**, enabling the model to learn speech-to-text alignment without pre-segmented timestamps.
*   **Optimization:** Implements **Batch Normalization** and **Dropout** (0.5) to ensure stable training and prevent overfitting.

## 📂 Project Structure
*   `data_loader.py`: Handles audio preprocessing, normalization, and the `tf.data` pipeline.
*   `model.py`: Defines the DeepSpeech2 architecture and the custom CTC loss function.
*   `train.py`: Main execution script for model initialization and training.
*   `requirements.txt`: List of required libraries (TensorFlow, Pandas, Jiwer, Matplotlib).

## 🛠️ Installation & Usage
1. **Clone the repo:**
   ```bash
   git clone https://github.com
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Training:**
   ```bash
   python train.py
   ```

## 🔒 File Integrity (MD5)
To ensure the code hasn't been corrupted or modified, you can verify the files using these checksums:


| File | MD5 Checksum |
| :--- | :--- |
| `data_loader.py` | `04e3a8c55117be279e64af885bd789ca` |
| `model.py` | `e459dd7e25b6e9a351b5cb798d4a71a7` |
| `train.py` | `3ebd3f3ec71c8aa5514469c0c04886a5` |

---
*Developed for the IIT Bombay MS (Research) TA Application.*
