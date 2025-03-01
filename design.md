                     ┌─────────────────────────────┐
                     │        Input Video          │
                     └───────────┬─────────────────┘
                                 │
                     ┌───────────▼───────────┐
                     │   Extract Frames       │  ← extract_frames.py
                     │   (OpenCV)             │
                     └───────────┬───────────┘
                                 │
                     ┌───────────▼───────────┐
                     │   Extract Faces        │  ← extract_faces.py
                     │   (MTCNN)              │
                     └───────────┬───────────┘
                                 │
                     ┌───────────▼───────────┐
                     │   Fine-Tune Swin       │  ← train_swin.py
                     │   (Custom Dataset)     │
                     └───────────┬───────────┘
                                 │
                     ┌───────────▼───────────┐
                     │   Extract Features     │  ← extract_features.py
                     │   (Fine-Tuned Swin)    │
                     └───────────┬───────────┘
                                 │
                     ┌───────────▼───────────┐
                     │   Train LSTM Model    │  ← train_lstm.py
                     │   (Temporal Analysis) │
                     └───────────┬───────────┘
                                 │
                     ┌───────────▼───────────┐
                     │   Evaluate Model       │  ← evaluate.py
                     │   (Confusion Matrix,   │
                     │   ROC Curve, Accuracy)│
                     └───────────┬───────────┘
                                 │
                     ┌───────────▼───────────┐
                     │   Predict Deepfake     │  ← detect.py
                     │   (Swin + LSTM)        │
                     └───────────┬───────────┘
                                 │
                     ┌───────────▼───────────┐
                     │    Output: Real /     │
                     │    Deepfake           │
                     └───────────────────────┘
# Deepfake Detection System

## Overview
This project is designed to detect deepfake videos using a combination of **Swin Transformer** and **LSTM**. The model first fine-tunes Swin Transformer on a deepfake dataset, extracts meaningful features, and then trains an LSTM model to analyze sequential patterns in videos. The system can classify a given image or video frame as **Real or Deepfake**.

## Explanation of Each Component

| **Step** | **Description** | **Script Used** |
|----------|---------------|----------------|
| **1. Input Video** | A video file containing real or deepfake faces. | - |
| **2. Extract Frames** | Extracts frames from the video at a fixed interval using OpenCV. | `extract_frames.py` |
| **3. Extract Faces** | Detects and extracts faces from each frame using MTCNN. | `extract_faces.py` |
| **4. Fine-Tune Swin** | Train a pre-trained Swin Transformer on a custom deepfake dataset. | `train_swin.py` |
| **5. Extract Features** | Use the fine-tuned Swin Transformer to extract deepfake-specific features from each face. | `extract_features.py` |
| **6. Train LSTM Model** | Train an LSTM model on the extracted Swin features to analyze sequential patterns in videos. | `train_lstm.py` |
| **7. Evaluate Model** | Evaluate the model using accuracy, precision, recall, confusion matrix, and ROC curve. | `evaluate.py` |
| **8. Predict Deepfake** | Pass a new image/video into Swin + LSTM to classify it as Real or Deepfake. | `detect.py` |


