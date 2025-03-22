# **Deepfake Detection Model - Training & Inference Guide**

This repository contains the implementation of a **Deepfake Detection Model** using **Swin Transformer** for feature extraction and **LSTM** for classification. The project includes automated pipelines for both training and testing phases.

## **Prerequisites**

Ensure you have Python **3.8+** installed. Then, install the required dependencies:

```bash
pip install torch torchvision timm facenet-pytorch opencv-python pillow numpy scikit-learn
```

### **Check GPU Availability (Optional but Recommended)**

```python
import torch
print(torch.cuda.is_available())  # Should return True if GPU is available
```

---

## **Project Structure**

```
Deepfake-Detection/
├── data/
│   ├── videos/                  # Raw video files
│   │   ├── real/               # Real videos for training
│   │   ├── fake/               # Fake videos for training
│   │   └── test/               # Test videos for evaluation
│   ├── extracted_frames/        # Frames extracted from videos
│   │   ├── real/
│   │   ├── fake/
│   │   └── test/
│   ├── extracted_faces/         # Faces extracted from frames
│   │   ├── real/
│   │   ├── fake/
│   │   └── test/
├── dataset/
│   ├── extracted_features/      # Features extracted using Swin Transformer
│   │   ├── real.pt             # Training features for real faces
│   │   └── fake.pt             # Training features for fake faces
│   └── test_features/          # Features for evaluation
│       └── test_features.pt    # Combined test features
├── models/
│   ├── swin_model_custom.pth   # Fine-tuned Swin Transformer model
│   └── lstm_model_best.pth     # Best trained LSTM model
├── results/                     # Directory for storing results
│   ├── detection_results_*.json # Detection results
│   └── evaluation_results_*.json # Evaluation results
├── train_pipeline.py           # Automated training pipeline
├── test_pipeline.py            # Automated testing pipeline
├── extract_frames.py           # Extract frames from video
├── extract_faces.py            # Detect and extract faces
├── swin_feature_extraction.py  # Feature extraction using Swin Transformer
├── train_swin.py              # Fine-tune Swin Transformer model
├── lstm_model.py              # Core LSTM model definition
├── train_lstm.py              # LSTM model training script
├── detect.py                  # Deepfake detection script
├── evaluation.py              # Model evaluation script
└── README.md                  # Project documentation
```

---

## **Quick Start Guide**

The project provides two main automation scripts:

1. **Training Pipeline** (`train_pipeline.py`):
   - Handles the entire training process
   - From frame extraction to LSTM model training
   - Automatically creates necessary directories

2. **Testing Pipeline** (`test_pipeline.py`):
   - Handles the entire testing and evaluation process
   - From test video processing to model evaluation
   - Generates detailed results and metrics

### **Step 1: Prepare Your Dataset**

1. Place your training videos:
   - Real videos in `data/videos/real/`
   - Fake videos in `data/videos/fake/`

2. Place your test videos:
   - Test videos in `data/videos/test/`

### **Step 2: Run the Training Pipeline**

Simply run the training pipeline with a single command:
```bash
python train_pipeline.py
```

This single command will automatically handle the entire training process:
1. Create all necessary directories
2. Extract frames from training videos
3. Extract faces from frames
4. Fine-tune Swin Transformer model
   - Starts from pretrained weights
   - Adapts to deepfake detection task
   - Saves as `swin_model_custom.pth`
5. Extract features using fine-tuned Swin Transformer
   - Features are split into 8 chunks of 128 dimensions each
   - Training features are saved separately as `real.pt` and `fake.pt`
6. Train the LSTM model
   - Uses attention mechanism for better feature focus
   - Saves best model as `lstm_model_best.pth`

### **Step 3: Run the Testing Pipeline**

Execute the testing pipeline:
```bash
python test_pipeline.py
```

This will automatically:
1. Create necessary test directories
2. Extract frames from test videos
3. Extract faces from test frames
4. Extract features using Swin Transformer
   - Test features are processed in the same 8x128 chunk format
   - Saved as a single `test_features.pt` file
5. Run deepfake detection on test faces
6. Evaluate model performance

Results will be saved in the `results/` directory:
- Detection results: `results/detection_results_*.json`
- Evaluation results: `results/evaluation_results_*.json`

---

## **Technical Details**

### **Feature Extraction**

The Swin Transformer extracts 1024-dimensional features from each face image, which are then:
- Split into 8 chunks of 128 features each
- Processed sequentially by the LSTM
- This chunking allows the model to focus on different aspects of the face

### **LSTM Architecture**

The DeepfakeLSTM model includes:
- Input size: 128 (chunk size)
- Hidden size: 256
- Number of layers: 2
- Bidirectional processing
- Attention mechanism for better feature focus
- Dropout for regularization

### **Training Process**

- Early stopping to prevent overfitting
- Learning rate scheduling
- Best model checkpoint saving
- Separate validation set for model selection

### **Testing Process**

- Unified feature processing for test data
- Confidence scores for each prediction
- Comprehensive evaluation metrics
- Detailed results logging

---

## **Results Format**

### **Training History** (`results/training_history.png`)
The training process generates a plot showing the model's learning progress over time:
- Top subplot: Training and validation loss curves
- Bottom subplot: Training and validation accuracy curves
- Helps visualize model convergence and potential overfitting
=======

2. Place your test videos:
   - Test videos in `data/videos/test/`

### **Step 2: Run the Training Pipeline**

Simply run the training pipeline with a single command:
```bash
python train_pipeline.py
```

This single command will automatically handle the entire training process:
1. Create all necessary directories
2. Extract frames from training videos
3. Extract faces from frames
4. Fine-tune Swin Transformer model
   - Starts from pretrained weights
   - Adapts to deepfake detection task
   - Saves as `swin_model_custom.pth`
5. Extract features using fine-tuned Swin Transformer
   - Features are split into 8 chunks of 128 dimensions each
   - Training features are saved separately as `real.pt` and `fake.pt`
6. Train the LSTM model
   - Uses attention mechanism for better feature focus
   - Saves best model as `lstm_model_best.pth`

### **Step 3: Run the Testing Pipeline**

Execute the testing pipeline:
```bash
python test_pipeline.py
```

This will automatically:
1. Create necessary test directories
2. Extract frames from test videos
3. Extract faces from test frames
4. Extract features using Swin Transformer
   - Test features are processed in the same 8x128 chunk format
   - Saved as a single `test_features.pt` file
5. Run deepfake detection on test faces
6. Evaluate model performance

Results will be saved in the `results/` directory:
- Detection results: `results/detection_results_*.json`
- Evaluation results: `results/evaluation_results_*.json`

---

## **Technical Details**

### **Feature Extraction**

The Swin Transformer extracts 1024-dimensional features from each face image, which are then:
- Split into 8 chunks of 128 features each
- Processed sequentially by the LSTM
- This chunking allows the model to focus on different aspects of the face

### **LSTM Architecture**

The DeepfakeLSTM model includes:
- Input size: 128 (chunk size)
- Hidden size: 256
- Number of layers: 2
- Bidirectional processing
- Attention mechanism for better feature focus
- Dropout for regularization

### **Training Process**

- Early stopping to prevent overfitting
- Learning rate scheduling
- Best model checkpoint saving
- Separate validation set for model selection

### **Testing Process**

- Unified feature processing for test data
- Confidence scores for each prediction
- Comprehensive evaluation metrics
- Detailed results logging

---

## **Results Format**

### **Detection Results** (`detection_results_*.json`)
```json
{
    "timestamp": "2024-03-21 15:30:45",
    "total_images": 100,
    "real_count": 60,
    "fake_count": 40,
    "real_percentage": 60.0,
    "fake_percentage": 40.0,
    "details": [
        {
            "image": "face_001.jpg",
            "prediction": "Real",
            "confidence": 0.9234
        }
    ]
}
```

### **Evaluation Results** (`evaluation_results_*.json`)
```json
{
    "timestamp": "2024-03-21 15:35:12",
    "model_path": "models/lstm_model_best.pth",
    "predictions": {
        "total_samples": 100,
        "predicted_real": 60,
        "predicted_fake": 40
    }
}
```

---

## **Troubleshooting**

1. **Missing Dependencies**
   - Ensure all required packages are installed
   - Run `pip install -r requirements.txt` if available

2. **GPU Issues**
   - Check CUDA installation if using GPU
   - Verify PyTorch CUDA version matches your system

3. **Directory Structure**
   - Ensure all required directories exist
   - Check file permissions for read/write access

4. **Model Loading**
   - Verify `lstm_model_best.pth` exists in `models/` directory
   - Check model file permissions

5. **Feature Extraction**
   - Ensure Swin Transformer can process your images
   - Verify face detection is working correctly
   - Check feature dimensions match expected format (8x128)

6. **Common Errors**
   - FileNotFoundError: Create necessary directories
   - CUDA out of memory: Reduce batch size
   - Model dimension mismatch: Ensure feature chunking is correct

---

## **License**
This project is open-source and free to use.


### **Detection Results** (`detection_results_*.json`)
```