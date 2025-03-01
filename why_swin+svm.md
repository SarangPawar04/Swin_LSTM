# Using Swin Transformer Features for Deepfake Detection

## **Why Use Feature Extraction Instead of Full Training?**
Normally, deep learning training involves **updating millions of parameters** in a Swin Transformer. This is slow and requires **a lot of labeled deepfake images**. Instead, we can:
- **Use Swin Transformer as a feature extractor** (without training it)
- **Train a smaller and faster model (SVM, Random Forest, or MLP) on those features**

This approach works well because Swin Transformer **already captures powerful image features**.

---

## **How Does Feature Extraction + SVM Work?**
Instead of modifying Swin Transformer and training it, we do the following:

### **Step 1: Extract Features from Each Image**
- Take an input deepfake/real image
- Pass it through a **pre-trained Swin Transformer**
- Remove the final classification layer and **extract the feature vector** for the image

### **Step 2: Train an SVM (or other ML classifier) on the Extracted Features**
- Use the extracted features as input for an SVM model
- Train the SVM to classify images as **real (0) or deepfake (1)**

### **Step 3: Use the SVM to Make Predictions**
- For new images, extract their features and let the SVM predict whether they are deepfakes.

---

## **Implementation**

### **📌 Step 1: Extract Features Using Swin Transformer**
```python
import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms

# Load a pre-trained Swin Transformer (no classification head)
model = timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=0)
model.eval()  # No training needed

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to extract features from an image
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        features = model(image_tensor)  # Get feature vector
    return features.squeeze(0).numpy()  # Convert to NumPy array

# Example usage
feature_vector = extract_features("your_image.jpg")
print("Feature Vector Shape:", feature_vector.shape)
```
✅ This gives us a **feature vector** representing the image.

---

### **📌 Step 2: Train an SVM on Extracted Features**
Once we have feature vectors for multiple images, we can train an **SVM classifier**.

```python
from sklearn.svm import SVC
import joblib

# Example dataset
image_paths = ["real1.jpg", "fake1.jpg", "real2.jpg", "fake2.jpg"]
labels = [0, 1, 0, 1]  # 0 = real, 1 = deepfake

# Extract features for all images
X = np.array([extract_features(img) for img in image_paths])
y = np.array(labels)

# Train an SVM classifier
svm_classifier = SVC(kernel="linear", probability=True)
svm_classifier.fit(X, y)

# Save the trained SVM model
joblib.dump(svm_classifier, "swin_svm_model.pkl")
print("SVM Model Trained and Saved!")
```
✅ Now we have an **SVM model trained on deepfake features**.

---

### **📌 Step 3: Predict on New Images**
```python
# Load the saved SVM model
svm_classifier = joblib.load("swin_svm_model.pkl")

# Extract features from a new image
new_feature = extract_features("test_image.jpg").reshape(1, -1)

# Predict class (0 = real, 1 = deepfake)
prediction = svm_classifier.predict(new_feature)
probability = svm_classifier.predict_proba(new_feature)

print("Predicted Class:", prediction[0])  # Output: 0 (Real) or 1 (Deepfake)
print("Probability:", probability)
```
✅ **New images are classified as real or deepfake instantly!**

---

## **Why Is This Approach Faster?**
| **Method** | **Training Time** | **Computational Cost** | **Accuracy** |
|------------|----------------|------------------|------------|
| **Fine-tuning Swin Transformer** | Several hours to days | Needs GPU power | 🔥 High |
| **Training Swin from Scratch** | Several days to weeks | Needs large dataset | ❌ Requires extensive training |
| **Extracting Features + SVM** | Only a few minutes | Works on CPU | ✅ Moderate to high |

👉 **If you need a real-time, CPU-friendly solution, using Swin as a feature extractor with an SVM is the best option!** 🚀  
