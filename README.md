# 🕵️‍♂️ Unmasking Deepfakes: How a Custom CNN Exposes AI-Generated Faces

*A research project to create a custom CNN model that identifies synthetic faces with high accuracy, designed to combat the rising threat of deepfake images.*

Dataset: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces/data

<img width="1026" alt="Screenshot 2025-04-17 at 1 14 10 PM" src="https://github.com/user-attachments/assets/9ee00ab2-70cb-4f48-864c-3305f7803353" />

---

## 🔍 Overview  
This project trains a custom convolutional neural network (CNN) to detect AI-generated faces by analyzing subtle artifacts left by GANs. As deepfake technology advances, this tool provides a scalable solution for verifying digital media authenticity.

**Key Achievements**:  
✅ **93.4% validation accuracy** on diverse real/fake face datasets  
✅ **0.983 AUC** – Exceptional at ranking "realness"  
✅ **Optimized for efficiency** – Runs on consumer hardware  

---



---

## 🛠️ Technical Approach  
### Model Architecture  
```python
Custom CNN with:
- 3 Conv2D layers (32 → 64 → 128 filters)  
- BatchNormalization + MaxPooling  
- Dropout (0.5) to prevent overfitting  
- Sigmoid output for binary classification  

