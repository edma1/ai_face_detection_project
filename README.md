# 🕵️‍♂️ Unmasking Deepfakes: How a Custom CNN Exposes AI-Generated Faces

## Project Overview
This project develops a convolutional neural network (CNN) to distinguish real human faces from AI-generated deepfakes with 93.1% accuracy. Built using accessible hardware (Google Colab CPU, M2 MacBook Air), the solution addresses the growing threat of synthetic media in spreading misinformation.

Dataset: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces/data

<img width="1026" alt="Screenshot 2025-04-17 at 1 14 10 PM" src="https://github.com/user-attachments/assets/9ee00ab2-70cb-4f48-864c-3305f7803353" />

---

## Key Features
🛡️ 93.1% test accuracy in detecting GAN-generated faces

🧠 Custom CNN architecture optimized for limited hardware

🔍 Comparative analysis of GAN generation vs CNN detection

⚡ Efficiency-focused design for CPU-only environments

---

## Project Structure
```
ai_face_detection_project/
├── AI Face Detection Project Slides Deck/               # Contains slides deck used for presentation
├── AI_Face_Detection_Project_Technical_Notebook.ipynb   # Jupyter notebooks with all code
└── README.md                                            # This file
```
---

## 🛠️ Technical Approach  
### Dataset
- 50,000 face images (50% real, 50% AI-generated)
- Balanced classes to prevent model bias
- Preprocessed to 256×256 resolution

### Model Architecture  
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3)),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.4),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

### Hardware Constraints & Solutions
| Challenge | Solution | Impact |
|----------|-----------|--------|
| **8GB RAM limitation** | Reduced image size (256×256) | 75% memory reduction |
| **CPU-only training** | Shallow network depth | 3x faster epochs |
| **Overfitting risk** (20%→30%) | Strategic dropout (25%/40%)	 | 1.7% train-val gap |

## Key Findings
### 1. GAN Analysis:
  - Generator-Discriminator adversarial training creates increasingly realistic fakes
  - Current models can fool humans in 47% of cases (MITRE study)

### 2. Detection Patterns:
- CNN focuses on subtle artifacts in:
  - Hairline texture
  - Eye reflection consistency
  - Tooth morphology

### 3. Performance Metrics:
  - Precision: 92.8%
  - Recall: 93.5%
  - F1-score: 93.1%

## Ethical Considerations
⚠️ Dual-use risk: Detection models could potentially improve GANs
🔒 Privacy: All training data was properly sourced and anonymized
📜 Transparency: Model cannot be used for facial recognition

## Future Work
- Implement real-time video analysis
- Develop browser extension for public use
- Research transformer-based detection
- Explore ensemble methods with ResNet/EfficientNet
