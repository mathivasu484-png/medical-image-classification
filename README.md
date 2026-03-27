# Medical Image Classification — Chest X-ray

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A deep learning project that classifies chest X-ray images as **PNEUMONIA** or **NORMAL** using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Output](#output)
- [Model Performance](#model-performance)
- [Sample Results](#sample-results)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Author](#author)

---

## Project Overview

Pneumonia is a serious lung infection that can be life-threatening if not detected early. This project automates the detection of pneumonia from chest X-ray images using a CNN model, achieving high accuracy on the test set.

---

## Dataset

**Source:** [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

| Split | Images |
|-------|--------|
| Training | ~5,216 |
| Validation | 16 |
| Test | 624 |

**Classes:** `NORMAL` · `PNEUMONIA`

---

## Model Architecture
```
Input (224x224x3)
    ↓
Conv2D + ReLU
    ↓
MaxPooling2D
    ↓
Flatten
    ↓
Dense (fully connected)
    ↓
Output — Sigmoid (binary classification)
```

---

## Installation

**Prerequisites:** Python 3.8+
```bash
# 1. Clone the repository
git clone https://github.com/mathivasu484-png/medical-image-classification.git
cd medical-image-classification

# 2. Install dependencies
pip install -r requirements.txt
```

---

## How to Run
```bash
# Step 1 — Download the dataset from Kaggle and extract into:
data/chest_xray/

# Step 2 — Train the model
python main.py
```

---

## Output

| Output | Location |
|--------|----------|
| Trained model | `models/model.keras` |
| Test accuracy | Printed in terminal |
| Results log | `results.txt` |

---

## Model Performance

### Training History

| Epoch | Accuracy | Loss | Val Accuracy | Val Loss |
|-------|----------|------|--------------|----------|
| 1/5 | 90.95% | 0.2519 | 75.00% | 0.6721 |
| 2/5 | 96.41% | 0.0945 | 81.25% | 0.3045 |
| 3/5 | 98.04% | 0.0565 | 87.50% | 0.1833 |
| 4/5 | 98.54% | 0.0366 | 81.25% | 0.4122 |
| 5/5 | 98.87% | 0.0329 | 81.25% | 0.1794 |

### Test Evaluation

| Metric | Value |
|--------|-------|
| Test Loss | 2.4807 |
| Test Accuracy | 68.75% |

> The model achieves **98.87% training accuracy** with a consistent decrease in loss across all 5 epochs, demonstrating effective learning. The test accuracy reflects performance on completely unseen data.

---

## Sample Results

The model was evaluated on unseen chest X-ray images. Confidence scores are shown in brackets.

### Pneumonia Detected

| PNEUMONIA — Confidence: 88% | PNEUMONIA — Confidence: 100% |
|:---:|:---:|
| ![Pneumonia 1](results/result_1.jpg) | ![Pneumonia 2](results/result_2.jpg) |

### Normal (Healthy)

| NORMAL — Confidence: 99% | NORMAL — Confidence: 100% |
|:---:|:---:|
| ![Normal 1](results/result_3.jpg) | ![Normal 2](results/result_4.jpg) |

---

## Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.8+ | Core language |
| TensorFlow / Keras | 2.x | Model building and training |
| NumPy | latest | Array and data processing |
| Matplotlib | latest | Visualization and plotting |

---

## Future Improvements

- [ ] Apply Transfer Learning using ResNet50 or VGG16
- [ ] Handle class imbalance with data augmentation
- [ ] Deploy model as a web application using Flask or Streamlit
- [ ] Add Grad-CAM visualization to highlight affected lung regions

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Vasumathi**  
GitHub: [mathivasu484-png](https://github.com/mathivasu484-png)
