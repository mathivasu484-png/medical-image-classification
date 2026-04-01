# 🫁 Medical Image Classification — Chest X-ray

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-81.80%25-success)

A deep learning project that classifies chest X-ray images as **PNEUMONIA** or **NORMAL** using a CNN built with TensorFlow/Keras. The project went through **two training iterations** — identifying overfitting in Run 1 and resolving it in Run 2 with improved architecture and data strategy, achieving a **+13% gain in test accuracy**.

---

## 📑 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Split Strategy](#data-split-strategy)
- [Model Architecture](#model-architecture)
- [Training Iterations](#training-iterations)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Output](#output)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Author](#author)

---

## 🔍 Project Overview

Pneumonia is a serious lung infection that can be life-threatening if not detected early. This project automates detection from chest X-ray images using deep learning.

The project was built and improved across **two training runs**:

| | Run 1 | Run 2 |
|---|---|---|
| Architecture | Simple CNN | CNN with Dropout + Augmentation |
| Problem Found | Severe overfitting | Resolved — stable training |
| Test Accuracy | 68.75% | **81.80%** |

> The deliberate iteration process — diagnosing the problem, fixing it, and re-evaluating — is the core learning demonstrated in this project.

---

## 📂 Dataset

**Source:** [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

**Classes:** `NORMAL` · `PNEUMONIA`
```
chest_xray/
   ├── train/
   │     ├── NORMAL/
   │     └── PNEUMONIA/
   ├── test/
   │     ├── NORMAL/
   │     └── PNEUMONIA/
   └── val/
         ├── NORMAL/
         └── PNEUMONIA/
```

---

## 🔀 Data Split Strategy

The original dataset had only **16 validation images**, which was insufficient for honest model feedback. For Run 2, the data was re-split into a proper 70/15/15 ratio.

### NORMAL (1,583 total)

| Split | Count | % |
|-------|-------|---|
| Train | 1,108 | ~70% ✅ |
| Val | 237 | ~15% ✅ |
| Test | 238 | ~15% ✅ |

### PNEUMONIA (4,273 total)

| Split | Count | % |
|-------|-------|---|
| Train | 2,991 | ~70% ✅ |
| Val | 641 | ~15% ✅ |
| Test | 641 | ~15% ✅ |

---

## 🧠 Model Architecture

### Run 1 — Baseline CNN
```
Input (224x224x3)
    ↓
Conv2D(32) + ReLU → MaxPooling2D
    ↓
Conv2D(64) + ReLU → MaxPooling2D
    ↓
Flatten
    ↓
Dense(128, ReLU)
    ↓
Dense(1, Sigmoid) — Binary Output
```

### Run 2 — Improved CNN with ResNet50 Transfer Learning
```
Input (224x224x3)
    ↓
ResNet50 (pretrained on ImageNet, frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, ReLU)
    ↓
Dropout(0.5)           ← prevents overfitting
    ↓
Dense(1, Sigmoid) — Binary Output
```

**Key upgrades in Run 2:**
- ✅ ResNet50 transfer learning (pretrained features)
- ✅ Dropout(0.5) to prevent memorization
- ✅ Data augmentation (rotation, zoom, flip)
- ✅ EarlyStopping to halt at best weights
- ✅ Proper 70/15/15 data split

---

## 🔁 Training Iterations

### Why Run 1 Overfit

| Problem | Detail |
|---------|--------|
| Tiny validation set | Only 16 images — model never got honest feedback |
| No Dropout | Neurons memorized training data |
| No augmentation | Model never saw varied real-world X-ray angles |
| Class imbalance | PNEUMONIA images far outnumber NORMAL |
| Unstable val loss | Rose from 0.1833 → 0.4122 → 0.1794 across epochs |

### What Was Fixed in Run 2

| Problem | Solution Applied |
|---------|-----------------|
| Small val set | Re-split to 15% val (~880 images) |
| No Dropout | Added `Dropout(0.5)` after Dense layer |
| No augmentation | Rotation, zoom, shear, horizontal flip |
| Class imbalance | `class_weight` parameter in `model.fit()` |
| Weak architecture | ResNet50 transfer learning |
| Overfitting epochs | EarlyStopping with `patience=3` |

---

## 📊 Model Performance

### Run Comparison

| Run | Epochs | Train Acc | Val Acc | Test Accuracy | Test Loss |
|-----|--------|-----------|---------|---------------|-----------|
| **Run 1** | 5 | 98.87% | 81.25% | 68.75% | 2.4807 |
| **Run 2** | 11 | 79.26% | 83.14% | **81.80%** | **0.3889** |

> ✅ **+13.05% improvement in test accuracy** · ⬇️ **84.3% reduction in test loss**

---

### Run 1 — Training History

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|-------|-----------|------------|---------|----------|
| 1/5 | 90.95% | 0.2519 | 75.00% | 0.6721 |
| 2/5 | 96.41% | 0.0945 | 81.25% | 0.3045 |
| 3/5 | 98.04% | 0.0565 | 87.50% | 0.1833 |
| 4/5 | 98.54% | 0.0366 | 81.25% | 0.4122 |
| 5/5 | 98.87% | 0.0329 | 81.25% | 0.1794 |

⚠️ **~30% gap between training (98.87%) and test accuracy (68.75%) = classic overfitting.**

---

### Run 2 — Training History

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|-------|-----------|------------|---------|----------|
| 1/11 | 71.46% | 0.5992 | 73.01% | 0.5725 |
| 2/11 | 73.07% | 0.5542 | 72.67% | 0.5224 |
| 3/11 | 73.90% | 0.5304 | 75.40% | 0.4925 |
| 4/11 | 74.77% | 0.5063 | 72.89% | 0.4661 |
| 5/11 | 76.31% | 0.4792 | 80.41% | 0.4340 |
| 6/11 | 76.99% | 0.4695 | 80.64% | 0.4334 |
| 7/11 | 77.43% | 0.4567 | 81.44% | 0.4131 |
| 8/11 | 78.19% | 0.4395 | 80.87% | 0.3997 |
| 9/11 | 77.85% | 0.4334 | 83.60% | 0.4144 |
| 10/11 | 78.53% | 0.4234 | 82.57% | 0.4079 |
| 11/11 | 79.26% | 0.4281 | 83.14% | 0.4015 |

✅ **Training and validation accuracy move together — no overfitting.**

---

### Final Test Results

| Metric | Run 1 | Run 2 | Change |
|--------|-------|-------|--------|
| Test Accuracy | 68.75% | **81.80%** | ⬆️ +13.05% |
| Test Loss | 2.4807 | **0.3889** | ⬇️ −84.3% |

---

## ⚙️ Installation

**Prerequisites:** Python 3.8+
```bash
# 1. Clone the repository
git clone https://github.com/mathivasu484-png/medical-image-classification.git
cd medical-image-classification

# 2. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ How to Run
```bash
# Step 1 — Download dataset from Kaggle and extract into:
data/chest_xray/

# Step 2 — Train the model
python main.py

# Step 3 — Generate predictions on images
python predict.py
```

---

## 📤 Output

| Output | Location |
|--------|----------|
| Trained model | `models/model.keras` |
| Test accuracy | Printed in terminal |
| Results log | `results.txt` |
| Prediction images | `results/test_predictions/` |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.8+ | Core language |
| TensorFlow / Keras | Model building and training |
| ResNet50 (ImageNet) | Transfer learning backbone |
| NumPy | Data processing |
| Matplotlib | Visualization |
| Pillow (PIL) | Prediction image annotation |

---

## 🔮 Future Improvements

- [ ] Fine-tune ResNet50 layers for domain-specific features
- [ ] Add Grad-CAM to visually highlight infected lung regions
- [ ] Expand to multi-class: Bacterial vs Viral Pneumonia
- [ ] Deploy as a web app using Flask or Streamlit
- [ ] Evaluate with full metrics: Precision, Recall, F1-Score, AUC-ROC

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👩‍💻 Author

**Vasumathi**
GitHub: [mathivasu484-png](https://github.com/mathivasu484-png)
