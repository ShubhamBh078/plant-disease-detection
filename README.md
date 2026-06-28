An end-to-end Deep Learning pipeline built using PyTorch to accurately classify plant species and diagnose health conditions/diseases from leaf images. 

## 🚀 Performance Summary
* **Test Accuracy:** ~96%
* **Validation Accuracy:** ~96%
* **Training Accuracy:** ~99%

---

## 📊 Dataset
The project utilizes the popular **Plant Village Dataset** consisting of **61,486 images** mapped across **39 distinct classes** (including various healthy and diseased conditions for apples, corn, grapes, tomatoes, etc.).

* **Data Split:** 80% Training, 10% Validation, 10% Testing (managed via `SubsetRandomSampler`).
* **Preprocessing:** Resized to $255 \times 255$, Center Cropped to $224 \times 224$, and normalized into PyTorch Tensors.

---

## 🧠 Model Architecture
Instead of utilizing heavy transfer learning models, a lightweight, highly optimized **custom 4-Block Convolutional Neural Network (CNN)** was engineered from scratch.

* **Convolutional Layers:** 4 sequential blocks utilizing `Conv2d`, `ReLU` activation, `BatchNorm2d` (for training stability), and `MaxPool2d` (for downsampling).
* **Regularization:** `Dropout(p=0.4)` implemented in the dense layers to actively mitigate overfitting.
* **Total Parameters:** ~52.5 Million
* **Loss Function:** Cross-Entropy Loss
* **Optimizer:** Adam Optimizer

---

## 📈 Training & Results
The network was trained for **10 epochs** using a batch size of 64 on a CUDA-enabled GPU environment. 
* **Model Checkpointing:** Implemented automated weight saving (`plant_disease_model_1.pt`) tracking minimal validation loss.
* The training plots demonstrate a clean convergence without indicators of severe overfitting.

---

## 🛠️ Tech Stack & Dependencies
* **Language:** Python 3
* **Deep Learning Framework:** PyTorch & Torchvision
* **Data Processing & Visualization:** NumPy, Pandas, Matplotlib
* **Environment:** Google Colab / Jupyter Notebooks

---

## 💻 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/ShubhamBh078/plant-disease-detection.git](https://github.com/ShubhamBh078/plant-disease-detection.git)
