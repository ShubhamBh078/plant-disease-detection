# 🌿 Plant Disease Detection & Classification using Custom PyTorch CNN

An end-to-end Deep Learning pipeline built from scratch using PyTorch to accurately classify plant species and diagnose health conditions from leaf images.

## 🚀 Performance Summary
* **Test Accuracy:** ~96%
* **Validation Accuracy:** ~96%
* **Training Accuracy:** ~99%

---

## 📌 Project Overview

In modern agriculture, early and accurate identification of plant diseases is critical to preventing crop loss and ensuring food security. Traditional manual diagnosis relies heavily on expert knowledge, making it slow, scaling poorly, and prone to human error. 

This project addresses this challenge by establishing a complete, automated computer vision system utilizing deep learning to rapidly classify leaf images into healthy or diseased categories.

### 🔍 Problem Statement
Given a dataset containing thousands of arbitrary leaf images across multiple plant varieties (e.g., Apple, Corn, Potato, Tomato, Grape), the objective is to build an intelligent classifier capable of:
1. Recognizing the plant species.
2. Diagnosing the exact health anomaly or disease strain (e.g., Black Rot, Common Rust, Late Blight).
3. Maintaining high evaluation metrics across balanced and imbalanced target classes alike.

### 💡 Why a Custom CNN?
While many solutions rely on heavy, pre-trained architectures like VGG16 or ResNet, this project intentionally implements a custom **4-Block Convolutional Neural Network (CNN)** engineered from scratch. 

* **Efficiency:** Standard pre-trained architectures contain up to hundreds of millions of parameters, requiring immense compute resources. This custom model minimizes computational overhead while maintaining structural depth.
* **Control:** Designing the architecture from the ground up allowed for deliberate regularization integration (such as **Batch Normalization** after each convolutional step and a **40% Dropout rate** prior to dense classification layers) to explicitly curb overfitting.
* **Granular Optimization:** The custom training loop utilizes a model checkpointing mechanism that evaluates validation loss at the end of every epoch, ensuring that only the most optimal mathematical weights are saved for final deployment.

---

## 📊 Dataset
The project utilizes the popular **Plant Village Dataset** consisting of **61,486 images** mapped across **39 distinct classes** (including various healthy and diseased conditions for apples, corn, grapes, tomatoes, etc.).

* **Data Split:** 80% Training, 10% Validation, 10% Testing (managed cleanly via `SubsetRandomSampler`).
* **Preprocessing:** Resized to 255x255, Center Cropped to 224x224, and normalized into PyTorch Tensors.

---

## 🧠 Model Architecture
* **Convolutional Layers:** 4 sequential blocks utilizing `Conv2d`, `ReLU` activation, `BatchNorm2d` (for training stability), and `MaxPool2d` (for downsampling).
* **Regularization:** `Dropout(p=0.4)` implemented in the dense layers to actively mitigate overfitting.
* **Total Parameters:** ~52.5 Million
* **Loss Function:** Cross-Entropy Loss
* **Optimizer:** Adam Optimizer

---

## 📈 Training & Results
The network was trained for **10 epochs** using a batch size of 64 on a CUDA-enabled GPU environment. 
* **Model Checkpointing:** Implemented an automated weight-saving mechanism (`plant_disease_model_1.pt`) tracking minimal validation loss.
* Training plots demonstrate clean convergence without indicators of severe overfitting.

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
   git clone [https://github.com/YOUR_USERNAME/plant-disease-detection.git](https://github.com/YOUR_USERNAME/plant-disease-detection.git)
