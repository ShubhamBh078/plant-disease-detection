# Plant Disease Detection with PyTorch

This repository contains a custom Convolutional Neural Network (CNN) built with PyTorch to classify plant diseases from leaf images. 

I built this project to get hands-on experience with deep learning pipelines—from loading and transforming raw image data to writing a custom training loop and evaluating the results. The model can identify 38 different classes of plant species and diseases (like Apple Black Rot or Tomato Late Blight).

## Performance
* **Test Accuracy:** ~88%
* **Validation Accuracy:** ~86%
* **Training Accuracy:** ~86%

## The Dataset
The model was trained on the **PlantVillage dataset**, which contains 61,486 images. 
* I split the data into 80% training, 10% validation, and 10% testing.
* All images are resized to 255x255, center-cropped to 224x224, and converted to tensors.

## Model Architecture
Instead of just fine-tuning a massive pre-trained model like ResNet or VGG16, I wanted to see how well a lighter, custom-built model could perform. 

I built a 4-block CNN from scratch. The architecture looks like this:
* **4 Convolutional Blocks:** Each block uses `Conv2d`, followed by `ReLU` activation, `BatchNorm2d` (to keep training stable and fast), and `MaxPool2d` for downsampling.
* **Classifier:** A fully connected layer at the end.
* **Regularization:** I added a 40% Dropout (`p=0.4`) before the final output layer to prevent the model from just memorizing the training data.

The final model has about 52.5 million parameters. It was trained using **Cross-Entropy Loss** and the **Adam Optimizer**.

## Results
I trained the network for 10 epochs (batch size = 64). I also wrote a checkpointing script in the training loop that tracks the validation loss at the end of each epoch and only saves the model weights (`plant_disease_model_1.pt`) if the loss improves. 

The validation and test accuracies matched up closely (both around 96%), meaning the model generalizes well to new leaves it hasn't seen before.

<!-- <p align="center">
  <img src="assets/metrics.png" width="500" alt="Training and Validation Metrics">
</p> -->

*Above: The training and validation loss/accuracy curves over 10 epochs.*

## How to Run It

If you want to run this code or train the model yourself:

 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/ShubhamBh078/plant-disease-detection.git
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Open plant_disease_detection.ipynb in Google Colab or Jupyter and run all cells.   
