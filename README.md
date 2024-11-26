# **Handwritten Digit Recognition Model Training**

This repository contains a machine learning project that builds and trains models to recognize handwritten digits using the MNIST dataset. The project includes implementations of different neural network architectures, ranging from a simple feed-forward network to an optimized convolutional neural network (CNN).

---

## **Table of Contents**

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
  - [Simple DNN](#1-simple-dnn)
  - [Basic CNN](#2-basic-cnn)
  - [Optimized CNN](#3-optimized-cnn)
- [Training and Evaluation](#training-and-evaluation)
- [How to Run the Project](#how-to-run-the-project)
- [Visualization](#visualization)
- [Future Improvements](#future-improvements)

---

## **Overview**

The goal of this project is to classify handwritten digits (0-9) from grayscale images. The project leverages the MNIST dataset, which is a standard benchmark for digit recognition tasks.

Three models are trained, evaluated, and compared to demonstrate the progression from simple architectures to more advanced and optimized ones.

---

## **Technologies Used**

- **Python 3.11**
- **TensorFlow/Keras**: For building and training the neural networks.
- **NumPy**: For data manipulation and mathematical operations.
- **Matplotlib**: For visualizing results and performance metrics.

---

## **Dataset**

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) consists of:

- 60,000 training images.
- 10,000 test images.
- Each image is a grayscale image of size 28x28 pixels, representing handwritten digits (0-9).

---

## **Models Implemented**

### **1. Simple DNN**

- A fully connected neural network with:
  - One hidden layer (128 neurons, ReLU activation).
  - Dropout for regularization (20%).
  - Softmax output layer for classification.
- **Purpose**: Establish a baseline model for comparison.

### **2. Basic CNN**

- A convolutional neural network with:
  - Two convolutional layers (32 and 64 filters, ReLU activation).
  - Max-pooling for down-sampling.
  - Dense layer with 128 neurons.
  - Dropout (50%) to prevent overfitting.
- **Purpose**: Explore spatial feature extraction with CNNs.

### **3. Optimized CNN**

- Enhances the basic CNN with:
  - Batch normalization for faster convergence.
  - Regularization (L2 kernel regularizer) to reduce overfitting.
  - Tuned dropout and learning rate.
- **Purpose**: Achieve the best performance with a more robust architecture.

---

## **Training and Evaluation**

Each model is trained for:

- **Epochs**: 10-15 (based on the model).
- **Batch Size**: 64.
- **Validation Split**: 20% of the training data.

### **Evaluation Metrics**

- **Accuracy**: Measures the proportion of correct predictions.
- **Loss**: Measures how well the model fits the data.

---

## **How to Run the Project**

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/iskburcin/2.Handwritten-Digit-Recognition-Model-Training.git
   cd 2.Handwritten-Digit-Recognition-Model-Training
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Create a Virtual Environment**:
   ```bash
   python -m venv env
   ./env/Scripts/activate
   ```

Run the Notebook: Open the model.ipynb file in a Jupyter Notebook or an IDE like VSCode and execute the cells.

---

## **Visualization**

The project includes scripts to visualize:

- **Training History**: Plots of accuracy and loss over epochs.
- **Sample Predictions**: Visual comparison of true vs. predicted labels.
- **Confusion Matrix**: To evaluate the performance of the classification.
- **Model Comparisons**: Bar charts are used to compare test accuracies and losses of all three models

---

## **Future Improvements**

- **Hyperparameter Tuning**: Experiment with different hyperparameters to improve model performance.
- **Data Augmentation**: Apply techniques like rotation, scaling, and translation to increase dataset diversity.
- **Advanced Architectures**: Implement more complex models like ResNet or GANs for better accuracy.

---

## **Contributors**

- **Author**: Burcin Işık
- **Contributors**: Feel free to fork the repository and submit pull requests.

---
