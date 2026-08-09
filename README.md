#  Breast Cancer Classification Using Deep Learning

##  Project Overview

This project implements a **Deep Learning classification model** to predict whether a breast tumor is **Malignant** or **Benign** using the Breast Cancer dataset available through Scikit-learn.

The project follows an end-to-end Deep Learning workflow, starting from data loading and exploratory data analysis (EDA) to preprocessing, neural network development, model optimization, activation-function comparison, Dropout, and regularization techniques.

The primary objective is to understand how different Deep Learning techniques affect model performance and generalization.

---

##  Objectives

* Load and understand the Breast Cancer dataset.
* Perform Exploratory Data Analysis (EDA).
* Analyze the target-class distribution.
* Preprocess and standardize the input features.
* Build a Multilayer Perceptron (MLP) using TensorFlow/Keras.
* Compare different activation functions.
* Analyze model training and validation performance.
* Implement Dropout to reduce overfitting.
* Implement L1, L2, and L1-L2 regularization.
* Evaluate Deep Learning models on unseen test data.
* Compare different model configurations.

---

##  Dataset

The project uses the **Breast Cancer Wisconsin Diagnostic dataset** provided by `sklearn.datasets`.

### Dataset Information

| Property            |                 Value |
| ------------------- | --------------------: |
| Total Records       |                   569 |
| Input Features      |                    30 |
| Target Classes      |                     2 |
| Feature Type        |             Numerical |
| Missing Values      |                  None |
| Classification Type | Binary Classification |

The target variable contains two classes:

* `0` → Malignant
* `1` → Benign

The dataset contains 357 benign samples and 212 malignant samples.

---

##  Technologies & Libraries

### Programming Language

* Python

### Libraries

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow
* Keras

### Environment

* Jupyter Notebook
* Google Colab

---

##  Project Workflow

```text
Dataset Loading
       ↓
Exploratory Data Analysis
       ↓
Data Understanding
       ↓
Class Distribution Analysis
       ↓
Train-Test Split
       ↓
Feature Standardization
       ↓
Neural Network / MLP
       ↓
Model Training
       ↓
Activation Function Comparison
       ↓
Dropout
       ↓
Regularization
       ↓
Model Evaluation
       ↓
Performance Comparison
```

---

#  Task 1: Data Loading, EDA & Preprocessing

The first stage focuses on understanding and preparing the dataset.

### Key Steps

* Import required libraries.
* Load the Breast Cancer dataset.
* Separate independent features `X` and target `y`.
* Analyze dataset dimensions.
* Generate statistical summaries.
* Check for missing values.
* Analyze target-class distribution.
* Standardize numerical features.

### Dataset Observation

* **569 rows**
* **30 columns**
* All input features are numerical.
* No missing values are present.
* The dataset is suitable for Deep Learning after preprocessing.

---
# Deep Learning Model

A **Multilayer Perceptron (MLP)** is implemented using TensorFlow/Keras.

The neural network learns relationships between the 30 input features and the binary target variable.

A typical architecture used in the project consists of:

```text
Input Layer
    ↓
Dense Layer — 128 Neurons
    ↓
Activation Function
    ↓
Dense Layer — 64 Neurons
    ↓
Activation Function
    ↓
Output Layer — 1 Neuron
    ↓
Sigmoid
```

The output layer uses the **Sigmoid activation function**, making the model suitable for binary classification.

---

# Activation Function Comparison

Different activation functions are tested to understand their effect on model performance.

The project compares:

* ReLU
* Tanh
* Sigmoid

### Observed Validation Performance

| Activation Function | Best Validation Accuracy |
| ------------------- | -----------------------: |
| ReLU                |                     100% |
| Tanh                |                   97.83% |
| Sigmoid             |                   97.83% |

In the performed experiment, **ReLU achieved the highest validation accuracy** among the tested activation functions.

> Note: Validation accuracy depends on the particular train/validation split, random state, and training configuration.

---

# Dropout Regularization

Dropout is introduced to help reduce **overfitting** and improve the model's ability to generalize to unseen data.

The notebook evaluates the model on the test dataset and compares training and validation performance.

### Why Dropout?

Dropout randomly disables a proportion of neurons during training.

```text
Without Dropout
      ↓
Higher Risk of Overfitting

With Dropout
      ↓
Reduced Dependence on Individual Neurons
      ↓
Better Generalization
```

The notebook specifically analyzes the gap between training and validation accuracy to understand the effect of Dropout.

---

# Regularization

The project also explores different regularization techniques.

### Techniques Implemented

* L1 Regularization
* L2 Regularization
* L1-L2 Regularization

Regularization adds a penalty to the model's loss function to discourage unnecessarily complex models.

### L2 Regularization

The project implements L2 regularization on the Dense layers using:

```python
kernel_regularizer=regularizers.l2(0.001)
```

The tested architecture includes:

```text
Input
  ↓
Dense — 128 neurons + ReLU + L2
  ↓
Dense — 64 neurons + ReLU + L2
  ↓
Output — 1 neuron + Sigmoid
```

---

# Model Evaluation
The Deep Learning models are evaluated using training, validation, and test performance.

Important evaluation factors include:

* Accuracy
* Training Loss
* Validation Loss
* Training Accuracy
* Validation Accuracy
* Test Accuracy
* Overfitting / Underfitting behavior

Training and validation curves are used to understand how the model behaves during training.

---

#  Key Findings

### Dataset

* The dataset contains 569 observations and 30 numerical input features.
* There are no missing values.
* The target variable represents malignant and benign tumor classes.

### Deep Learning

* A neural-network-based MLP can effectively perform binary classification on the dataset.
* Feature standardization is important before training the neural network.
* Different activation functions can produce different validation performance.

### Activation Functions

* ReLU achieved the highest recorded validation accuracy in the notebook's comparison.
* Tanh and Sigmoid produced slightly lower validation accuracy in the same experiment.

### Regularization

* Dropout helps control overfitting by randomly deactivating neurons during training.
* L1/L2-based regularization provides another mechanism for controlling model complexity.
* Comparing training and validation performance helps identify generalization issues.

---

#  Deep Learning Concepts Covered

This project provides practical exposure to:

* Artificial Neural Networks
* Multilayer Perceptron (MLP)
* Dense Layers
* Activation Functions
* ReLU
* Tanh
* Sigmoid
* Forward Propagation
* Backpropagation
* Loss Functions
* Optimizers
* Model Training
* Validation
* Binary Classification
* Feature Standardization
* Dropout
* L1 Regularization
* L2 Regularization
* L1-L2 Regularization
* Overfitting
* Underfitting
* Model Generalization

---

#  Repository Structure

```text
PR1_DL/
│
├── PR1_DL.ipynb
│
└── README.md
```

---


#  Example Import Setup

```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
```

---

#  Future Improvements

The project can be further enhanced by:

* Hyperparameter tuning.
* Implementing Early Stopping.
* Using Batch Normalization.
* Performing K-Fold Cross Validation.
* Comparing the ANN with CNN-based approaches where appropriate.
* Testing additional optimizers.
* Performing systematic learning-rate tuning.
* Creating a confusion matrix and classification report.
* Saving the best trained model.
* Deploying the model using Flask, FastAPI, or Streamlit.
* Creating an interactive prediction application.

---

#  Learning Outcomes

After completing this project, the following concepts are practically demonstrated:

1. How to prepare tabular data for Deep Learning.
2. How to build an MLP using TensorFlow/Keras.
3. How activation functions affect neural-network performance.
4. How Dropout can help reduce overfitting.
5. How L1 and L2 regularization control model complexity.
6. How to monitor training and validation performance.
7. How to evaluate a binary classification model.
8. How to compare multiple Deep Learning configurations.

---

#  Author

** Smit Patel **

Aspiring Data Scientist
Python | SQL | Machine Learning | Deep Learning | Power BI

---

#  Project Repository

**GitHub:**
https://github.com/smitp2705/DL_PR1

**Notebook:**
https://github.com/smitp2705/DL_PR1/blob/main/PR_1_DL.ipynb

---

#  Conclusion

This project demonstrates an end-to-end **Deep Learning classification workflow** using the Breast Cancer dataset.

It goes beyond simply building a neural network by experimentally analyzing **activation functions, Dropout, and regularization techniques**. The project provides practical understanding of how neural-network architecture and regularization strategies influence model performance and generalization.

Overall, the project serves as a practical implementation of fundamental **Deep Learning and Neural Network concepts using TensorFlow and Keras**.
