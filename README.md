# Customer Churn Prediction Using Artificial Neural Networks (ANN)

## Project Overview

Customer retention is a key challenge for businesses, as losing customers can significantly impact revenue and long-term growth. This project aims to develop a deep learning model using **Artificial Neural Networks (ANN)** to predict customer churn based on various demographic and transactional features. By identifying at-risk customers early, businesses can implement targeted retention strategies to improve customer loyalty and reduce churn rates.

## Objective

The goal of this project is to:

- Apply deep learning techniques to solve a real-world **customer churn prediction** problem.
- Enhance machine learning and model optimization skills through **hyperparameter tuning and deployment**.
- Provide a user-friendly interface for businesses to predict customer churn using a **Streamlit web application**.
- Optimize model performance using **feature engineering, data preprocessing, and ANN architecture tuning**.

This project serves as a practical application of my deep learning knowledge while also addressing a critical business problem—understanding customer attrition and improving retention strategies.

---

## Model Performance

After training and optimizing the ANN model, I achieved an **accuracy of 86% on the test set**, demonstrating the model's capability to effectively predict customer churn.

---

## Project Structure

Your repository comprises the following files and directories:

ANN-Classification-Churn/
├── Churn_Modelling.csv
├── LICENSE
├── README.md
├── app.py
├── experiments.ipynb
├── hyperparametertunning.ipynb
├── label_encoder_gender.pkl
├── model.h5
├── onehot_encoder_geo.pkl
├── predictions.ipynb
├── requirements.txt
└── scaler.pickle



### Key Components

- **Data Files**:
  - `Churn_Modelling.csv`: Dataset used for training and evaluation.
  - `label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`, `scaler.pickle`: Pre-trained encoders and scaler for data preprocessing.

- **Notebooks**:
  - `experiments.ipynb`: Contains exploratory data analysis and initial model development.
  - `hyperparametertunning.ipynb`: Notebook dedicated to hyperparameter tuning of the ANN model.
  - `predictions.ipynb`: Demonstrates model predictions on sample data.

- **Scripts**:
  - `app.py`: Streamlit application for deploying the churn prediction model.

- **Model Artifacts**:
  - `model.h5`: Trained ANN model saved in HDF5 format.

- **Configuration**:
  - `requirements.txt`: Lists the dependencies required to run the project.

---

## Code Breakdown

### `app.py`

This script sets up a **Streamlit web application** to serve the churn prediction model.

```python
import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

st.title("Customer Churn Prediction")

credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
age = st.number_input("Age", min_value=18, max_value=100, value=30)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(np.array([[credit_score, age]]))
    result = "Churn" if prediction > 0.5 else "No Churn"
    st.write(f'Prediction: {result}')

hyperparametertunning.ipynb
This notebook focuses on optimizing the ANN model’s hyperparameters. It covers:

Hyperparameter Selection: Identifying key hyperparameters such as the number of neurons, learning rate, and batch size.
Optimization Process: Utilizing techniques like GridSearchCV or RandomizedSearchCV to find the optimal hyperparameter combinations.
Results Analysis: Comparing different models and selecting the best-performing configuration.
Key Code Snippet
python
Copy
Edit
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

def build_model(optimizer='adam', activation='relu'):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation=activation, input_shape=(11,)),
        tf.keras.layers.Dense(8, activation=activation),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=build_model, verbose=0)

param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'activation': ['relu', 'tanh'],
    'batch_size': [16, 32, 64],
    'epochs': [10, 20, 30]
}

grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=5, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Accuracy: {grid_search.best_score_}')
Uses RandomizedSearchCV to tune optimizer, activation function, batch size, and epochs.
Defines a function to build the ANN model dynamically.
Performs cross-validation to select the best-performing hyperparameters.
For more details, please go through the hyperparametertunning.ipynb file.

Conclusion
This project demonstrates the power of deep learning in predicting customer churn. By leveraging an ANN-based classification model, we can identify at-risk customers and help businesses implement proactive retention strategies. The project includes hyperparameter tuning, exploratory data analysis (EDA), and a Streamlit web application for model deployment, making it a comprehensive approach to churn prediction.

If you have any questions or suggestions for improvement, feel free to reach out. 🚀



