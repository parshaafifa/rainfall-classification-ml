# 🌧️ Rainfall Classification in Mymensingh using Machine Learning

This project applies various machine learning models to classify rainfall levels in the Mymensingh region of Bangladesh. It uses **MLPClassifier**, **Support Vector Machine (SVM)**, and **Random Forest**, with hyperparameter tuning via `GridSearchCV`.

## 📂 Dataset

The dataset contains rainfall and weather station data for the Mymensingh region. It includes attributes such as:

- `WID`: Weather station ID
- `RAN`: Rainfall amount (target variable)
- Various meteorological attributes

> **Note:** Due to privacy, the original dataset file is not included in this repository. You must supply your own `.csv` file.

## 🧠 Models Used

- **Multi-Layer Perceptron (MLPClassifier)**
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**

## 🛠️ Techniques

- **Data Preprocessing**:
  - Dropping missing values
  - Label encoding categorical variables
  - MinMax normalization
- **Model Training & Evaluation**:
  - `train_test_split` (75% train / 25% test)
  - `GridSearchCV` for hyperparameter tuning
  - Accuracy scoring with cross-validation

## 📊 Results

Each model is evaluated using cross-validated accuracy. The best hyperparameters are identified for each classifier using grid search.

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
