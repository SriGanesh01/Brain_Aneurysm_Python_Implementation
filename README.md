﻿# Brain_Aneurysm_Python_Implementation
# Aneurysm Rupture Detection Project

## Overview
This project presents an end-to-end machine learning pipeline to detect aneurysm rupture status based on patient data and imaging features. It integrates data loading, cleansing, exploratory data analysis (EDA), model training, hyperparameter tuning, evaluation, and interpretability.

## Key Steps:
- **Data Loading & Cleaning:** Load the dataset from a CSV, drop rows with missing values.
- **Exploratory Data Analysis (EDA):** Analyze feature distributions, correlations, and apply PCA to assess separability.
- **Modeling:** Train and compare two models (Random Forest and XGBoost). Conduct hyperparameter tuning with GridSearchCV using a dynamically determined cross-validation scheme.
- **Evaluation:** Evaluate model performance using confusion matrices, ROC curves, and calibration plots. 
- **Interpretability:** Use SHAP to determine the contribution of each feature in the predictions.
- **Deployment:** The best model and scaler are saved for future predictions.

Advanced Aneurysm Rupture Detection Pipeline

This md provides an in‐depth analysis of the aneurysm dataset contained in merged.csv.
It performs:
  • Comprehensive EDA, including:
      - Data profiling, distribution plots,
      - Correlation clustering (with filtering of non-finite values),
      - PCA visualization.
  • Data cleaning and feature selection.
  • Model comparison using Random Forest and XGBoost.
  • Hyperparameter tuning with GridSearchCV using a dynamically adjusted StratifiedKFold.
  • Advanced evaluation: confusion matrix, ROC curve, calibration curve, and cross-validation.
  • Model interpretability via SHAP (with shape-transposition fix).
  • Saving of the best model and scaler for future predictions.

Instructions:
  1. Place this script and merged.csv in the same folder.
  2. Install required packages:
       pip install pandas numpy matplotlib seaborn scikit-learn joblib xgboost shap
  3. Run the script.

![image](https://github.com/user-attachments/assets/11e653a0-f8c7-4803-8a8a-b4c06e7b2377)

![image](https://github.com/user-attachments/assets/8db926b2-9eac-445d-b082-e6db7230b5a0)

![image](https://github.com/user-attachments/assets/186d3540-8df2-4def-b8a3-4cde40ded7fc)

![image](https://github.com/user-attachments/assets/5c1f3d0d-837f-42a0-824e-2a4492bcbdd4)

![image](https://github.com/user-attachments/assets/ed7cec0f-8a7c-45b9-89b4-d76829148bfd)

![image](https://github.com/user-attachments/assets/22c820d3-212f-4f9e-9005-ac43382f396b)


