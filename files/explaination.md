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

## How to Run the Pipeline:
1. Ensure you have Python installed along with the required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib xgboost shap
