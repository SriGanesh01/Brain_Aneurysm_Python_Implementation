"""
Advanced Aneurysm Rupture Detection Pipeline

This script provides an in‐depth analysis of the aneurysm dataset contained in merged.csv.
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
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             roc_curve, auc)
from sklearn.calibration import calibration_curve  # Correct import location
import joblib
import shap
from collections import Counter

# --------------------------
# 1. Data Loading and Cleaning
# --------------------------
csv_path = r"merged.csv"  # Ensure merged.csv is in the same folder
df = pd.read_csv(csv_path)
print("✅ CSV Loaded. Shape:", df.shape)

print("\n--- Dataset Info ---")
print(df.info())
print("\n--- Basic Statistics ---")
print(df.describe())

# Drop rows with missing values (note: this may reduce sample size drastically)
df_clean = df.dropna()
print("\n✅ Cleaned Data Shape (after dropping NA):", df_clean.shape)

# --------------------------
# 2. Exploratory Data Analysis (EDA)
# --------------------------
# 2a. Plot target distribution (ruptureStatus: e.g., 'R' = ruptured, 'U' = unruptured)
plt.figure(figsize=(6, 4))
sns.countplot(x='ruptureStatus', data=df_clean, palette='Set2')
plt.title("Distribution of Rupture Status (R=Ruptured, U=Unruptured)")
plt.xlabel("Rupture Status")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 2b. Correlation Clustermap among numeric features
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
# Remove columns with zero variation
filtered_numeric = [col for col in numeric_cols if df_clean[col].std() > 0]
corr_matrix = df_clean[filtered_numeric].corr()
corr_matrix = corr_matrix.replace([np.inf, -np.inf], np.nan).fillna(0)
plt.figure(figsize=(12, 10))
sns.clustermap(corr_matrix, cmap='viridis', linewidths=0.5)
plt.title("Clustermap of Numeric Feature Correlations", pad=100)
plt.tight_layout()
plt.show()

# 2c. Histogram distributions for a sample of numeric features
sample_features = filtered_numeric[:6]  # Use first six numeric features for sample plots
df_clean[sample_features].hist(bins=20, figsize=(12, 8))
plt.suptitle("Histograms of Sample Numeric Features")
plt.tight_layout()
plt.show()

# 2d. PCA visualization for class separability using numeric features
from sklearn.decomposition import PCA
scaler_viz = StandardScaler()
X_viz = scaler_viz.fit_transform(df_clean[filtered_numeric])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_viz)
plt.figure(figsize=(8, 6))
le_viz = LabelEncoder()
y_viz = le_viz.fit_transform(df_clean['ruptureStatus'])
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_viz, cmap='coolwarm', alpha=0.6)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Numeric Features")
plt.colorbar(label='Rupture Status (Encoded)')
plt.tight_layout()
plt.show()

# --------------------------
# 3. Data Preparation for Modeling
# --------------------------
label_col = 'ruptureStatus'
ignore_cols = ['case_id', 'patient_id', 'aneurysmLocation', 'aneurysmType', 'sex', 'vesselName']
feature_cols = [col for col in df_clean.columns if col not in ignore_cols + [label_col]]
print("\nFeatures used for modeling:")
print(feature_cols)

X = df_clean[feature_cols]
y = df_clean[label_col]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode target variable ('R' and 'U')
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("\nTarget classes after encoding:", le.classes_)

# --------------------------
# 4. Train/Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
print("\n✅ Data split: Training size =", X_train.shape[0], ", Testing size =", X_test.shape[0])

# --------------------------
# 5. Modeling: Random Forest and XGBoost Comparison
# --------------------------
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}
model_results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc * 100:.2f}%")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
    model_results[name] = (model, acc)

# --------------------------
# 6. Hyperparameter Tuning (Random Forest Example)
# --------------------------
# Determine cv based on the smallest class count to avoid errors.
from collections import Counter
train_class_counts = Counter(y_train)
min_count = min(train_class_counts.values())
n_splits = min(5, min_count)
if n_splits < 2:
    n_splits = 2
print(f"\nUsing {n_splits}-fold StratifiedKFold for GridSearchCV.")
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print("\nBest parameters from Grid Search (Random Forest):", grid_search.best_params_)
print("Best cross-validation accuracy: {:.2f}%".format(grid_search.best_score_ * 100))
best_rf = grid_search.best_estimator_

# --------------------------
# 7. Advanced Model Evaluation (Using Best Random Forest)
# --------------------------
# 7a. Confusion Matrix
y_pred_best = best_rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix (Best Random Forest)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# 7b. ROC Curve and AUC
y_prob = best_rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Best RF (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 7c. Calibration Curve
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Best RF Calibration')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curve")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()

# 7d. Cross-Validation Scores for Best RF
cv_scores = cross_val_score(best_rf, X_scaled, y_encoded, cv=cv)
print("\nCross-Validation Accuracy (Best RF): {:.2f}% ± {:.2f}%".format(
    cv_scores.mean() * 100, cv_scores.std() * 100))

# --------------------------
# 8. Model Interpretability with SHAP (Using Best RF)
# --------------------------
print("\nPerforming SHAP analysis...")
explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test)

# Convert X_test back to a DataFrame with feature names.
X_test_df = pd.DataFrame(X_test, columns=feature_cols)
print("X_test_df shape:", X_test_df.shape)
print("SHAP value shapes:", [sv.shape for sv in shap_values])

# For binary classification, shap_values[1] should correspond to class "1".
# Here, we transpose the SHAP array so that its shape becomes (n_samples, n_features)
shap_values_transposed = shap_values[1].T  # Now shape should be (n_samples, n_features)
# Confirm the shape matches X_test_df (should be (2,55))
print("Transposed SHAP values shape:", shap_values_transposed.shape)

# Generate SHAP summary plot (bar plot)
shap.summary_plot(shap_values_transposed, X_test_df, feature_names=feature_cols, plot_type="bar")
plt.title("SHAP Feature Importance (Bar Plot)")
plt.tight_layout()
plt.show()

# --------------------------
# 9. Save the Best Model and Scaler
# --------------------------
model_filename = "best_rf_model.pkl"
scaler_filename = "scaler.pkl"
joblib.dump(best_rf, model_filename)
joblib.dump(scaler, scaler_filename)
print(f"\n✅ Final best Random Forest model saved as '{model_filename}'")
print(f"✅ Scaler saved as '{scaler_filename}'")
