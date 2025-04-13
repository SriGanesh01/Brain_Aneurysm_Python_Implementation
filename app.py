# app.py
import os
import io
import base64
import sys
import contextlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             roc_curve, auc)
from sklearn.calibration import calibration_curve
from collections import Counter
import joblib
import shap

# New Plotly imports for interactive charts
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

app = Flask(__name__)

def capture_info(df):
    """Capture the output of df.info() into a string."""
    buf = io.StringIO()
    df.info(buf=buf)
    info_str = buf.getvalue()
    buf.close()
    return info_str

def run_pipeline():
    """
    Runs the complete aneurysm detection pipeline and returns a dictionary containing:
       - text_logs: a string with printed outputs and commentary.
       - tables: HTML representations of dataframes.
       - images: a list of dictionaries with 'title' and interactive 'html' for each Plotly plot.
    """
    outputs = {"text_logs": "", "tables": {}, "images": []}
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        # 1. Data Loading & Cleaning
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merged.csv")
        df = pd.read_csv(csv_path)
        print("✅ CSV Loaded. Shape:", df.shape)
        print("\n--- Dataset Info ---")
        info_str = capture_info(df)
        print(info_str)
        print("\n--- Basic Statistics ---")
        print(df.describe())

        # Drop rows with missing values
        df_clean = df.dropna()
        print("\n✅ Cleaned Data Shape (after dropping NA):", df_clean.shape)
        outputs["tables"]["Basic Statistics"] = df.describe().to_html(classes="table table-striped")
        outputs["tables"]["Cleaned Data"] = f"<p>{df_clean.shape}</p>"
        outputs["tables"]["Dataset Info"] = f"<pre>{info_str}</pre>"

        # 2. Interactive EDA Plots with Plotly

        # 2a. Count Plot for 'ruptureStatus'
        fig = px.histogram(
            df_clean, 
            x='ruptureStatus', 
            color='ruptureStatus', 
            title="Distribution of Rupture Status (R=Ruptured, U=Unruptured)",
            labels={'ruptureStatus': "Rupture Status"}
        )
        div_count = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        outputs["images"].append({"title": "Count Plot", "html": div_count, "type": "plotly"})

        # 2b. Correlation Heatmap (interactive version instead of clustermap)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        filtered_numeric = [col for col in numeric_cols if df_clean[col].std() > 0]
        corr_matrix = df_clean[filtered_numeric].corr().replace([np.inf, -np.inf], np.nan).fillna(0)
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="viridis",
            title="Correlation Heatmap of Numeric Features"
        )
        div_corr = pio.to_html(fig, full_html=False, include_plotlyjs=False)
        outputs["images"].append({"title": "Correlation Heatmap", "html": div_corr, "type": "plotly"})

        # 2c. Histograms of a sample of numeric features using subplots
        sample_features = filtered_numeric[:6]
        fig = make_subplots(rows=2, cols=3, subplot_titles=sample_features)
        for i, feat in enumerate(sample_features):
            row = i // 3 + 1
            col = i % 3 + 1
            fig.add_trace(
                go.Histogram(x=df_clean[feat], nbinsx=20, name=feat),
                row=row, col=col
            )
        fig.update_layout(title_text="Histograms of Sample Numeric Features", showlegend=False)
        div_hist = pio.to_html(fig, full_html=False, include_plotlyjs=False)
        outputs["images"].append({"title": "Histograms", "html": div_hist, "type": "plotly"})

        # 2d. PCA Visualization (interactive scatter plot)
        from sklearn.decomposition import PCA
        scaler_viz = StandardScaler()
        X_viz = scaler_viz.fit_transform(df_clean[filtered_numeric])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_viz)
        fig = px.scatter(
            x=X_pca[:, 0], 
            y=X_pca[:, 1], 
            color=df_clean['ruptureStatus'],
            labels={'x':"Principal Component 1", 'y': "Principal Component 2"},
            title="PCA of Numeric Features"
        )
        div_pca = pio.to_html(fig, full_html=False, include_plotlyjs=False)
        outputs["images"].append({"title": "PCA Plot", "html": div_pca, "type": "plotly"})

        # 3. Data Preparation for Modeling
        label_col = 'ruptureStatus'
        ignore_cols = ['case_id', 'patient_id', 'aneurysmLocation', 'aneurysmType', 'sex', 'vesselName']
        feature_cols = [col for col in df_clean.columns if col not in ignore_cols + [label_col]]
        print("\nFeatures used for modeling:")
        print(feature_cols)
        X = df_clean[feature_cols]
        y = df_clean[label_col]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        print("\nTarget classes after encoding:", le.classes_)

        # 4. Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
        print("\n✅ Data split: Training size =", X_train.shape[0], ", Testing size =", X_test.shape[0])

        # 5. Modeling: Random Forest and XGBoost Comparison
        models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"{name} Accuracy: {acc * 100:.2f}%")
            print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

        # 6. Hyperparameter Tuning (Random Forest Example)
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

        # 7. Advanced Model Evaluation (Using Best Random Forest)
        y_pred_best = best_rf.predict(X_test)
        # Interactive Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_best)
        fig = px.imshow(
            cm,
            text_auto=True,
            x=le.classes_,
            y=le.classes_,
            color_continuous_scale="Blues",
            title="Confusion Matrix (Best Random Forest)"
        )
        div_cm = pio.to_html(fig, full_html=False, include_plotlyjs=False)
        outputs["images"].append({"title": "Confusion Matrix", "html": div_cm, "type": "plotly"})

        # ROC Curve
        y_prob = best_rf.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Best RF (AUC = {roc_auc:.2f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), showlegend=False))
        fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        div_roc = pio.to_html(fig, full_html=False, include_plotlyjs=False)
        outputs["images"].append({"title": "ROC Curve", "html": div_roc, "type": "plotly"})

        # Calibration Curve
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='markers+lines', name='Best RF Calibration'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), showlegend=False))
        fig.update_layout(title='Calibration Curve', xaxis_title='Mean Predicted Probability', yaxis_title='Fraction of Positives')
        div_cal = pio.to_html(fig, full_html=False, include_plotlyjs=False)
        outputs["images"].append({"title": "Calibration Curve", "html": div_cal, "type": "plotly"})

        cv_scores = cross_val_score(best_rf, X_scaled, y_encoded, cv=cv)
        print("\nCross-Validation Accuracy (Best RF): {:.2f}% ± {:.2f}%".format(
            cv_scores.mean() * 100, cv_scores.std() * 100))
        
        # 8. SHAP Analysis (static plot)
        print("\nPerforming SHAP analysis...")
        explainer = shap.TreeExplainer(best_rf)
        shap_values = explainer.shap_values(X_test)
        # Convert X_test to DataFrame for proper column names.
        X_test_df = pd.DataFrame(X_test, columns=feature_cols)
        shap_vals = shap_values[1]
        if shap_vals.shape != X_test_df.shape:
            shap_vals = shap_vals.T
        plt.figure()
        shap.summary_plot(shap_vals, X_test_df, feature_names=feature_cols, plot_type="bar", show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        plt.clf()
        # Add SHAP plot as a static image
        outputs["images"].append({"title": "SHAP Summary", "html": f'<img src="data:image/png;base64,{img_str}" alt="SHAP Summary" class="hover-image">', "type": "static"})

    outputs["text_logs"] = log.getvalue()
    log.close()
    return outputs

@app.route('/pipeline')
def pipeline():
    """Run the pipeline and render the output on a webpage."""
    results = run_pipeline()
    return render_template("pipeline.html", logs=results["text_logs"], tables=results["tables"], images=results["images"])

@app.route('/')
def index():
    """Landing page with navigation to the pipeline results."""
    return render_template("pipeline.html", logs="Welcome! Click the 'Run Pipeline' button below to execute the aneurysm detection pipeline and view all outputs.", tables={}, images=[])

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
