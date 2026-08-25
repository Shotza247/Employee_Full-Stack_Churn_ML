# Databricks notebook source
import logging
from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .config import MODEL_PATH
from .database import get_training_data, save_model_run
FEATURE_COLUMNS = ["age", "income", "tenure_months"]

def build_model():
    return Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(random_state=42))])

def train_model(db_path=None, model_path=MODEL_PATH):
    data = get_training_data(db_path) if db_path else get_training_data()
    if data.empty or data["churn"].nunique() < 2: raise ValueError("Training data must contain both churn classes (0 and 1).")
    X, y = data[FEATURE_COLUMNS], data["churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = build_model().fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]; labels = (probabilities >= 0.5).astype(int)
    metrics = {"accuracy": float(accuracy_score(y_test, labels)), "precision": float(precision_score(y_test, labels, zero_division=0)), "recall": float(recall_score(y_test, labels, zero_division=0)), "f1": float(f1_score(y_test, labels, zero_division=0)), "roc_auc": float(roc_auc_score(y_test, probabilities))}
    artifact_path = Path(model_path); artifact_path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(model, artifact_path)
    all_probabilities = model.predict_proba(X)[:, 1]
    predictions = pd.DataFrame({"employee_id": data["employee_id"], "predicted_prob": all_probabilities, "predicted_label": (all_probabilities >= 0.5).astype(int), "true_label": y})
    run_id = save_model_run(metrics, predictions, db_path) if db_path else save_model_run(metrics, predictions)
    logging.info("Saved model to %s as run %s", artifact_path, run_id)
    return model, metrics, predictions, run_id