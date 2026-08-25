# Databricks notebook source
from pathlib import Path
import joblib
from .config import MODEL_PATH

def risk_band(probability):
    return "high" if probability >= 0.7 else "medium" if probability >= 0.4 else "low"

def load_model(model_path=MODEL_PATH):
    path = Path(model_path)
    if not path.exists(): raise FileNotFoundError(f"Model artifact not found: {path}. Run train_and_evaluate.py first.")
    return joblib.load(path)

def predict_employee(model, row):
    probability = float(model.predict_proba([[row["age"], row["annual_income"], row["tenure_months"]]])[0][1])
    return {"churn_probability": probability, "risk_band": risk_band(probability), "age": int(row["age"]), "income": float(row["annual_income"]), "tenure_months": int(row["tenure_months"])}