# Databricks notebook source
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .config import MODEL_PATH
from .training import CATEGORICAL_FEATURES, FEATURE_COLUMNS

def risk_band(probability):
    return "high" if probability >= 0.7 else "medium" if probability >= 0.4 else "low"

def load_model(model_path=MODEL_PATH):
    path = Path(model_path)
    if not path.exists(): raise FileNotFoundError(f"Model artifact not found: {path}. Run train_and_evaluate.py first.")
    return joblib.load(path)

def predict_employee(model, row):
    features = pd.DataFrame(
        [[
            row["age"],
            row["annual_income"],
            row["tenure_months"],
            row["department"],
            row["contract_type"],
            row["job_title"],
        ]],
        columns=FEATURE_COLUMNS,
    )
    features[CATEGORICAL_FEATURES] = features[CATEGORICAL_FEATURES].astype(object).where(
        features[CATEGORICAL_FEATURES].notna(),
        np.nan,
    )
    probability = float(model.predict_proba(features)[0][1])
    age = row["age"]
    income = row["annual_income"]
    tenure_months = row["tenure_months"]
    return {
        "churn_probability": probability,
        "risk_band": risk_band(probability),
        "age": int(age) if age is not None else None,
        "income": float(income) if income is not None else None,
        "tenure_months": int(tenure_months) if tenure_months is not None else None,
    }