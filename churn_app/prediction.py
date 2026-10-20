# Databricks notebook source
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .config import MODEL_PATH
from .feature_engineering import impute_and_engineer_features
from .training import CATEGORICAL_FEATURES, FEATURE_COLUMNS


def risk_band(probability: float) -> str:
    return "high" if probability >= 0.7 else "medium" if probability >= 0.4 else "low"


def load_model(model_path=MODEL_PATH):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}. Run train_and_evaluate.py first.")
    model = joblib.load(path)
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is None or list(model_features) != FEATURE_COLUMNS:
        raise ValueError(
            "The model artifact was trained with an incompatible feature set. "
            "Retrain it with `python -m churn_app.train_and_evaluate`."
        )
    return model


def predict_employee(model, row):
    raw_dict = {
        "age": row["age"],
        "annual_income": row["annual_income"],
        "tenure_months": row["tenure_months"],
        "department": row["department"],
        "contract_type": row["contract_type"],
        "job_title": row["job_title"],
    }
    raw_df = pd.DataFrame([raw_dict])
    clean_df = impute_and_engineer_features(raw_df)
    features = pd.DataFrame([[
        clean_df.iloc[0]["age"],
        clean_df.iloc[0]["annual_income"],
        clean_df.iloc[0]["tenure_months"],
        clean_df.iloc[0]["department"],
        clean_df.iloc[0]["contract_type"],
        clean_df.iloc[0]["job_title"],
    ]], columns=FEATURE_COLUMNS)

    probability = float(model.predict_proba(features)[0][1])
    return {
        "churn_probability": probability,
        "risk_band": risk_band(probability),
        "age": int(clean_df.iloc[0]["age"]),
        "income": float(clean_df.iloc[0]["annual_income"]),
        "tenure_months": int(clean_df.iloc[0]["tenure_months"]),
    }
