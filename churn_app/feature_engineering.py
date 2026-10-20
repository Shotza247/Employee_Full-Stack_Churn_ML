# Databricks notebook source
import numpy as np
import pandas as pd
from .config import DB_PATH
from .database import get_training_data

DEFAULT_FALLBACKS = {
    "age": 35,
    "annual_income": 75000.0,
    "tenure_months": 36,
    "department": "Engineering",
    "contract_type": "permanent",
    "job_title": "Specialist",
}


def get_imputation_baselines(db_path=DB_PATH) -> dict:
    """Compute central tendency metrics (medians/modes) from database records for imputation."""
    try:
        data = get_training_data(db_path)
        if not data.empty:
            baselines = {
                "age": int(data["age"].dropna().median()) if data["age"].notna().any() else DEFAULT_FALLBACKS["age"],
                "annual_income": float(data["income"].dropna().median()) if data["income"].notna().any() else DEFAULT_FALLBACKS["annual_income"],
                "tenure_months": int(data["tenure_months"].dropna().median()) if data["tenure_months"].notna().any() else DEFAULT_FALLBACKS["tenure_months"],
                "department": str(data["department"].dropna().mode().iloc[0]) if data["department"].notna().any() else DEFAULT_FALLBACKS["department"],
                "contract_type": str(data["contract_type"].dropna().mode().iloc[0]) if data["contract_type"].notna().any() else DEFAULT_FALLBACKS["contract_type"],
                "job_title": str(data["job_title"].dropna().mode().iloc[0]) if data["job_title"].notna().any() else DEFAULT_FALLBACKS["job_title"],
            }
            return baselines
    except Exception:
        pass
    return DEFAULT_FALLBACKS.copy()


def impute_and_engineer_features(df: pd.DataFrame, db_path=DB_PATH) -> pd.DataFrame:
    """
    Ensure zero null/none values exist before model inference.
    Applies domain feature engineering and imputes any missing values using learned baselines.
    """
    if df.empty:
        return df

    cleaned = df.copy()
    baselines = get_imputation_baselines(db_path)

    # Impute numeric features
    if "age" in cleaned.columns:
        cleaned["age"] = pd.to_numeric(cleaned["age"], errors="coerce").fillna(baselines["age"]).astype(int)
    else:
        cleaned["age"] = baselines["age"]

    if "annual_income" in cleaned.columns:
        cleaned["annual_income"] = pd.to_numeric(cleaned["annual_income"], errors="coerce").fillna(baselines["annual_income"]).astype(float)
    elif "income" in cleaned.columns:
        cleaned["annual_income"] = pd.to_numeric(cleaned["income"], errors="coerce").fillna(baselines["annual_income"]).astype(float)
    else:
        cleaned["annual_income"] = baselines["annual_income"]

    if "tenure_months" in cleaned.columns:
        cleaned["tenure_months"] = pd.to_numeric(cleaned["tenure_months"], errors="coerce").fillna(baselines["tenure_months"]).astype(int)
    else:
        cleaned["tenure_months"] = baselines["tenure_months"]

    # Impute categorical features
    if "department" in cleaned.columns:
        cleaned["department"] = cleaned["department"].fillna(baselines["department"]).astype(str).replace("", baselines["department"])
    else:
        cleaned["department"] = baselines["department"]

    if "contract_type" in cleaned.columns:
        cleaned["contract_type"] = cleaned["contract_type"].fillna(baselines["contract_type"]).astype(str).replace("", baselines["contract_type"])
    else:
        cleaned["contract_type"] = baselines["contract_type"]

    if "job_title" in cleaned.columns:
        cleaned["job_title"] = cleaned["job_title"].fillna(baselines["job_title"]).astype(str).replace("", baselines["job_title"])
    else:
        cleaned["job_title"] = baselines["job_title"]

    return cleaned
