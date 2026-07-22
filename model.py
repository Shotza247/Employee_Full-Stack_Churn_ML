# Databricks notebook source
from fastapi import FastAPI, HTTPException
import joblib
import logging

from service_setup import PredictionRequest, PredictionResponse
from database import init_db, populate_sample_data, get_employee_features

logging.basicConfig(level=logging.INFO)
logging.info("Starting the prediction service")
app = FastAPI()

init_db()
populate_sample_data()

# Load the trained model once, when the service starts
model = joblib.load("churn_model.pkl")
logging.info("Model loaded successfully")

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    row = get_employee_features(request.employee_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Employee not found: {request.employee_id}")

    age = row["age"]
    income = row["annual_income"]
    tenure_months = row["tenure_months"]
    features = [[age, income, tenure_months]]

    probability = float(model.predict_proba(features)[0][1])
    logging.info(
        f"Predicted churn probability for employee {request.employee_id}: {probability:.4f}"
    )

    if probability >= 0.7:
        risk_band = "high"
        logging.warning(f"High churn risk detected: {probability:.4f}")
    elif probability >= 0.4:
        risk_band = "medium"
        logging.info(f"Medium churn risk detected: {probability:.4f}")
    else:
        risk_band = "low"
        logging.info(f"Low churn risk detected: {probability:.4f}")

    logging.info("Prediction response sent successfully")
    return PredictionResponse(
        churn_probability=probability,
        risk_band=risk_band,
        age=age,
        income=income,
        tenure_months=tenure_months,
    )
    