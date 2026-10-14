# Databricks notebook source
import logging
from fastapi import FastAPI, HTTPException
from .database import get_employee_features, init_db, populate_sample_data, save_inference_predictions
from .prediction import load_model, predict_employee
from .schemas import PredictionRequest, PredictionResponse
app = FastAPI(title="Employee Churn Prediction API")
init_db(); populate_sample_data(); model = load_model()

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    row = get_employee_features(request.employee_id)
    if row is None: raise HTTPException(status_code=404, detail=f"Employee not found: {request.employee_id}")
    result = predict_employee(model, row)
    save_inference_predictions([{
        "employee_id": request.employee_id,
        "predicted_prob": result["churn_probability"],
        "predicted_label": int(result["churn_probability"] >= 0.5),
    }])
    logging.info("Predicted churn probability for employee %s: %.4f", request.employee_id, result["churn_probability"])
    return PredictionResponse(**result)