# Databricks notebook source
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .database import get_employee_features, init_db, populate_sample_data, save_inference_predictions
from .prediction import load_model, predict_employee
from .schemas import FeaturePredictionRequest, PredictionRequest, PredictionResponse

app = FastAPI(
    title="Employee Churn Prediction API",
    description="Serverless-ready FastAPI endpoint for employee churn predictions.",
    version="1.0.0",
)

# Enable CORS for deployed frontend applications (e.g. Streamlit Cloud)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database & load model
init_db()
populate_sample_data()
model = load_model()


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Employee Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    row = get_employee_features(request.employee_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Employee not found: {request.employee_id}")
    result = predict_employee(model, row)
    try:
        save_inference_predictions([{
            "employee_id": request.employee_id,
            "predicted_prob": result["churn_probability"],
            "predicted_label": int(result["churn_probability"] >= 0.5),
        }])
    except Exception as err:
        logging.warning("Inference record could not be persisted: %s", err)

    logging.info("Predicted churn probability for employee %s: %.4f", request.employee_id, result["churn_probability"])
    return PredictionResponse(**result)


@app.post("/predict/features", response_model=PredictionResponse)
def predict_features(request: FeaturePredictionRequest):
    result = predict_employee(model, request.model_dump())
    logging.info("Predicted churn probability from features: %.4f", result["churn_probability"])
    return PredictionResponse(**result)
