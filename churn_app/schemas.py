# Databricks notebook source
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    employee_id: int

class PredictionResponse(BaseModel):
    churn_probability: float
    risk_band: str
    age: int
    income: float
    tenure_months: int