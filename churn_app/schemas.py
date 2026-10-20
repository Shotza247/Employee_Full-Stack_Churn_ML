# Databricks notebook source
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    employee_id: int


class FeaturePredictionRequest(BaseModel):
    age: int | None = None
    department: str | None = None
    contract_type: str | None = None
    annual_income: float | None = None
    job_title: str | None = None
    tenure_months: int | None = None


class PredictionResponse(BaseModel):
    churn_probability: float
    risk_band: str
    age: int | None
    income: float | None
    tenure_months: int | None
