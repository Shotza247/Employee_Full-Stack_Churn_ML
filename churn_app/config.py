# Databricks notebook source
from pathlib import Path
import os
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("CHURN_DB_PATH", PROJECT_ROOT / "employees.db"))
MODEL_PATH = Path(os.getenv("CHURN_MODEL_PATH", PROJECT_ROOT / "churn_model.pkl"))