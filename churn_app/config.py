# Databricks notebook source
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# In Vercel / serverless environments, root is read-only so use /tmp for SQLite
if os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
    tmp_db = Path("/tmp/employees.db")
    bundled_db = PROJECT_ROOT / "employees.db"
    if not tmp_db.exists() and bundled_db.exists():
        try:
            shutil.copyfile(bundled_db, tmp_db)
        except Exception:
            pass
    DB_PATH = Path(os.getenv("CHURN_DB_PATH", tmp_db))
else:
    DB_PATH = Path(os.getenv("CHURN_DB_PATH", PROJECT_ROOT / "employees.db"))

MODEL_PATH = Path(os.getenv("CHURN_MODEL_PATH", PROJECT_ROOT / "churn_model.pkl"))
