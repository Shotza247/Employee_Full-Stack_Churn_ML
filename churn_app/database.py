# Databricks notebook source
import sqlite3
from pathlib import Path
from contextlib import contextmanager
import pandas as pd
from .config import DB_PATH

SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (employee_id INTEGER PRIMARY KEY AUTOINCREMENT, first_name TEXT NOT NULL, last_name TEXT NOT NULL, age INTEGER NOT NULL, department TEXT NOT NULL, contract_type TEXT NOT NULL, churn INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS salaries (salary_id INTEGER PRIMARY KEY AUTOINCREMENT, employee_id INTEGER NOT NULL UNIQUE, annual_income REAL NOT NULL);
CREATE TABLE IF NOT EXISTS jobs (job_id INTEGER PRIMARY KEY AUTOINCREMENT, employee_id INTEGER NOT NULL UNIQUE, job_title TEXT NOT NULL, tenure_months INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS model_runs (run_id INTEGER PRIMARY KEY AUTOINCREMENT, run_timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, accuracy REAL, precision REAL, recall REAL, f1 REAL, roc_auc REAL);
CREATE TABLE IF NOT EXISTS predictions (prediction_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL, employee_id INTEGER NOT NULL, predicted_prob REAL NOT NULL, predicted_label INTEGER NOT NULL, true_label INTEGER, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
"""
SAMPLE_EMPLOYEES = [("Aaliyah", "Smith", 29, "Marketing", "permanent", 72000.0, "Analyst", 18, 0), ("Noah", "Johnson", 42, "Sales", "permanent", 92000.0, "Account Executive", 72, 0), ("Mia", "Williams", 34, "Engineering", "contract", 104000.0, "Software Engineer", 36, 0), ("James", "Brown", 51, "HR", "permanent", 78000.0, "HR Business Partner", 96, 0), ("Emma", "Jones", 27, "Support", "contract", 61000.0, "Support Specialist", 12, 1), ("Liam", "Garcia", 38, "Operations", "permanent", 88000.0, "Operations Manager", 60, 0), ("Olivia", "Miller", 31, "Finance", "permanent", 91000.0, "Financial Analyst", 30, 0), ("Ethan", "Davis", 45, "Sales", "permanent", 99000.0, "Regional Manager", 84, 0), ("Ava", "Rodriguez", 26, "Marketing", "contract", 64000.0, "Content Creator", 14, 1), ("Sophia", "Martinez", 41, "Engineering", "permanent", 115000.0, "Team Lead", 78, 0), ("Jackson", "Hernandez", 53, "Support", "permanent", 72000.0, "Support Manager", 108, 0), ("Isabella", "Lopez", 30, "Finance", "contract", 78000.0, "Payroll Specialist", 20, 0), ("Lucas", "Gonzalez", 36, "Engineering", "permanent", 108000.0, "DevOps Engineer", 48, 0), ("Charlotte", "Wilson", 39, "HR", "permanent", 82000.0, "Recruiter", 52, 0), ("Mason", "Anderson", 28, "Sales", "contract", 65000.0, "Sales Associate", 16, 1), ("Amelia", "Thomas", 47, "Operations", "permanent", 93000.0, "Logistics Specialist", 70, 0), ("Logan", "Taylor", 33, "Engineering", "permanent", 102000.0, "QA Engineer", 40, 0), ("Harper", "Moore", 24, "Marketing", "contract", 58000.0, "Social Media Coordinator", 10, 1), ("Elijah", "Jackson", 50, "Finance", "permanent", 98000.0, "Controller", 90, 0), ("Evelyn", "Martin", 32, "Operations", "permanent", 85000.0, "Supply Chain Analyst", 35, 0), ("William", "Lee", 44, "Engineering", "permanent", 118000.0, "Senior Engineer", 75, 0), ("Abigail", "Perez", 37, "HR", "contract", 69000.0, "HR Coordinator", 22, 1), ("Benjamin", "Thompson", 29, "Sales", "permanent", 86000.0, "Sales Engineer", 28, 0), ("Emily", "White", 40, "Marketing", "permanent", 94000.0, "Marketing Manager", 66, 0), ("Henry", "Harris", 35, "Engineering", "contract", 98000.0, "Backend Developer", 44, 0), ("Eleanor", "Sanchez", 31, "Finance", "permanent", 89000.0, "Accounting Manager", 38, 0)]

@contextmanager
def get_connection(db_path=DB_PATH):
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()

def init_db(db_path=DB_PATH):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as connection:
        connection.executescript(SQL_SCHEMA)
        columns = {row[1] for row in connection.execute("PRAGMA table_info(employees)")}
        if "churn" not in columns:
            connection.execute("ALTER TABLE employees ADD COLUMN churn INTEGER NOT NULL DEFAULT 0")
        _repair_legacy_demo_labels(connection)


def _repair_legacy_demo_labels(connection):
    """Restore labels lost when the churn column was added to an old demo DB."""
    if connection.execute("SELECT COUNT(DISTINCT churn) FROM employees").fetchone()[0] != 1:
        return
    labels = {(employee[0], employee[1]): employee[8] for employee in SAMPLE_EMPLOYEES}
    for (first_name, last_name), churn in labels.items():
        connection.execute(
            "UPDATE employees SET churn = ? WHERE first_name = ? AND last_name = ?",
            (churn, first_name, last_name),
        )

def populate_sample_data(db_path=DB_PATH):
    init_db(db_path)
    with get_connection(db_path) as connection:
        if connection.execute("SELECT COUNT(*) FROM employees").fetchone()[0]: return
        for employee in SAMPLE_EMPLOYEES:
            cursor = connection.execute("INSERT INTO employees (first_name,last_name,age,department,contract_type,churn) VALUES (?,?,?,?,?,?)", employee[:5] + (employee[8],))
            employee_id = cursor.lastrowid
            connection.execute("INSERT INTO salaries VALUES (NULL,?,?)", (employee_id, employee[5]))
            connection.execute("INSERT INTO jobs VALUES (NULL,?,?,?)", (employee_id, employee[6], employee[7]))

def get_employee_features(employee_id, db_path=DB_PATH):
    with get_connection(db_path) as connection:
        return connection.execute("SELECT e.age,e.churn,s.annual_income,j.tenure_months FROM employees e JOIN salaries s ON s.employee_id=e.employee_id JOIN jobs j ON j.employee_id=e.employee_id WHERE e.employee_id=?", (employee_id,)).fetchone()

def get_all_employees(db_path=DB_PATH):
    with get_connection(db_path) as connection:
        return connection.execute("SELECT e.employee_id,e.first_name,e.last_name,e.age,e.department,e.contract_type,s.annual_income,j.job_title,j.tenure_months FROM employees e JOIN salaries s ON s.employee_id=e.employee_id JOIN jobs j ON j.employee_id=e.employee_id ORDER BY e.employee_id").fetchall()

def get_training_data(db_path=DB_PATH):
    with get_connection(db_path) as connection:
        return pd.read_sql_query("SELECT e.employee_id,e.age,s.annual_income AS income,j.tenure_months,e.churn FROM employees e JOIN salaries s ON s.employee_id=e.employee_id JOIN jobs j ON j.employee_id=e.employee_id ORDER BY e.employee_id", connection)

def insert_employee(first_name,last_name,age,department,contract_type,annual_income,job_title,tenure_months,db_path=DB_PATH):
    return insert_employee_with_label(first_name, last_name, age, department, contract_type, annual_income, job_title, tenure_months, None, db_path)

def insert_employee_with_label(first_name,last_name,age,department,contract_type,annual_income,job_title,tenure_months,churn=None,db_path=DB_PATH):
    with get_connection(db_path) as connection:
        if churn is None:
            employee_id = connection.execute("INSERT INTO employees (first_name,last_name,age,department,contract_type) VALUES (?,?,?,?,?)", (first_name,last_name,age,department,contract_type)).lastrowid
        else:
            employee_id = connection.execute("INSERT INTO employees (first_name,last_name,age,department,contract_type,churn) VALUES (?,?,?,?,?,?)", (first_name,last_name,age,department,contract_type,churn)).lastrowid
        connection.execute("INSERT INTO salaries VALUES (NULL,?,?)", (employee_id,annual_income))
        connection.execute("INSERT INTO jobs VALUES (NULL,?,?,?)", (employee_id,job_title,tenure_months))
        return employee_id

def import_employee_dataframe(data, db_path=DB_PATH):
    required = {"first_name", "last_name", "age", "department", "contract_type", "annual_income", "job_title", "tenure_months"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    accepted, rejected, seen = [], [], set()
    valid_departments = {"Marketing", "Sales", "Engineering", "HR", "Support", "Operations", "Finance"}
    valid_contracts = {"permanent", "contract"}
    init_db(db_path)
    for row_number, row in data.iterrows():
        reason = None
        try:
            key = (str(row.first_name).strip().lower(), str(row.last_name).strip().lower(), int(row.age), float(row.annual_income), int(row.tenure_months))
            if not row.first_name or not row.last_name or not row.job_title:
                reason = "missing name or job title"
            elif not 18 <= key[2] <= 99 or key[3] < 0 or not 0 <= key[4] <= 240:
                reason = "age, income, or tenure outside accepted range"
            elif row.department not in valid_departments or row.contract_type not in valid_contracts:
                reason = "unknown department or contract type"
            elif "churn" in data.columns and int(row.churn) not in {0, 1}:
                reason = "churn must be 0 or 1"
            elif key in seen:
                reason = "duplicate record in uploaded file"
            else:
                seen.add(key)
                with get_connection(db_path) as connection:
                    existing = connection.execute(
                        """SELECT COUNT(*) FROM employees e
                           JOIN salaries s ON s.employee_id = e.employee_id
                           JOIN jobs j ON j.employee_id = e.employee_id
                           WHERE lower(e.first_name) = ? AND lower(e.last_name) = ?
                           AND e.age = ? AND s.annual_income = ? AND j.tenure_months = ?""",
                        key,
                    ).fetchone()[0]
                if existing:
                    reason = "record already exists in database"
        except (TypeError, ValueError):
            reason = "non-numeric age, income, tenure, or churn"
        if reason:
            rejected.append({"source_row": int(row_number) + 2, "reason": reason, **row.to_dict()})
            continue
        employee_id = insert_employee_with_label(row.first_name.strip(), row.last_name.strip(), key[2], row.department, row.contract_type, key[3], row.job_title.strip(), key[4], int(row.churn) if "churn" in data.columns else None, db_path)
        accepted.append({"source_row": int(row_number) + 2, "employee_id": employee_id})
    return pd.DataFrame(accepted), pd.DataFrame(rejected)

def save_model_run(metrics,predictions,db_path=DB_PATH):
    with get_connection(db_path) as connection:
        run_id = connection.execute("INSERT INTO model_runs (accuracy,precision,recall,f1,roc_auc) VALUES (?,?,?,?,?)", tuple(metrics[name] for name in ("accuracy","precision","recall","f1","roc_auc"))).lastrowid
        for row in predictions.itertuples(index=False): connection.execute("INSERT INTO predictions (run_id,employee_id,predicted_prob,predicted_label,true_label) VALUES (?,?,?,?,?)", (run_id,int(row.employee_id),float(row.predicted_prob),int(row.predicted_label),int(row.true_label)))
        return run_id

def get_model_runs(start_ts=None,end_ts=None,db_path=DB_PATH):
    query="SELECT * FROM model_runs"; params=[]
    if start_ts and end_ts: query += " WHERE run_timestamp BETWEEN ? AND ?"; params=[start_ts,end_ts]
    with get_connection(db_path) as connection: return pd.read_sql_query(query + " ORDER BY run_timestamp DESC", connection, params=params)

def get_predictions_by_run(run_id,db_path=DB_PATH):
    with get_connection(db_path) as connection: return pd.read_sql_query("SELECT * FROM predictions WHERE run_id=? ORDER BY prediction_id", connection, params=[run_id])