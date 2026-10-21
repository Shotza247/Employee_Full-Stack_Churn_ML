# Databricks notebook source
import sqlite3
from math import isfinite
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import pandas as pd
from .config import DB_PATH

SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (employee_id INTEGER PRIMARY KEY AUTOINCREMENT, first_name TEXT NOT NULL, last_name TEXT NOT NULL, age INTEGER, department TEXT, contract_type TEXT, churn INTEGER);
CREATE TABLE IF NOT EXISTS salaries (salary_id INTEGER PRIMARY KEY AUTOINCREMENT, employee_id INTEGER NOT NULL UNIQUE, annual_income REAL);
CREATE TABLE IF NOT EXISTS jobs (job_id INTEGER PRIMARY KEY AUTOINCREMENT, employee_id INTEGER NOT NULL UNIQUE, job_title TEXT, tenure_months INTEGER);
CREATE TABLE IF NOT EXISTS model_runs (run_id INTEGER PRIMARY KEY AUTOINCREMENT, run_timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, accuracy REAL, precision REAL, recall REAL, f1 REAL, roc_auc REAL, model_type TEXT DEFAULT 'logistic_regression');
CREATE TABLE IF NOT EXISTS predictions (prediction_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL, employee_id INTEGER NOT NULL, predicted_prob REAL NOT NULL, predicted_label INTEGER NOT NULL, true_label INTEGER, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS inference_predictions (prediction_id INTEGER PRIMARY KEY AUTOINCREMENT, employee_id INTEGER NOT NULL, predicted_prob REAL NOT NULL, predicted_label INTEGER NOT NULL, true_label INTEGER, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
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


def _is_missing_value(value):
    return pd.isna(value) or (isinstance(value, str) and not value.strip())


def init_db(db_path=DB_PATH):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as connection:
        connection.executescript(SQL_SCHEMA)
        columns = {row[1] for row in connection.execute("PRAGMA table_info(employees)")}
        if "churn" not in columns:
            connection.execute("ALTER TABLE employees ADD COLUMN churn INTEGER NOT NULL DEFAULT 0")
        columns_mr = {row[1] for row in connection.execute("PRAGMA table_info(model_runs)")}
        if "model_type" not in columns_mr:
            connection.execute("ALTER TABLE model_runs ADD COLUMN model_type TEXT DEFAULT 'logistic_regression'")
        _repair_legacy_demo_labels(connection)
        _migrate_nullable_employee_fields(connection)


def _migrate_nullable_employee_fields(connection):
    tables = {
        "employees": (
            "CREATE TABLE employees_nullable (employee_id INTEGER PRIMARY KEY AUTOINCREMENT, first_name TEXT NOT NULL, last_name TEXT NOT NULL, age INTEGER, department TEXT, contract_type TEXT, churn INTEGER)",
            "employee_id, first_name, last_name, age, department, contract_type, churn",
            ("age", "department", "contract_type", "churn"),
        ),
        "salaries": (
            "CREATE TABLE salaries_nullable (salary_id INTEGER PRIMARY KEY AUTOINCREMENT, employee_id INTEGER NOT NULL UNIQUE, annual_income REAL)",
            "salary_id, employee_id, annual_income",
            ("annual_income",),
        ),
        "jobs": (
            "CREATE TABLE jobs_nullable (job_id INTEGER PRIMARY KEY AUTOINCREMENT, employee_id INTEGER NOT NULL UNIQUE, job_title TEXT, tenure_months INTEGER)",
            "job_id, employee_id, job_title, tenure_months",
            ("job_title", "tenure_months"),
        ),
    }
    for table, (create_sql, columns, nullable_columns) in tables.items():
        constraints = {
            row["name"]: row["notnull"]
            for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if not any(constraints.get(column, 0) for column in nullable_columns):
            continue
        replacement = f"{table}_nullable"
        connection.execute(f"DROP TABLE IF EXISTS {replacement}")
        connection.execute(create_sql)
        connection.execute(f"INSERT INTO {replacement} ({columns}) SELECT {columns} FROM {table}")
        connection.execute(f"DROP TABLE {table}")
        connection.execute(f"ALTER TABLE {replacement} RENAME TO {table}")


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
        return connection.execute("SELECT e.age,e.churn,s.annual_income,j.tenure_months,e.department,e.contract_type,j.job_title FROM employees e JOIN salaries s ON s.employee_id=e.employee_id JOIN jobs j ON j.employee_id=e.employee_id WHERE e.employee_id=?", (employee_id,)).fetchone()

def get_all_employees(db_path=DB_PATH):
    with get_connection(db_path) as connection:
        return connection.execute("SELECT e.employee_id,e.first_name,e.last_name,e.age,e.department,e.contract_type,s.annual_income,j.job_title,j.tenure_months FROM employees e JOIN salaries s ON s.employee_id=e.employee_id JOIN jobs j ON j.employee_id=e.employee_id ORDER BY e.employee_id").fetchall()

def get_training_data(db_path=DB_PATH):
    with get_connection(db_path) as connection:
        data = pd.read_sql_query("SELECT e.employee_id,e.age,s.annual_income AS income,j.tenure_months,e.department,e.contract_type,j.job_title,e.churn FROM employees e JOIN salaries s ON s.employee_id=e.employee_id JOIN jobs j ON j.employee_id=e.employee_id ORDER BY e.employee_id", connection)
        categorical_columns = ["department", "contract_type", "job_title"]
        data[categorical_columns] = data[categorical_columns].astype(object).where(
            data[categorical_columns].notna(),
            np.nan,
        )
        return data

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

VALID_DEPARTMENTS = {"Marketing", "Sales", "Engineering", "HR", "Support", "Operations", "Finance"}
VALID_CONTRACTS = {"permanent", "contract"}

LEGACY_ISSUE_MAP = {
    "missing_income": ("Missing Values", "Missing annual income"),
    "missing_financial_feature": ("Missing Values", "Missing financial feature (income)"),
    "missing_contract_type": ("Missing Values", "Missing contract type"),
    "missing_employment_feature": ("Missing Values", "Missing employment feature"),
    "missing_department": ("Missing Values", "Missing department"),
    "missing_name": ("Missing Values", "Missing employee name"),
    "missing_first_name": ("Missing Values", "Missing first name"),
    "missing_last_name": ("Missing Values", "Missing last name"),
    "duplicate_record": ("Duplicate Record", "Duplicate record identity"),
    "duplicate_name_and_features": ("Duplicate Record", "Duplicate name and features"),
    "age_outlier": ("Invalid Range", "Age outlier outside accepted range"),
    "invalid_age_range": ("Invalid Range", "Age outside valid 18–99 range"),
    "tenure_outlier": ("Invalid Range", "Tenure outlier outside accepted range"),
    "invalid_tenure_range": ("Invalid Range", "Tenure outside valid 0–240 months"),
}


def diagnose_row_quality(row, seen_keys=None, check_db=False, db_path=DB_PATH) -> tuple[str, str]:
    """
    Diagnoses the data quality category and detailed issue for a single record.
    Returns:
        tuple[category, detail]
        Categories: "Clean Record", "Missing Values", "Duplicate Record", "Invalid Range", "Invalid Category", "Format Issue"
    """
    record = row if isinstance(row, dict) else row.to_dict()

    # 1. Check for legacy issue codes in raw CSV if provided
    raw_issue = record.get("data_quality_issue")
    if raw_issue is not None and not _is_missing_value(raw_issue):
        raw_str = str(raw_issue).strip().lower()
        if raw_str in LEGACY_ISSUE_MAP:
            return LEGACY_ISSUE_MAP[raw_str]

    # 2. Check for missing values in required fields
    missing_fields = []
    first_name = None if _is_missing_value(record.get("first_name")) else str(record.get("first_name")).strip()
    last_name = None if _is_missing_value(record.get("last_name")) else str(record.get("last_name")).strip()
    department = None if _is_missing_value(record.get("department")) else str(record.get("department")).strip()
    contract_type = None if _is_missing_value(record.get("contract_type")) else str(record.get("contract_type")).strip()
    job_title = None if _is_missing_value(record.get("job_title")) else str(record.get("job_title")).strip()

    income_val = record.get("annual_income") if "annual_income" in record else record.get("income")
    age_val = record.get("age")
    tenure_val = record.get("tenure_months")

    if first_name is None: missing_fields.append("first_name")
    if last_name is None: missing_fields.append("last_name")
    if department is None: missing_fields.append("department")
    if contract_type is None: missing_fields.append("contract_type")
    if job_title is None: missing_fields.append("job_title")
    if _is_missing_value(income_val): missing_fields.append("annual_income")
    if _is_missing_value(age_val): missing_fields.append("age")
    if _is_missing_value(tenure_val): missing_fields.append("tenure_months")

    if missing_fields:
        return "Missing Values", f"Missing: {', '.join(missing_fields)}"

    # 3. Check format / numeric validity
    try:
        age_num = float(age_val)
        income_num = float(income_val)
        tenure_num = float(tenure_val)
    except (ValueError, TypeError):
        return "Format Issue", "Non-numeric values in age, income, or tenure"

    if not (isfinite(age_num) and isfinite(income_num) and isfinite(tenure_num)):
        return "Format Issue", "Non-finite numeric values in age, income, or tenure"

    if not age_num.is_integer() or not tenure_num.is_integer():
        return "Format Issue", "Age and tenure must be whole integer numbers"

    age_int = int(age_num)
    tenure_int = int(tenure_num)

    # 4. Check domain range boundaries
    if not (18 <= age_int <= 99):
        return "Invalid Range", f"Age {age_int} is outside accepted 18–99 range"
    if income_num < 0:
        return "Invalid Range", f"Negative income: R {income_num:,.0f}"
    if not (0 <= tenure_int <= 240):
        return "Invalid Range", f"Tenure {tenure_int}m is outside 0–240 months"

    # 5. Check categorical validity
    if department not in VALID_DEPARTMENTS:
        return "Invalid Category", f"Unknown department '{department}'"
    if contract_type not in VALID_CONTRACTS:
        return "Invalid Category", f"Unknown contract type '{contract_type}'"

    # 6. Check churn format if present
    churn_val = record.get("churn")
    if churn_val is not None and not _is_missing_value(churn_val):
        try:
            churn_num = float(churn_val)
            if not isfinite(churn_num) or churn_num not in (0.0, 1.0):
                return "Format Issue", f"Churn must be 0 or 1 (got {churn_val})"
        except (ValueError, TypeError):
            return "Format Issue", f"Churn must be 0 or 1 (got {churn_val})"

    # 7. Check duplicates
    key = (first_name.lower(), last_name.lower(), age_int, income_num, tenure_int)
    if seen_keys is not None:
        if key in seen_keys:
            return "Duplicate Record", "Duplicate record in uploaded file"
        seen_keys.add(key)

    if check_db:
        try:
            with get_connection(db_path) as connection:
                existing = connection.execute(
                    """SELECT COUNT(*) FROM employees e
                       JOIN salaries s ON s.employee_id = e.employee_id
                       JOIN jobs j ON j.employee_id = e.employee_id
                       WHERE lower(e.first_name) = ? AND lower(e.last_name) = ?
                       AND e.age IS ? AND s.annual_income IS ? AND j.tenure_months IS ?""",
                    key,
                ).fetchone()[0]
                if existing:
                    return "Duplicate Record", "Record already exists in SQLite database"
        except Exception:
            pass

    return "Clean Record", "No issues detected (Valid)"


def diagnose_dataframe_quality(df: pd.DataFrame, db_path=DB_PATH, check_db=False) -> pd.DataFrame:
    """
    Inspects all rows in a DataFrame and adds or populates 'data_quality_category'
    and 'data_quality_issue' columns with explicit descriptive categories.
    """
    if df.empty:
        return df.copy()

    diagnosed = df.copy()
    if "income" in diagnosed.columns and "annual_income" not in diagnosed.columns:
        diagnosed["annual_income"] = diagnosed["income"]

    seen_keys = set()
    categories, issues = [], []
    for _, row in diagnosed.iterrows():
        cat, detail = diagnose_row_quality(row, seen_keys=seen_keys, check_db=check_db, db_path=db_path)
        categories.append(cat)
        if cat == "Clean Record":
            issues.append("Clean Record (No Issues)")
        else:
            issues.append(f"{cat}: {detail}")

    diagnosed["data_quality_category"] = categories
    diagnosed["data_quality_issue"] = issues
    return diagnosed


def import_employee_dataframe(data, db_path=DB_PATH):
    input_data = data.copy()
    if "income" in input_data.columns and "annual_income" not in input_data.columns:
        input_data["annual_income"] = input_data["income"]

    required = {"first_name", "last_name", "age", "department", "contract_type", "annual_income", "job_title", "tenure_months"}
    missing_columns = required - set(input_data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")

    accepted, rejected, seen = [], [], set()
    init_db(db_path)

    for row_number, row in input_data.iterrows():
        cat, detail = diagnose_row_quality(row, seen_keys=seen, check_db=True, db_path=db_path)
        if cat != "Clean Record":
            row_dict = row.to_dict()
            row_dict["data_quality_category"] = cat
            row_dict["data_quality_issue"] = f"{cat}: {detail}"
            rejected.append({"source_row": int(row_number) + 2, "reason": detail, **row_dict})
            continue

        first_name = str(row["first_name"]).strip()
        last_name = str(row["last_name"]).strip()
        age = int(float(row["age"]))
        department = str(row["department"]).strip()
        contract_type = str(row["contract_type"]).strip()
        annual_income = float(row["annual_income"])
        job_title = str(row["job_title"]).strip()
        tenure_months = int(float(row["tenure_months"]))
        churn = None
        if "churn" in input_data.columns and not _is_missing_value(row.get("churn")):
            churn = int(float(row["churn"]))

        employee_id = insert_employee_with_label(
            first_name, last_name, age, department, contract_type, annual_income, job_title, tenure_months, churn, db_path
        )
        accepted.append({"source_row": int(row_number) + 2, "employee_id": employee_id})

    return pd.DataFrame(accepted), pd.DataFrame(rejected)

def save_model_run(metrics,predictions,db_path=DB_PATH,model_type="logistic_regression"):
    with get_connection(db_path) as connection:
        columns_mr = {row[1] for row in connection.execute("PRAGMA table_info(model_runs)")}
        if "model_type" not in columns_mr:
            connection.execute("ALTER TABLE model_runs ADD COLUMN model_type TEXT DEFAULT 'logistic_regression'")
        run_id = connection.execute("INSERT INTO model_runs (accuracy,precision,recall,f1,roc_auc,model_type) VALUES (?,?,?,?,?,?)", tuple(metrics[name] for name in ("accuracy","precision","recall","f1","roc_auc")) + (model_type,)).lastrowid
        for row in predictions.itertuples(index=False): connection.execute("INSERT INTO predictions (run_id,employee_id,predicted_prob,predicted_label,true_label) VALUES (?,?,?,?,?)", (run_id,int(row.employee_id),float(row.predicted_prob),int(row.predicted_label),int(row.true_label)))
        return run_id

def save_inference_predictions(predictions,db_path=DB_PATH):
    with get_connection(db_path) as connection:
        for prediction in predictions:
            cursor = connection.execute(
                """INSERT INTO inference_predictions
                   (employee_id,predicted_prob,predicted_label,true_label)
                   SELECT ?,?,?,churn FROM employees WHERE employee_id=?""",
                (
                    int(prediction["employee_id"]),
                    float(prediction["predicted_prob"]),
                    int(prediction["predicted_label"]),
                    int(prediction["employee_id"]),
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Employee not found while saving prediction: {prediction['employee_id']}")

def get_prediction_history(start_date=None,end_date=None,db_path=DB_PATH):
    query = """WITH prediction_history AS (
                   SELECT 'App inference' AS prediction_source, ip.prediction_id,
                          ip.employee_id, e.first_name || ' ' || e.last_name AS employee_name,
                          ip.predicted_prob, ip.predicted_label, ip.true_label, ip.created_at
                   FROM inference_predictions ip
                   JOIN employees e ON e.employee_id = ip.employee_id
                   UNION ALL
                   SELECT 'Model evaluation' AS prediction_source, p.prediction_id,
                          p.employee_id, e.first_name || ' ' || e.last_name AS employee_name,
                          p.predicted_prob, p.predicted_label, p.true_label, p.created_at
                   FROM predictions p
                   JOIN employees e ON e.employee_id = p.employee_id
               )
               SELECT * FROM prediction_history"""
    conditions, params = [], []
    if start_date is not None:
        conditions.append("date(created_at) >= date(?)")
        params.append(start_date)
    if end_date is not None:
        conditions.append("date(created_at) <= date(?)")
        params.append(end_date)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY created_at DESC, prediction_source, prediction_id DESC"
    with get_connection(db_path) as connection:
        return pd.read_sql_query(query, connection, params=params)

def get_model_runs(start_ts=None,end_ts=None,db_path=DB_PATH):
    query="SELECT * FROM model_runs"; params=[]
    if start_ts and end_ts: query += " WHERE run_timestamp BETWEEN ? AND ?"; params=[start_ts,end_ts]
    with get_connection(db_path) as connection: return pd.read_sql_query(query + " ORDER BY run_timestamp DESC", connection, params=params)

def get_predictions_by_run(run_id,db_path=DB_PATH):
    with get_connection(db_path) as connection: return pd.read_sql_query("SELECT * FROM predictions WHERE run_id=? ORDER BY prediction_id", connection, params=[run_id])