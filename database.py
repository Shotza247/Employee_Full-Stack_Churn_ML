# Databricks notebook source
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "employees.db"

SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    employee_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    age INTEGER NOT NULL,
    department TEXT NOT NULL,
    contract_type TEXT NOT NULL,
    churn INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS salaries (
    salary_id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER NOT NULL UNIQUE,
    annual_income REAL NOT NULL,
    FOREIGN KEY(employee_id) REFERENCES employees(employee_id)
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER NOT NULL UNIQUE,
    job_title TEXT NOT NULL,
    tenure_months INTEGER NOT NULL,
    FOREIGN KEY(employee_id) REFERENCES employees(employee_id)
);
"""

SAMPLE_EMPLOYEES = [
    ("Aaliyah", "Smith", 29, "Marketing", "permanent", 72000.0, "Analyst", 18, 0),
    ("Noah", "Johnson", 42, "Sales", "permanent", 92000.0, "Account Executive", 72, 0),
    ("Mia", "Williams", 34, "Engineering", "contract", 104000.0, "Software Engineer", 36, 0),
    ("James", "Brown", 51, "HR", "permanent", 78000.0, "HR Business Partner", 96, 0),
    ("Emma", "Jones", 27, "Support", "contract", 61000.0, "Support Specialist", 12, 1),
    ("Liam", "Garcia", 38, "Operations", "permanent", 88000.0, "Operations Manager", 60, 0),
    ("Olivia", "Miller", 31, "Finance", "permanent", 91000.0, "Financial Analyst", 30, 0),
    ("Ethan", "Davis", 45, "Sales", "permanent", 99000.0, "Regional Manager", 84, 0),
    ("Ava", "Rodriguez", 26, "Marketing", "contract", 64000.0, "Content Creator", 14, 1),
    ("Sophia", "Martinez", 41, "Engineering", "permanent", 115000.0, "Team Lead", 78, 0),
    ("Jackson", "Hernandez", 53, "Support", "permanent", 72000.0, "Support Manager", 108, 0),
    ("Isabella", "Lopez", 30, "Finance", "contract", 78000.0, "Payroll Specialist", 20, 0),
    ("Lucas", "Gonzalez", 36, "Engineering", "permanent", 108000.0, "DevOps Engineer", 48, 0),
    ("Charlotte", "Wilson", 39, "HR", "permanent", 82000.0, "Recruiter", 52, 0),
    ("Mason", "Anderson", 28, "Sales", "contract", 65000.0, "Sales Associate", 16, 1),
    ("Amelia", "Thomas", 47, "Operations", "permanent", 93000.0, "Logistics Specialist", 70, 0),
    ("Logan", "Taylor", 33, "Engineering", "permanent", 102000.0, "QA Engineer", 40, 0),
    ("Harper", "Moore", 24, "Marketing", "contract", 58000.0, "Social Media Coordinator", 10, 1),
    ("Elijah", "Jackson", 50, "Finance", "permanent", 98000.0, "Controller", 90, 0),
    ("Evelyn", "Martin", 32, "Operations", "permanent", 85000.0, "Supply Chain Analyst", 35, 0),
    ("William", "Lee", 44, "Engineering", "permanent", 118000.0, "Senior Engineer", 75, 0),
    ("Abigail", "Perez", 37, "HR", "contract", 69000.0, "HR Coordinator", 22, 1),
    ("Benjamin", "Thompson", 29, "Sales", "permanent", 86000.0, "Sales Engineer", 28, 0),
    ("Emily", "White", 40, "Marketing", "permanent", 94000.0, "Marketing Manager", 66, 0),
    ("Henry", "Harris", 35, "Engineering", "contract", 98000.0, "Backend Developer", 44, 0),
    ("Eleanor", "Sanchez", 31, "Finance", "permanent", 89000.0, "Accounting Manager", 38, 0),
]


def init_db():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.executescript(SQL_SCHEMA)
    connection.commit()
    return connection


def populate_sample_data():
    conn = init_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM employees")
    count = cursor.fetchone()[0]
    if count > 0:
        conn.close()
        return

    for first_name, last_name, age, department, contract_type, annual_income, job_title, tenure_months, churn in SAMPLE_EMPLOYEES:
        cursor.execute(
            "INSERT INTO employees (first_name, last_name, age, department, contract_type, churn) VALUES (?, ?, ?, ?, ?, ?)",
            (first_name, last_name, age, department, contract_type, churn),
        )
        employee_id = cursor.lastrowid
        cursor.execute(
            "INSERT INTO salaries (employee_id, annual_income) VALUES (?, ?)",
            (employee_id, annual_income),
        )
        cursor.execute(
            "INSERT INTO jobs (employee_id, job_title, tenure_months) VALUES (?, ?, ?)",
            (employee_id, job_title, tenure_months),
        )

    conn.commit()
    conn.close()


def get_employee_features(employee_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            e.first_name,
            e.last_name,
            e.age,
            e.department,
            e.contract_type,
            e.churn,
            s.annual_income,
            j.job_title,
            j.tenure_months
        FROM employees e
        JOIN salaries s ON s.employee_id = e.employee_id
        JOIN jobs j ON j.employee_id = e.employee_id
        WHERE e.employee_id = ?
        """,
        (employee_id,),
    )
    row = cursor.fetchone()
    conn.close()
    return row


def get_all_employees():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT e.employee_id, e.first_name, e.last_name, e.age, e.department, e.contract_type, s.annual_income, j.job_title, j.tenure_months FROM employees e JOIN salaries s ON s.employee_id = e.employee_id JOIN jobs j ON j.employee_id = e.employee_id ORDER BY e.employee_id"
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_training_data():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT e.age, s.annual_income AS income, j.tenure_months, e.churn FROM employees e JOIN salaries s ON s.employee_id = e.employee_id JOIN jobs j ON j.employee_id = e.employee_id"
    )
    rows = cursor.fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=["age", "income", "tenure_months", "churn"])

#accuracy
#precision
#recall
#f1
#roc_auc
#employee_id
#predicted_prob
#predicted_label
#true_label
#created_at



def insert_employee(
    first_name: str,
    last_name: str,
    age: int,
    department: str,
    contract_type: str,
    annual_income: float,
    job_title: str,
    tenure_months: int,
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO employees (first_name, last_name, age, department, contract_type) VALUES (?, ?, ?, ?, ?)",
        (first_name, last_name, age, department, contract_type),
    )
    employee_id = cursor.lastrowid
    cursor.execute(
        "INSERT INTO salaries (employee_id, annual_income) VALUES (?, ?)",
        (employee_id, annual_income),
    )
    cursor.execute(
        "INSERT INTO jobs (employee_id, job_title, tenure_months) VALUES (?, ?, ?)",
        (employee_id, job_title, tenure_months),
    )
    conn.commit()
    conn.close()
    return employee_id
