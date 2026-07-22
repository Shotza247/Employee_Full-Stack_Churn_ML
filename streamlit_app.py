# Databricks notebook source
import streamlit as st
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timedelta

from database import init_db, populate_sample_data, get_all_employees, insert_employee, get_employee_features

API_URL = "http://127.0.0.1:8000/predict"

init_db()
populate_sample_data()


def call_predict_api(employee_id: int, timeout: float = 5.0):
    try:
        resp = requests.post(API_URL, json={"employee_id": int(employee_id)}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def calculate_risk_band(probability: float) -> str:
    if probability >= 0.7:
        return "high"
    if probability >= 0.45:
        return "medium"
    return "low"


st.set_page_config(page_title="Employee Churn Onboarding", layout="wide")
st.title("Employee Churn Onboarding + Prediction")

st.markdown("""Use this form to add new employee records into the SQLite database.
The app will also show current employees and apply churn scoring when the prediction API is available.""")

with st.expander("Add a new employee"):
    with st.form("employee_form"):
        first_name = st.text_input("First name")
        last_name = st.text_input("Last name")
        age = st.number_input("Age", min_value=18, max_value=99, value=30)
        department = st.selectbox(
            "Department",
            ["Marketing", "Sales", "Engineering", "HR", "Support", "Operations", "Finance"],
        )
        contract_type = st.selectbox("Contract type", ["permanent", "contract"])
        annual_income = st.number_input("Annual income", min_value=0.0, value=52000.0, step=1000.0, format="%.2f")
        job_title = st.text_input("Job title")
        tenure_months = st.number_input("Tenure months", min_value=0, max_value=240, value=12)
        submit_button = st.form_submit_button("Insert employee")

        if submit_button:
            if not first_name or not last_name or not job_title:
                st.error("Please fill in first name, last name, and job title.")
            else:
                new_id = insert_employee(
                    first_name,
                    last_name,
                    age,
                    department,
                    contract_type,
                    float(annual_income),
                    job_title,
                    int(tenure_months),
                )
                st.success(f"Inserted employee with ID {new_id}.")
                api_result = call_predict_api(new_id)
                if api_result.get("error"):
                    st.warning(f"Prediction API error: {api_result.get('error')}")
                else:
                    probability = float(api_result.get("churn_probability", 0.0))
                    risk_band = api_result.get("risk_band", calculate_risk_band(probability))
                    st.write(f"Churn probability: {probability:.3f}")
                    st.write(f"Risk band: **{risk_band}**")

st.write("---")

employees = get_all_employees()

if not employees:
    st.info("No employee records found. Add a new employee above.")
else:
    employee_df = pd.DataFrame(
        employees,
        columns=[
            "employee_id",
            "first_name",
            "last_name",
            "age",
            "department",
            "contract_type",
            "annual_income",
            "job_title",
            "tenure_months",
        ],
    )
    employee_df["employee_name"] = employee_df["first_name"] + " " + employee_df["last_name"]

    st.subheader("Current employee roster")
    st.dataframe(
        employee_df[
            [
                "employee_id",
                "employee_name",
                "age",
                "department",
                "contract_type",
                "job_title",
                "annual_income",
                "tenure_months",
            ]
        ],
        use_container_width=True,
    )

    st.subheader("Batch churn predictions")
    probs = []
    bands = []
    with st.spinner("Requesting predictions from API..."):
        for eid in employee_df["employee_id"]:
            res = call_predict_api(eid)
            if res.get("error"):
                probs.append(None)
                bands.append(None)
            else:
                probs.append(float(res.get("churn_probability", 0.0)))
                bands.append(res.get("risk_band", calculate_risk_band(float(res.get("churn_probability", 0.0)))))

    employee_df["churn_probability"] = probs
    employee_df["risk_band"] = bands

    st.dataframe(
        employee_df[
            [
                "employee_id",
                "employee_name",
                "age",
                "department",
                "job_title",
                "annual_income",
                "tenure_months",
                "churn_probability",
                "risk_band",
            ]
        ].sort_values("churn_probability", ascending=False),
        use_container_width=True,
    )

# --- Model Dashboard ---
st.markdown("---")
st.header("Model Dashboard")

filter_option = st.selectbox("Date filter for runs", ["All", "Today", "Yesterday", "This week", "Last week", "Custom range"])

def compute_range(option):
    today = datetime.now().date()
    if option == "Today":
        start = datetime.combine(today, datetime.min.time())
        end = datetime.combine(today, datetime.max.time())
    elif option == "Yesterday":
        y = today - timedelta(days=1)
        start = datetime.combine(y, datetime.min.time())
        end = datetime.combine(y, datetime.max.time())
    elif option == "This week":
        start = datetime.combine(today - timedelta(days=today.weekday()), datetime.min.time())
        end = datetime.combine(today, datetime.max.time())
    elif option == "Last week":
        start = datetime.combine(today - timedelta(days=today.weekday() + 7), datetime.min.time())
        end = datetime.combine(today - timedelta(days=today.weekday() + 1), datetime.max.time())
    else:
        return None, None
    return start.isoformat(), end.isoformat()

start_ts = None
end_ts = None
if filter_option == "Custom range":
    col1, col2 = st.columns(2)
    with col1:
        sdate = st.date_input("Start date", value=datetime.now().date())
    with col2:
        edate = st.date_input("End date", value=datetime.now().date())
    start_ts = datetime.combine(sdate, datetime.min.time()).isoformat()
    end_ts = datetime.combine(edate, datetime.max.time()).isoformat()
elif filter_option != "All":
    start_ts, end_ts = compute_range(filter_option)

runs_df = get_model_runs(start_ts, end_ts)

if runs_df.empty:
    st.info("No model runs found for selected period. Run training first (use train_and_evaluate.py).")
else:
    # convert run_timestamp
    runs_df["run_timestamp"] = pd.to_datetime(runs_df["run_timestamp"])
    st.subheader("Model runs")
    st.dataframe(runs_df.sort_values("run_timestamp", ascending=False), use_container_width=True)

    # Metrics trend
    metrics_to_plot = st.multiselect("Metrics to plot", ["f1", "accuracy", "precision", "recall", "roc_auc"], default=["f1"])
    if metrics_to_plot:
        chart_df = runs_df.set_index("run_timestamp")[metrics_to_plot]
        st.line_chart(chart_df)

    st.subheader("Per-run details")
    run_options = runs_df["run_id"].astype(str).tolist()
    selected_run = st.selectbox("Select run", run_options)
    if selected_run:
        run_id = int(selected_run)
        preds = get_predictions_by_run(run_id)
        if preds.empty:
            st.info("No predictions stored for this run.")
        else:
            # show summary metrics from stored predictions
            preds_display = preds[["employee_id", "predicted_prob", "predicted_label", "true_label", "created_at"]]
            st.dataframe(preds_display.head(200), use_container_width=True)
            # quick evaluation
            valid = preds.dropna(subset=["predicted_label", "true_label"]).astype({"predicted_label": int, "true_label": int})
            if not valid.empty:
                tp = int(((valid["predicted_label"] == 1) & (valid["true_label"] == 1)).sum())
                tn = int(((valid["predicted_label"] == 0) & (valid["true_label"] == 0)).sum())
                fp = int(((valid["predicted_label"] == 1) & (valid["true_label"] == 0)).sum())
                fn = int(((valid["predicted_label"] == 0) & (valid["true_label"] == 1)).sum())
                st.write(f"TP: {tp}  TN: {tn}  FP: {fp}  FN: {fn}")
                # basic rates
                try:
                    acc = (tp + tn) / (tp + tn + fp + fn)
                except ZeroDivisionError:
                    acc = None
                st.write(f"Accuracy: {acc}")

                # show distribution of predicted probabilities
                st.subheader("Predicted probability distribution")
                import matplotlib.pyplot as plt
                fig = pd.DataFrame({"prob": valid["predicted_prob"]}).plot(kind="hist", bins=20).figure
                st.pyplot(fig)

st.markdown("---")
st.caption("Model runs and predictions are stored in the local SQLite database (model_runs, predictions). Use `train_and_evaluate.py` to run training non-interactively.")