# Databricks notebook source
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from sklearn.metrics import auc, confusion_matrix, roc_curve

from churn_app.database import (
    get_all_employees,
    get_employee_features,
    get_model_runs,
    get_predictions_by_run,
    get_training_data,
    import_employee_dataframe,
    init_db,
    insert_employee,
    populate_sample_data,
)
from churn_app.prediction import load_model, predict_employee

REQUIRED_COLUMNS = {"first_name", "last_name", "age", "department", "contract_type", "annual_income", "job_title", "tenure_months"}
VALID_DEPARTMENTS = {"Marketing", "Sales", "Engineering", "HR", "Support", "Operations", "Finance"}
VALID_CONTRACTS = {"permanent", "contract"}

init_db()
populate_sample_data()


@st.cache_resource
def get_prediction_model():
    return load_model()


def predict_employee_locally(employee_id: int):
    row = get_employee_features(int(employee_id))
    if row is None:
        return {"error": f"Employee not found: {employee_id}"}
    return predict_employee(get_prediction_model(), row)


def calculate_risk_band(probability: float) -> str:
    if probability >= 0.7:
        return "high"
    if probability >= 0.45:
        return "medium"
    return "low"


def quality_report(data: pd.DataFrame) -> dict[str, int | float]:
    numeric_columns = [column for column in ["age", "annual_income", "tenure_months"] if column in data]
    numeric_data = data[numeric_columns].apply(pd.to_numeric, errors="coerce") if numeric_columns else pd.DataFrame()
    missing_values = int(data.isna().sum().sum())
    duplicate_rows = int(data.duplicated().sum())
    invalid_values = 0
    if "age" in numeric_data:
        invalid_values += int(((numeric_data["age"] < 18) | (numeric_data["age"] > 99)).fillna(False).sum())
    if "annual_income" in numeric_data:
        invalid_values += int((numeric_data["annual_income"] < 0).fillna(False).sum())
    if "tenure_months" in numeric_data:
        invalid_values += int(((numeric_data["tenure_months"] < 0) | (numeric_data["tenure_months"] > 240)).fillna(False).sum())
    if "department" in data:
        invalid_values += int((~data["department"].isin(VALID_DEPARTMENTS)).fillna(True).sum())
    if "contract_type" in data:
        invalid_values += int((~data["contract_type"].isin(VALID_CONTRACTS)).fillna(True).sum())
    if "churn" in data:
        churn_values = pd.to_numeric(data["churn"], errors="coerce")
        invalid_values += int((~churn_values.isin([0, 1])).sum())
    return {"rows": int(len(data)), "columns": int(len(data.columns)), "duplicates": duplicate_rows, "missing": missing_values, "invalid": invalid_values, "numeric_mean": float(numeric_data.mean().mean()) if not numeric_data.empty else 0.0}


def show_quality_cards(report: dict[str, int | float], location=st):
    cards = location.columns(3)
    cards[0].metric("Rows", f"{report['rows']:,}")
    cards[1].metric("Duplicates", f"{report['duplicates']:,}")
    cards[2].metric("Missing values", f"{report['missing']:,}")
    cards = location.columns(3)
    cards[0].metric("Invalid values", f"{report['invalid']:,}")
    cards[1].metric("Columns", f"{report['columns']:,}")
    cards[2].metric("Numeric mean", f"{report['numeric_mean']:,.1f}")


def show_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, title: str):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual non-churner", "Actual churner"],
        columns=["Predicted non-churner", "Predicted churner"],
    )
    st.subheader(title)
    matrix_col, counts_col = st.columns([3, 2], gap="large")
    with matrix_col:
        matrix_figure, matrix_axis = plt.subplots(figsize=(5.0, 2.9))
        image = matrix_axis.imshow(matrix, cmap="Blues")
        matrix_axis.set_xticks([0, 1], ["Non-churner", "Churner"])
        matrix_axis.set_yticks([0, 1], ["Non-churner", "Churner"])
        matrix_axis.set_xlabel("Predicted class")
        matrix_axis.set_ylabel("Actual class")
        matrix_axis.set_title("Confusion matrix")
        for row_index in range(2):
            for column_index in range(2):
                matrix_axis.text(column_index, row_index, str(matrix[row_index, column_index]), ha="center", va="center", color="white" if matrix[row_index, column_index] > matrix.max() / 2 else "#1f2937", fontsize=13, fontweight="bold")
        matrix_figure.colorbar(image, ax=matrix_axis, fraction=0.046, pad=0.04)
        matrix_figure.tight_layout()
        st.pyplot(matrix_figure, use_container_width=True)
        plt.close(matrix_figure)
    with counts_col:
        st.dataframe(matrix_df, use_container_width=True)
    return matrix


st.set_page_config(page_title="Employee Churn Intelligence", layout="wide")
st.title("Employee Churn Intelligence")
st.caption("Onboard employees, inspect data quality, and monitor churn model behavior.")

with st.sidebar:
    st.header("Batch import")
    st.caption("Upload the noisy experiment CSV to simulate drift and anomalies.")
    uploaded_file = st.file_uploader("Choose a CSV dataset", type="csv")
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            missing_columns = REQUIRED_COLUMNS - set(uploaded_df.columns)
            if missing_columns:
                st.error(f"Missing columns: {', '.join(sorted(missing_columns))}")
            else:
                st.subheader("Data quality tracker")
                show_quality_cards(quality_report(uploaded_df), st.sidebar)
                st.caption("Preview")
                st.dataframe(uploaded_df.head(5), hide_index=True, use_container_width=True)
                if st.button("Import valid rows", type="primary", use_container_width=True):
                    accepted, rejected = import_employee_dataframe(uploaded_df)
                    st.session_state["last_import"] = {
                        "accepted": accepted,
                        "rejected": rejected,
                        "predictions": None,
                    }
                    st.rerun()
        except (pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(f"Could not read CSV: {error}")
    else:
        st.subheader("Database quality")
        show_quality_cards(quality_report(get_training_data()), st.sidebar)
    last_import = st.session_state.get("last_import")
    if last_import:
        st.success(f"Imported {len(last_import['accepted'])} rows")
        if not last_import["accepted"].empty:
            if st.button("Predict imported employees", use_container_width=True):
                prediction_rows = []
                try:
                    with st.spinner("Predicting churn for imported employees..."):
                        for employee_id in last_import["accepted"]["employee_id"]:
                            result = predict_employee_locally(int(employee_id))
                            prediction_rows.append({"employee_id": int(employee_id), **result})
                except FileNotFoundError as error:
                    st.error(f"Cannot predict until a trained model is available: {error}")
                else:
                    last_import["predictions"] = pd.DataFrame(prediction_rows)
                    st.session_state["last_import"] = last_import
        if not last_import["rejected"].empty:
            st.warning(f"Rejected {len(last_import['rejected'])} rows")
            with st.expander("Rejected rows"):
                st.dataframe(last_import["rejected"], hide_index=True, use_container_width=True)

last_import = st.session_state.get("last_import")
if last_import and last_import.get("predictions") is not None:
    st.subheader("Predictions for imported employees")
    imported_predictions = last_import["predictions"]
    if "error" in imported_predictions.columns and imported_predictions["error"].notna().any():
        st.warning("Some imported employees could not be scored.")
    st.dataframe(imported_predictions, hide_index=True, use_container_width=True)

with st.expander("Add a single employee"):
    with st.form("employee_form"):
        first_name = st.text_input("First name")
        last_name = st.text_input("Last name")
        age = st.number_input("Age", min_value=18, max_value=99, value=30)
        department = st.selectbox("Department", sorted(VALID_DEPARTMENTS))
        contract_type = st.selectbox("Contract type", sorted(VALID_CONTRACTS))
        annual_income = st.number_input("Annual income", min_value=0.0, value=52000.0, step=1000.0, format="%.2f")
        job_title = st.text_input("Job title")
        tenure_months = st.number_input("Tenure months", min_value=0, max_value=240, value=12)
        submit_button = st.form_submit_button("Insert employee")
        if submit_button:
            if not first_name.strip() or not last_name.strip() or not job_title.strip():
                st.error("First name, last name, and job title are required.")
            else:
                new_id = insert_employee(first_name.strip(), last_name.strip(), int(age), department, contract_type, float(annual_income), job_title.strip(), int(tenure_months))
                if new_id is None:
                    st.error("Employee was inserted, but no employee ID was returned; prediction is unavailable.")
                else:
                    st.success(f"Inserted employee with ID {new_id}.")
                    try:
                        result = predict_employee_locally(new_id)
                    except FileNotFoundError as error:
                        st.warning(f"Prediction unavailable: {error}")
                    else:
                        if result.get("error"):
                            st.warning(result["error"])
                        else:
                            st.write(f"Churn probability: {float(result['churn_probability']):.1%} | Risk: **{result['risk_band']}**")

employees = get_all_employees()
if employees:
    employee_df = pd.DataFrame(employees, columns=["employee_id", "first_name", "last_name", "age", "department", "contract_type", "annual_income", "job_title", "tenure_months"])
    employee_df["employee_name"] = employee_df["first_name"] + " " + employee_df["last_name"]
    st.header("Employee monitoring")
    with st.container(border=True):
        st.subheader("Employee roster")
        st.dataframe(employee_df[["employee_id", "employee_name", "age", "department", "contract_type", "job_title", "annual_income", "tenure_months"]], hide_index=True, use_container_width=True, height=320)
    st.subheader("Batch churn predictions")
    probabilities, bands = [], []
    prediction_errors = []
    try:
        with st.spinner("Scoring employees with the churn model..."):
            for employee_id in employee_df["employee_id"]:
                result = predict_employee_locally(int(employee_id))
                if result.get("error"):
                    prediction_errors.append(f"{employee_id}: {result['error']}")
                probability = None if result.get("error") else float(result["churn_probability"])
                probabilities.append(probability)
                bands.append(None if probability is None else result.get("risk_band", calculate_risk_band(probability)))
    except FileNotFoundError as error:
        st.error(f"Cannot score employees until a trained model is available: {error}")
        probabilities = [None] * len(employee_df)
        bands = [None] * len(employee_df)
    if prediction_errors:
        st.warning(f"Some employees could not be scored: {'; '.join(prediction_errors)}")
    employee_df["churn_probability"] = probabilities
    employee_df["risk_band"] = bands
    with st.container(border=True):
        st.dataframe(employee_df[["employee_id", "employee_name", "department", "job_title", "annual_income", "tenure_months", "churn_probability", "risk_band"]].sort_values("churn_probability", ascending=False, na_position="last"), hide_index=True, use_container_width=True, height=360)
    risk_counts = employee_df["risk_band"].value_counts().reindex(["high", "medium", "low"], fill_value=0)
    risk_chart = pd.DataFrame({"Employees": risk_counts})
    risk_chart.index = ["High risk", "Medium risk", "Low risk"]
    risk_chart["Share (%)"] = (risk_chart["Employees"] / risk_chart["Employees"].sum() * 100).round(1) if risk_chart["Employees"].sum() else 0.0
    st.subheader("Churn risk distribution")
    risk_col, share_col = st.columns([3, 2], gap="large")
    with risk_col:
        with st.container(border=True):
            st.bar_chart(risk_chart["Employees"], height=260, color="#d95f02")
    with share_col:
        with st.container(border=True):
            st.dataframe(risk_chart, hide_index=False, use_container_width=True, height=260)
    live_labels = get_training_data()[["employee_id", "churn"]].rename(columns={"churn": "true_label"})
    live_predictions = employee_df[["employee_id", "churn_probability"]].merge(live_labels, on="employee_id")
    live_predictions = live_predictions.dropna(subset=["churn_probability", "true_label"])
    if live_predictions["true_label"].nunique() > 1:
        live_predictions["predicted_label"] = (live_predictions["churn_probability"] >= 0.5).astype(int)
        show_confusion_matrix(live_predictions["true_label"], live_predictions["predicted_label"], "Live batch confusion matrix")
        st.caption(f"Based on {len(live_predictions)} live predictions with known outcomes.")

st.divider()
st.header("Model dashboard")
with st.container(border=True):
    filter_option = st.selectbox("Date filter", ["All", "Today", "Yesterday", "This week", "Last week", "Custom range"])

def compute_range(option):
    today = datetime.now().date()
    if option == "Today":
        return datetime.combine(today, datetime.min.time()).isoformat(), datetime.combine(today, datetime.max.time()).isoformat()
    if option == "Yesterday":
        day = today - timedelta(days=1)
        return datetime.combine(day, datetime.min.time()).isoformat(), datetime.combine(day, datetime.max.time()).isoformat()
    if option == "This week":
        return datetime.combine(today - timedelta(days=today.weekday()), datetime.min.time()).isoformat(), datetime.combine(today, datetime.max.time()).isoformat()
    if option == "Last week":
        start = today - timedelta(days=today.weekday() + 7)
        end = today - timedelta(days=today.weekday() + 1)
        return datetime.combine(start, datetime.min.time()).isoformat(), datetime.combine(end, datetime.max.time()).isoformat()
    return None, None

start_ts, end_ts = compute_range(filter_option)
if filter_option == "Custom range":
    start_date, end_date = st.columns(2)
    with start_date:
        custom_start = st.date_input("Start date", value=datetime.now().date())
    with end_date:
        custom_end = st.date_input("End date", value=datetime.now().date())
    start_ts = datetime.combine(custom_start, datetime.min.time()).isoformat()
    end_ts = datetime.combine(custom_end, datetime.max.time()).isoformat()

runs_df = get_model_runs(start_ts, end_ts)
if runs_df.empty:
    st.info("No model runs found. Run `py -m churn_app.train_and_evaluate` first.")
else:
    latest = runs_df.iloc[0]
    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    with st.container(border=True):
        metric_cards = st.columns(len(metric_columns), gap="medium")
        for card, metric in zip(metric_cards, metric_columns):
            card.metric(metric.upper(), f"{float(latest[metric]):.1%}")
    st.subheader("Evaluation history")
    st.dataframe(runs_df, hide_index=True, use_container_width=True, height=260)
    selected_run = st.selectbox("Inspect run", runs_df["run_id"].astype(str).tolist())
    predictions = get_predictions_by_run(int(selected_run))
    if not predictions.empty:
        valid = predictions.dropna(subset=["predicted_label", "true_label", "predicted_prob"]).astype({"predicted_label": int, "true_label": int})
        st.dataframe(predictions[["employee_id", "predicted_prob", "predicted_label", "true_label", "created_at"]].head(200), hide_index=True, use_container_width=True)
        if not valid.empty:
            actual_counts = valid["true_label"].value_counts().reindex([0, 1], fill_value=0)
            predicted_counts = valid["predicted_label"].value_counts().reindex([0, 1], fill_value=0)
            classification_df = pd.DataFrame({"Non-churners": [actual_counts[0], predicted_counts[0]], "Churners": [actual_counts[1], predicted_counts[1]]}, index=["Actual", "Predicted"])
            roc_auc_value = float(latest["roc_auc"])
            try:
                false_positive_rate, true_positive_rate, _ = roc_curve(valid["true_label"], valid["predicted_prob"])
                stored_auc = float(auc(false_positive_rate, true_positive_rate))
            except ValueError:
                false_positive_rate, true_positive_rate, stored_auc = [], [], roc_auc_value
            st.subheader("Classification and discrimination")
            with st.container(border=True):
                chart_col, roc_col = st.columns(2, gap="large")
                with chart_col:
                    st.caption("Actual versus predicted employee classes")
                    st.bar_chart(classification_df, height=280)
                    st.caption("Class counts: 0 = non-churner, 1 = churner")
                with roc_col:
                    st.metric("AUC", f"{roc_auc_value:.1%}")
                    roc_figure, roc_axis = plt.subplots(figsize=(4.8, 3.1))
                    if len(false_positive_rate):
                        roc_axis.plot(false_positive_rate, true_positive_rate, color="#d95f02", linewidth=2, label=f"ROC (AUC {stored_auc:.2f})")
                    roc_axis.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
                    roc_axis.set_xlabel("False positive rate")
                    roc_axis.set_ylabel("True positive rate")
                    roc_axis.set_title("Stored prediction ROC")
                    roc_axis.legend(loc="lower right", fontsize=8)
                    roc_figure.tight_layout()
                    st.pyplot(roc_figure, use_container_width=True)
                    plt.close(roc_figure)
            show_confusion_matrix(valid["true_label"], valid["predicted_label"], "Evaluation confusion matrix")
            tp = int(((valid["predicted_label"] == 1) & (valid["true_label"] == 1)).sum())
            tn = int(((valid["predicted_label"] == 0) & (valid["true_label"] == 0)).sum())
            fp = int(((valid["predicted_label"] == 1) & (valid["true_label"] == 0)).sum())
            fn = int(((valid["predicted_label"] == 0) & (valid["true_label"] == 1)).sum())
            st.caption(f"TP {tp} | TN {tn} | FP {fp} | FN {fn}")
            st.subheader("Prediction probability distribution")
            with st.container(border=True):
                figure, axis = plt.subplots(figsize=(6.0, 2.6))
                axis.hist(valid["predicted_prob"], bins=12, color="#2f6fed", alpha=0.8, edgecolor="white")
                axis.set_xlabel("Churn probability")
                axis.set_ylabel("Employees")
                axis.set_title("Prediction distribution")
                figure.tight_layout()
                st.pyplot(figure, use_container_width=True)
                plt.close(figure)
