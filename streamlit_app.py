# Databricks notebook source
from datetime import datetime, timedelta
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import auc, confusion_matrix, roc_curve
import streamlit as st

from churn_app.config import MODEL_PATH
from churn_app.database import (
    get_all_employees,
    get_employee_features,
    get_model_runs,
    get_prediction_history,
    get_predictions_by_run,
    get_training_data,
    import_employee_dataframe,
    init_db,
    insert_employee,
    populate_sample_data,
    save_inference_predictions,
)
from churn_app.faker_producer import FakerProducer
from churn_app.feature_engineering import impute_and_engineer_features
from churn_app.prediction import load_model, predict_employee
from churn_app.training import (
    compute_feature_correlations,
    get_feature_importances,
    train_model,
)

REQUIRED_COLUMNS = {"first_name", "last_name", "age", "department", "contract_type", "annual_income", "job_title", "tenure_months"}
VALID_DEPARTMENTS = {"Marketing", "Sales", "Engineering", "HR", "Support", "Operations", "Finance"}
VALID_CONTRACTS = {"permanent", "contract"}

init_db()
populate_sample_data()


@st.cache_resource
def _load_cached_prediction_model(model_path: str, modified_ns: int, size: int):
    return load_model(model_path)


def get_prediction_model():
    artifact = MODEL_PATH.stat()
    return _load_cached_prediction_model(
        str(MODEL_PATH),
        artifact.st_mtime_ns,
        artifact.st_size,
    )


def predict_employee_locally(employee_id: int):
    row = get_employee_features(int(employee_id))
    if row is None:
        return {"error": f"Employee not found: {employee_id}"}
    return predict_employee(get_prediction_model(), row)


def predict_employee_via_api(employee_id: int, api_url: str):
    url = f"{api_url.rstrip('/')}/predict"
    try:
        response = requests.post(url, json={"employee_id": int(employee_id)}, timeout=8)
        if response.status_code == 200:
            return response.json()
        return {"error": f"API Error ({response.status_code}): {response.text}"}
    except Exception as exc:
        return {"error": f"Failed to connect to API ({url}): {exc}"}


def calculate_risk_band(probability: float) -> str:
    if probability >= 0.7:
        return "high"
    if probability >= 0.45:
        return "medium"
    return "low"


def quality_report(data: pd.DataFrame) -> dict[str, int | float]:
    numeric_columns = [column for column in ["age", "annual_income", "income", "tenure_months"] if column in data]
    numeric_data = data[numeric_columns].apply(pd.to_numeric, errors="coerce") if numeric_columns else pd.DataFrame()
    missing_values = int(data.isna().sum().sum())
    duplicate_rows = int(data.duplicated().sum())
    invalid_values = 0
    if "age" in numeric_data:
        invalid_values += int(((numeric_data["age"] < 18) | (numeric_data["age"] > 99)).fillna(False).sum())
    if "annual_income" in numeric_data:
        invalid_values += int((numeric_data["annual_income"] < 0).fillna(False).sum())
    if "income" in numeric_data:
        invalid_values += int((numeric_data["income"] < 0).fillna(False).sum())
    if "tenure_months" in numeric_data:
        invalid_values += int(((numeric_data["tenure_months"] < 0) | (numeric_data["tenure_months"] > 240)).fillna(False).sum())
    if "department" in data:
        invalid_values += int((~data["department"].isin(VALID_DEPARTMENTS)).fillna(True).sum())
    if "contract_type" in data:
        invalid_values += int((~data["contract_type"].isin(VALID_CONTRACTS)).fillna(True).sum())
    if "churn" in data:
        churn_values = pd.to_numeric(data["churn"], errors="coerce")
        invalid_values += int((~churn_values.isin([0, 1])).sum())
    return {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "duplicates": duplicate_rows,
        "missing": missing_values,
        "invalid": invalid_values,
        "numeric_mean": float(numeric_data.mean().mean()) if not numeric_data.empty else 0.0,
    }


def show_quality_cards(report: dict[str, int | float], location=st):
    cards = location.columns(3)
    cards[0].metric("Rows", f"{report['rows']:,}")
    cards[1].metric("Duplicates", f"{report['duplicates']:,}")
    cards[2].metric("Missing", f"{report['missing']:,}")
    cards2 = location.columns(3)
    cards2[0].metric("Invalid", f"{report['invalid']:,}")
    cards2[1].metric("Columns", f"{report['columns']:,}")
    cards2[2].metric("Num Mean", f"{report['numeric_mean']:,.0f}")


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
                matrix_axis.text(
                    column_index,
                    row_index,
                    str(matrix[row_index, column_index]),
                    ha="center",
                    va="center",
                    color="white" if matrix[row_index, column_index] > matrix.max() / 2 else "#1f2937",
                    fontsize=13,
                    fontweight="bold",
                )
        matrix_figure.colorbar(image, ax=matrix_axis, fraction=0.046, pad=0.04)
        matrix_figure.tight_layout()
        st.pyplot(matrix_figure, use_container_width=True)
        plt.close(matrix_figure)
    with counts_col:
        st.dataframe(matrix_df, use_container_width=True)
    return matrix


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, title: str = "Feature Correlation Heatmap"):
    if corr_matrix.empty:
        st.info("No correlation data available.")
        return
    fig, ax = plt.subplots(figsize=(max(6.5, len(corr_matrix.columns) * 0.72), max(4.8, len(corr_matrix.index) * 0.58)))
    cax = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr_matrix.index, fontsize=9)
    ax.set_title(title, fontsize=12, pad=12, fontweight="bold")

    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            if pd.notna(val):
                text_color = "white" if abs(val) > 0.45 else "#1e293b"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8, fontweight="bold")

    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# =========================================================
# Page Setup
# =========================================================
st.set_page_config(page_title="Employee Churn Intelligence", layout="wide", page_icon="🏢")
st.title("🏢 Employee Churn Intelligence")
st.caption("Interactive workforce analytics, scenario modeling, zero-null feature engineering, and churn intelligence.")

# =========================================================
# Persistent Sidebar: Ingestion & Simulation Studio
# =========================================================
with st.sidebar:
    st.header("⚙️ Settings & Ingestion")

    with st.expander("🌐 Cloud API / Vercel Backend", expanded=False):
        api_backend_url = st.text_input(
            "FastAPI URL",
            value=os.getenv("CHURN_API_URL", ""),
            placeholder="e.g. https://your-app.vercel.app",
            help="If set, predictions route through your deployed Vercel FastAPI backend.",
        )
        use_remote_api = st.checkbox("Use Remote API for Predictions", value=bool(api_backend_url))
        if api_backend_url and st.button("Test API Connection", use_container_width=True):
            try:
                res = requests.get(f"{api_backend_url.rstrip('/')}/health", timeout=5)
                if res.status_code == 200:
                    st.success("API connection active & healthy!")
                else:
                    st.warning(f"API returned status {res.status_code}")
            except Exception as e:
                st.error(f"Cannot reach API: {e}")

    # CSV Batch Import
    st.markdown("### 📥 Batch CSV Import")
    uploaded_file = st.file_uploader("Upload employee dataset", type="csv")
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            missing_columns = REQUIRED_COLUMNS - set(uploaded_df.columns)
            if missing_columns:
                st.error(f"Missing columns: {', '.join(sorted(missing_columns))}")
            else:
                with st.expander("Uploaded Data Quality", expanded=False):
                    show_quality_cards(quality_report(uploaded_df), st)
                st.caption("Preview first 3 rows:")
                st.dataframe(uploaded_df.head(3), hide_index=True, use_container_width=True)
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

    last_import = st.session_state.get("last_import")
    if last_import:
        st.success(f"Imported {len(last_import['accepted'])} rows")
        if not last_import["rejected"].empty:
            st.warning(f"Rejected {len(last_import['rejected'])} rows")
            with st.expander("Rejected rows summary", expanded=False):
                st.dataframe(last_import["rejected"], hide_index=True, use_container_width=True)

            # Python Faker Producer Section to repair unapproved / rejected records
            with st.container(border=True):
                st.markdown("#### 🪄 Faker Auto-Repair")
                st.caption("Synthesize valid identities for rejected rows.")
                if st.button("⚡ Repair Rejected Rows", use_container_width=True):
                    producer = FakerProducer()
                    repaired_df = producer.repair_rejected_rows(last_import["rejected"])
                    if not repaired_df.empty:
                        cols_to_import = [c for c in repaired_df.columns if c not in ["source_row", "reason", "faker_repairs"]]
                        import_payload = repaired_df[cols_to_import].copy()
                        new_accepted, new_rejected = import_employee_dataframe(import_payload)

                        combined_accepted = pd.concat([last_import["accepted"], new_accepted], ignore_index=True)
                        st.session_state["last_import"]["accepted"] = combined_accepted
                        st.session_state["last_import"]["rejected"] = new_rejected
                        st.session_state["repaired_preview"] = repaired_df[["first_name", "last_name", "department", "job_title", "faker_repairs"]]
                        st.success(f"Repaired and imported {len(new_accepted)} rows!")
                        st.rerun()

    # Synthetic Training Batch Producer
    with st.expander("🧪 Synthetic Batch Producer (Faker)"):
        st.caption("Generate synthetic employee records for stress-testing:")
        synth_count = st.slider("Records to generate", min_value=5, max_value=50, value=10, step=5)
        if st.button("Generate & Import Synthetic Batch", use_container_width=True):
            synth_producer = FakerProducer()
            synth_batch = synth_producer.generate_synthetic_batch(count=synth_count)
            synth_accepted, synth_rejected = import_employee_dataframe(synth_batch)
            st.success(f"Generated and imported {len(synth_accepted)} synthetic records.")
            st.rerun()

    # Add Single Employee Form
    with st.expander("➕ Add Single Employee"):
        with st.form("sidebar_employee_form"):
            first_name = st.text_input("First name")
            last_name = st.text_input("Last name")
            age = st.number_input("Age", min_value=18, max_value=99, value=30)
            department = st.selectbox("Department", sorted(VALID_DEPARTMENTS))
            contract_type = st.selectbox("Contract type", sorted(VALID_CONTRACTS))
            annual_income = st.number_input("Annual income (R / ZAR)", min_value=0.0, value=52000.0, step=1000.0, format="%.2f")
            job_title = st.text_input("Job title", value="Specialist")
            tenure_months = st.number_input("Tenure months", min_value=0, max_value=240, value=12)
            submit_button = st.form_submit_button("Insert & Score Employee")
            if submit_button:
                if not first_name.strip() or not last_name.strip() or not job_title.strip():
                    st.error("First name, last name, and job title are required.")
                else:
                    new_id = insert_employee(
                        first_name.strip(),
                        last_name.strip(),
                        int(age),
                        department,
                        contract_type,
                        float(annual_income),
                        job_title.strip(),
                        int(tenure_months),
                    )
                    if new_id is None:
                        st.error("Employee was inserted, but ID was not returned.")
                    else:
                        st.success(f"Inserted employee ID #{new_id}")
                        try:
                            result = predict_employee_locally(new_id)
                            if not result.get("error"):
                                save_inference_predictions(
                                    [{
                                        "employee_id": new_id,
                                        "predicted_prob": result["churn_probability"],
                                        "predicted_label": int(result["churn_probability"] >= 0.5),
                                    }]
                                )
                                st.info(f"Churn Risk: **{result['risk_band'].upper()}** ({result['churn_probability']:.1%})")
                        except Exception as ex:
                            st.warning(f"Prediction note: {ex}")
                        st.rerun()

    with st.expander("📊 SQLite Database Health", expanded=False):
        show_quality_cards(quality_report(get_training_data()), st)


# =========================================================
# Main Panel: 4-Tab Stakeholder Architecture
# =========================================================
tab_insights, tab_roster, tab_signals, tab_model = st.tabs([
    "📊 Executive HR Insights & Demographics",
    "👥 Workforce Roster & Predictions",
    "🔍 Feature Signals & Retention Drivers",
    "⚙️ Model Studio & Evaluation History",
])

# Load base workforce dataset
full_training_data = get_training_data()

# =========================================================
# TAB 1: Executive HR Insights & Demographics
# =========================================================
with tab_insights:
    st.header("Executive HR Insights & Workforce Demographics")
    st.caption("Real-time workforce demographics, turnover distribution, and department benchmarks.")

    if full_training_data.empty:
        st.info("No employee records found in the database. Use the sidebar to import data or generate synthetic rows.")
    else:
        # Dynamic Cohort Filters
        with st.container(border=True):
            st.markdown("#### 🎯 Interactive Cohort Drill-Down Filters")
            filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 3])

            with filter_col1:
                available_depts = sorted([str(d) for d in full_training_data["department"].dropna().unique()])
                selected_depts = st.multiselect("Department(s)", available_depts, default=available_depts)

            with filter_col2:
                available_contracts = sorted([str(c) for c in full_training_data["contract_type"].dropna().unique()])
                selected_contracts = st.multiselect("Contract Type(s)", available_contracts, default=available_contracts)

            with filter_col3:
                min_age = int(full_training_data["age"].min()) if full_training_data["age"].notna().any() else 18
                max_age = int(full_training_data["age"].max()) if full_training_data["age"].notna().any() else 70
                selected_age_range = st.slider("Age Range", min_value=min(18, min_age), max_value=max(70, max_age), value=(min(18, min_age), max(70, max_age)))

        # Apply Filters
        filtered_df = full_training_data.copy()
        if selected_depts:
            filtered_df = filtered_df[filtered_df["department"].isin(selected_depts)]
        if selected_contracts:
            filtered_df = filtered_df[filtered_df["contract_type"].isin(selected_contracts)]
        filtered_df = filtered_df[(filtered_df["age"] >= selected_age_range[0]) & (filtered_df["age"] <= selected_age_range[1])]

        if filtered_df.empty:
            st.warning("No employees match the selected filter criteria. Please broaden your selection.")
        else:
            # Macro KPI Ribbon
            total_headcount = len(filtered_df)
            labeled_rows = filtered_df[filtered_df["churn"].notna()]
            churn_count = int(labeled_rows["churn"].sum()) if not labeled_rows.empty else 0
            churn_rate = (churn_count / len(labeled_rows)) if not labeled_rows.empty else 0.0
            retention_rate = 1.0 - churn_rate
            avg_salary = float(filtered_df["income"].mean()) if filtered_df["income"].notna().any() else 0.0
            avg_tenure = float(filtered_df["tenure_months"].mean()) if filtered_df["tenure_months"].notna().any() else 0.0

            kpi_cols = st.columns(5)
            kpi_cols[0].metric("👥 Headcount", f"{total_headcount:,}")
            kpi_cols[1].metric("🛡️ Retention Rate", f"{retention_rate:.1%}")
            kpi_cols[2].metric("⚠️ Churn Rate", f"{churn_rate:.1%}")
            kpi_cols[3].metric("💵 Avg Salary", f"R {avg_salary:,.0f}")
            kpi_cols[4].metric("⏱️ Avg Tenure", f"{avg_tenure:.1f} mos")

            st.divider()

            # Department Analysis & Distribution
            col_dept_chart, col_dept_stats = st.columns([1, 1], gap="large")

            with col_dept_chart:
                st.subheader("🏢 Department Turnover Distribution")
                dept_counts = filtered_df.groupby(["department", "churn"]).size().unstack(fill_value=0)
                if 0 not in dept_counts.columns:
                    dept_counts[0] = 0
                if 1 not in dept_counts.columns:
                    dept_counts[1] = 0
                dept_counts = dept_counts.rename(columns={0: "Retained", 1: "Churned"})

                fig_dept, ax_dept = plt.subplots(figsize=(6.0, 3.8))
                dept_counts[["Retained", "Churned"]].plot(
                    kind="bar",
                    stacked=True,
                    color=["#0f766e", "#dc2626"],
                    ax=ax_dept,
                    edgecolor="white",
                )
                ax_dept.set_title("Employee Count by Department & Status", fontsize=11, fontweight="bold")
                ax_dept.set_xlabel("")
                ax_dept.set_ylabel("Employees")
                ax_dept.set_xticklabels(ax_dept.get_xticklabels(), rotation=35, ha="right")
                ax_dept.legend(["Retained", "Churned"], loc="upper right")
                fig_dept.tight_layout()
                st.pyplot(fig_dept, use_container_width=True)
                plt.close(fig_dept)

            with col_dept_stats:
                st.subheader("📋 Department Benchmark Statistics")
                dept_summary = []
                for dept, group in filtered_df.groupby("department"):
                    dept_total = len(group)
                    dept_churners = int(group["churn"].fillna(0).sum())
                    dept_churn_pct = (dept_churners / dept_total * 100) if dept_total else 0.0
                    retained_grp = group[group["churn"] == 0]
                    churned_grp = group[group["churn"] == 1]

                    avg_sal_ret = retained_grp["income"].mean() if not retained_grp.empty else 0.0
                    avg_sal_chu = churned_grp["income"].mean() if not churned_grp.empty else 0.0
                    avg_ten_ret = retained_grp["tenure_months"].mean() if not retained_grp.empty else 0.0
                    avg_ten_chu = churned_grp["tenure_months"].mean() if not churned_grp.empty else 0.0

                    dept_summary.append({
                        "Department": dept,
                        "Headcount": dept_total,
                        "Churn %": f"{dept_churn_pct:.1f}%",
                        "Avg Sal (Retained)": f"R {avg_sal_ret:,.0f}",
                        "Avg Sal (Churned)": f"R {avg_sal_chu:,.0f}",
                        "Avg Ten (Ret)": f"{avg_ten_ret:.1f}m",
                        "Avg Ten (Chu)": f"{avg_ten_chu:.1f}m",
                    })
                st.dataframe(pd.DataFrame(dept_summary), hide_index=True, use_container_width=True, height=280)

            st.divider()

            # Age Cohorts & Compensation Dynamics
            col_age, col_salary = st.columns([1, 1], gap="large")

            with col_age:
                st.subheader("👥 Age Cohort Vulnerability Analysis")
                age_bins = [0, 24, 34, 44, 54, 120]
                age_labels = ["< 25 (Entry)", "25–34 (Early)", "35–44 (Mid)", "45–54 (Senior)", "55+ (Veteran)"]
                filtered_df["age_cohort"] = pd.cut(filtered_df["age"], bins=age_bins, labels=age_labels, right=True)

                cohort_stats = filtered_df.groupby(["age_cohort", "churn"], observed=False).size().unstack(fill_value=0)
                if 0 not in cohort_stats.columns:
                    cohort_stats[0] = 0
                if 1 not in cohort_stats.columns:
                    cohort_stats[1] = 0
                cohort_stats = cohort_stats.rename(columns={0: "Retained", 1: "Churned"})

                fig_age, ax_age = plt.subplots(figsize=(6.0, 3.5))
                cohort_stats[["Retained", "Churned"]].plot(
                    kind="bar",
                    color=["#2563eb", "#ea580c"],
                    ax=ax_age,
                    edgecolor="white",
                )
                ax_age.set_title("Turnover Across Age Cohorts", fontsize=11, fontweight="bold")
                ax_age.set_xlabel("")
                ax_age.set_ylabel("Employee Count")
                ax_age.set_xticklabels(ax_age.get_xticklabels(), rotation=25, ha="right")
                ax_age.legend(["Retained", "Churned"], loc="upper right")
                fig_age.tight_layout()
                st.pyplot(fig_age, use_container_width=True)
                plt.close(fig_age)

            with col_salary:
                st.subheader("💰 Compensation & Contract Type Influences")
                contract_summary = filtered_df.groupby("contract_type")["churn"].agg(
                    Headcount="count",
                    Churners="sum",
                    Churn_Rate=lambda x: f"{x.mean():.1%}" if len(x) else "0.0%",
                    Avg_Salary=lambda x: f"R {filtered_df.loc[x.index, 'income'].mean():,.0f}",
                ).reset_index()

                st.markdown("##### Retention by Contract Type")
                st.dataframe(contract_summary, hide_index=True, use_container_width=True)

                st.markdown("##### Salary Distribution (Retained vs. Churned)")
                fig_sal, ax_sal = plt.subplots(figsize=(6.0, 2.3))
                ret_sal = filtered_df[filtered_df["churn"] == 0]["income"].dropna()
                chu_sal = filtered_df[filtered_df["churn"] == 1]["income"].dropna()

                if not ret_sal.empty:
                    ax_sal.hist(ret_sal, bins=15, alpha=0.6, color="#0f766e", label="Retained", edgecolor="white")
                if not chu_sal.empty:
                    ax_sal.hist(chu_sal, bins=15, alpha=0.6, color="#dc2626", label="Churned", edgecolor="white")
                ax_sal.set_xlabel("Annual Salary (R)")
                ax_sal.set_ylabel("Employees")
                ax_sal.legend(loc="upper right")
                fig_sal.tight_layout()
                st.pyplot(fig_sal, use_container_width=True)
                plt.close(fig_sal)


# =========================================================
# TAB 2: Workforce Roster & Predictions
# =========================================================
with tab_roster:
    st.header("Workforce Roster & Batch Predictions")
    st.caption("Score active employees, benchmark model inference latency, and monitor risk bands.")

    employees = get_all_employees()
    if not employees:
        st.info("No active employee roster found. Use the sidebar to import or create employee records.")
    else:
        raw_employee_df = pd.DataFrame(
            employees,
            columns=["employee_id", "first_name", "last_name", "age", "department", "contract_type", "annual_income", "job_title", "tenure_months"],
        )
        raw_employee_df["employee_name"] = raw_employee_df["first_name"] + " " + raw_employee_df["last_name"]
        employee_df = impute_and_engineer_features(raw_employee_df)

        with st.container(border=True):
            st.subheader("Active Workforce Roster")
            st.dataframe(
                employee_df[["employee_id", "employee_name", "age", "department", "contract_type", "job_title", "annual_income", "tenure_months"]],
                hide_index=True,
                use_container_width=True,
                height=280,
            )

        st.subheader("Batch Churn Scoring")
        probabilities, bands, prediction_errors = [], [], []
        elapsed_scoring_time = 0.0

        try:
            start_score_time = time.perf_counter()
            with st.spinner("Scoring employees with the active churn model..."):
                for emp_id in employee_df["employee_id"]:
                    if use_remote_api and api_backend_url:
                        result = predict_employee_via_api(int(emp_id), api_backend_url)
                        if result.get("error"):
                            result = predict_employee_locally(int(emp_id))
                    else:
                        result = predict_employee_locally(int(emp_id))

                    if result.get("error"):
                        prediction_errors.append(f"{emp_id}: {result['error']}")
                    prob = None if result.get("error") else float(result["churn_probability"])
                    probabilities.append(prob)
                    bands.append(None if prob is None else result.get("risk_band", calculate_risk_band(prob)))
            elapsed_scoring_time = time.perf_counter() - start_score_time
        except FileNotFoundError as error:
            st.error(f"Trained model artifact not found. Please train a model in Tab 4 first: {error}")
            probabilities = [None] * len(employee_df)
            bands = [None] * len(employee_df)

        if elapsed_scoring_time > 0:
            st.info(
                f"⚡ **Inference benchmark:** Scored {len(employee_df)} employees in **{elapsed_scoring_time:.3f}s** "
                f"({elapsed_scoring_time / max(1, len(employee_df)) * 1000:.1f}ms per employee). "
                f"Zero-null feature imputation active."
            )

        if prediction_errors:
            st.warning(f"Some records could not be scored: {'; '.join(prediction_errors)}")

        employee_df["churn_probability"] = probabilities
        employee_df["risk_band"] = bands

        with st.container(border=True):
            st.subheader("Scored Employee Predictions")
            st.dataframe(
                employee_df[[
                    "employee_id", "employee_name", "department", "job_title", "annual_income", "tenure_months", "churn_probability", "risk_band"
                ]].sort_values("churn_probability", ascending=False, na_position="last"),
                hide_index=True,
                use_container_width=True,
                height=320,
            )
            if st.button("💾 Save Batch Predictions to History", use_container_width=True):
                save_inference_predictions(
                    [
                        {
                            "employee_id": int(row.employee_id),
                            "predicted_prob": float(row.churn_probability),
                            "predicted_label": int(row.churn_probability >= 0.5),
                        }
                        for row in employee_df[["employee_id", "churn_probability"]].dropna(subset=["churn_probability"]).itertuples(index=False)
                    ]
                )
                st.success("Batch predictions logged to prediction history database.")

        st.subheader("🎯 Churn Risk Distribution")
        risk_counts = employee_df["risk_band"].value_counts().reindex(["high", "medium", "low"], fill_value=0)
        risk_chart = pd.DataFrame({"Employees": risk_counts})
        risk_chart.index = ["High risk (≥70%)", "Medium risk (45–69%)", "Low risk (<45%)"]
        risk_chart["Share (%)"] = (risk_chart["Employees"] / risk_chart["Employees"].sum() * 100).round(1) if risk_chart["Employees"].sum() else 0.0

        r_col1, r_col2 = st.columns([3, 2], gap="large")
        with r_col1:
            with st.container(border=True):
                st.bar_chart(risk_chart["Employees"], height=240, color="#d95f02")
        with r_col2:
            with st.container(border=True):
                st.dataframe(risk_chart, hide_index=False, use_container_width=True, height=240)


# =========================================================
# TAB 3: Feature Signals & Retention Drivers
# =========================================================
with tab_signals:
    st.header("Feature Signals & Retention Drivers")
    st.caption("Statistical correlation heatmaps, feature influence rankings, and retention drivers.")

    if full_training_data.empty or full_training_data["churn"].dropna().nunique() < 2:
        st.info("Insufficient labeled training data to compute correlation signals. Ensure employees have churn labels (0 or 1).")
    else:
        sig_tab1, sig_tab2 = st.tabs(["🔥 Correlation Heatmap Matrix", "📊 Model Feature Influence Weights"])

        with sig_tab1:
            corr_matrix = compute_feature_correlations(full_training_data)
            if not corr_matrix.empty and "Churn (Target)" in corr_matrix.columns:
                corr_subset_mode = st.radio(
                    "Select Heatmap View:",
                    ["Top Correlated with Churn", "Numeric Features", "Full Correlation Matrix"],
                    horizontal=True,
                )

                churn_corrs = corr_matrix["Churn (Target)"].drop("Churn (Target)", errors="ignore").dropna()
                top_correlated_cols = churn_corrs.abs().sort_values(ascending=False).head(10).index.tolist()

                if corr_subset_mode == "Top Correlated with Churn":
                    selected_cols = top_correlated_cols + ["Churn (Target)"]
                    display_corr = corr_matrix.loc[selected_cols, selected_cols]
                    plot_correlation_heatmap(display_corr, "Top 10 Feature Correlations with Churn")
                elif corr_subset_mode == "Numeric Features":
                    num_cols = ["Age", "Income", "Tenure Months", "Churn (Target)"]
                    avail_num_cols = [c for c in num_cols if c in corr_matrix.columns]
                    display_corr = corr_matrix.loc[avail_num_cols, avail_num_cols]
                    plot_correlation_heatmap(display_corr, "Numeric Features & Churn Correlation Matrix")
                else:
                    plot_correlation_heatmap(corr_matrix, "Full Feature Correlation Matrix")

                st.subheader("Key Retention Signals Ranked by Absolute Influence")
                corr_summary = pd.DataFrame({
                    "Feature": churn_corrs.index,
                    "Correlation with Churn": churn_corrs.values,
                    "Directional Impact": [
                        "⚠️ Increases Turnover Risk (Positive Corr)" if v > 0 else "🛡️ Promotes Retention (Negative Corr)"
                        for v in churn_corrs.values
                    ],
                    "Absolute Magnitude": churn_corrs.abs().values,
                }).sort_values("Absolute Magnitude", ascending=False)
                st.dataframe(corr_summary[["Feature", "Correlation with Churn", "Directional Impact"]].head(12), hide_index=True, use_container_width=True)
            else:
                st.info("Could not compute correlations on current training records.")

        with sig_tab2:
            try:
                active_model = get_prediction_model()
                importances_df = get_feature_importances(active_model)
                if not importances_df.empty:
                    st.subheader("Model Learned Weights / Feature Influence")
                    st.caption("Trained model coefficient/importance weights indicating directional impact on churn predictions.")

                    top_feats = importances_df.head(10).sort_values("Absolute Impact", ascending=True)
                    fig_imp, ax_imp = plt.subplots(figsize=(6.5, 3.8))
                    colors = ["#dc2626" if "Increases" in str(d) else "#0f766e" for d in top_feats["Direction"]]
                    ax_imp.barh(top_feats["Feature"], top_feats["Absolute Impact"], color=colors)
                    ax_imp.set_xlabel("Relative Feature Importance / Weight")
                    ax_imp.set_title("Top 10 Most Influential Features on Churn Model")
                    fig_imp.tight_layout()
                    st.pyplot(fig_imp, use_container_width=True)
                    plt.close(fig_imp)

                    st.dataframe(
                        importances_df[["Feature", "Absolute Impact", "Direction"]].head(15),
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.info("Feature importance data is not available for this model architecture.")
            except Exception as e:
                st.warning(f"Could not load feature importance from model: {e}")


# =========================================================
# TAB 4: Model Studio & Evaluation History
# =========================================================
with tab_model:
    st.header("Model Studio & Governance History")
    st.caption("Hyperparameter tuning, holdout evaluation metrics, and audited prediction history.")

    # Hyperparameter Tuning Section
    with st.container(border=True):
        st.subheader("🎛️ Hyperparameter Tuning & Model Retraining")
        col_alg, col_params = st.columns([1, 2], gap="large")

        with col_alg:
            st.markdown("##### 1. Algorithm")
            algo_choice = st.selectbox(
                "Model Architecture",
                ["Logistic Regression", "Random Forest", "Gradient Boosting"],
            )

        with col_params:
            st.markdown("##### 2. Parameters")
            if algo_choice == "Logistic Regression":
                p_c = st.slider("Regularization Strength (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.05)
                p_class_weight = st.selectbox("Class Weight", ["None", "balanced"], help="Balanced assigns higher penalty to minority churn class")
                p_solver = st.selectbox("Solver", ["lbfgs", "saga", "liblinear"])
                p_max_iter = st.slider("Max Iterations", min_value=100, max_value=1000, value=300, step=50)
                train_kwargs = {
                    "model_type": "logistic_regression",
                    "C": p_c,
                    "class_weight": None if p_class_weight == "None" else "balanced",
                    "solver": p_solver,
                    "max_iter": p_max_iter,
                }
            elif algo_choice == "Random Forest":
                p_trees = st.slider("Number of Trees (n_estimators)", min_value=20, max_value=300, value=100, step=20)
                p_depth = st.slider("Max Tree Depth (max_depth)", min_value=2, max_value=15, value=5)
                p_split = st.selectbox("Min Samples Split", [2, 5, 10])
                p_class_weight = st.selectbox("Class Weight", ["None", "balanced"])
                train_kwargs = {
                    "model_type": "random_forest",
                    "n_estimators": p_trees,
                    "max_depth": p_depth,
                    "min_samples_split": p_split,
                    "class_weight": None if p_class_weight == "None" else "balanced",
                }
            else:  # Gradient Boosting
                p_lr = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1, step=0.01)
                p_trees = st.slider("Number of Estimators", min_value=20, max_value=200, value=100, step=20)
                p_depth = st.slider("Max Depth", min_value=2, max_value=8, value=3)
                train_kwargs = {
                    "model_type": "gradient_boosting",
                    "learning_rate": p_lr,
                    "n_estimators": p_trees,
                    "max_depth": p_depth,
                }

        if st.button("🚀 Train & Evaluate Model with Tuned Parameters", type="primary", use_container_width=True):
            try:
                with st.spinner(f"Fitting and evaluating {algo_choice} pipeline on database records..."):
                    trained_model, metrics, preds, run_id = train_model(**train_kwargs)
                st.cache_resource.clear()
                st.success(f"Trained {algo_choice} successfully! (Saved as Run #{run_id})")

                res_cols = st.columns(5)
                res_cols[0].metric("Accuracy", f"{metrics['accuracy']:.1%}")
                res_cols[1].metric("Precision", f"{metrics['precision']:.1%}")
                res_cols[2].metric("Recall", f"{metrics['recall']:.1%}")
                res_cols[3].metric("F1 Score", f"{metrics['f1']:.1%}")
                res_cols[4].metric("ROC AUC", f"{metrics['roc_auc']:.1%}")
                st.rerun()
            except Exception as err:
                st.error(f"Training failed: {err}")

    st.divider()

    # Evaluation Runs History
    st.subheader("📜 Model Evaluation Runs History")
    runs_df = get_model_runs()
    if runs_df.empty:
        st.info("No training runs recorded yet. Retrain the model above to generate run benchmarks.")
    else:
        if "model_type" in runs_df.columns:
            runs_df["model_architecture"] = runs_df["model_type"].str.replace("_", " ").str.title()
        else:
            runs_df["model_architecture"] = "Logistic Regression"

        latest = runs_df.iloc[0]
        metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        with st.container(border=True):
            latest_arch = str(latest.get("model_architecture", latest.get("model_type", "Logistic Regression"))).replace("_", " ").title()
            st.markdown(f"##### Latest Model Run #{latest['run_id']} — Architecture: `{latest_arch}`")
            metric_cards = st.columns(len(metric_columns), gap="medium")
            for card, metric in zip(metric_cards, metric_columns):
                card.metric(metric.upper(), f"{float(latest[metric]):.1%}")

        display_runs_cols = [c for c in ["run_id", "model_architecture", "accuracy", "precision", "recall", "f1", "roc_auc", "run_timestamp"] if c in runs_df.columns]
        st.dataframe(runs_df[display_runs_cols], hide_index=True, use_container_width=True, height=220)

        run_labels = [
            f"Run #{row.run_id} ({getattr(row, 'model_architecture', getattr(row, 'model_type', 'Model')).replace('_', ' ').title()}) — AUC: {float(row.roc_auc):.1%}"
            for row in runs_df.itertuples()
        ]
        run_map = {label: row.run_id for label, row in zip(run_labels, runs_df.itertuples())}
        selected_label = st.selectbox("Inspect specific evaluation run:", run_labels)
        selected_run = run_map[selected_label]
        predictions = get_predictions_by_run(int(selected_run))

        if not predictions.empty:
            valid = predictions.dropna(subset=["predicted_label", "true_label", "predicted_prob"]).astype({"predicted_label": int, "true_label": int})
            if not valid.empty:
                st.markdown(f"#### Run #{selected_run} Deep-Dive")
                actual_counts = valid["true_label"].value_counts().reindex([0, 1], fill_value=0)
                predicted_counts = valid["predicted_label"].value_counts().reindex([0, 1], fill_value=0)
                classification_df = pd.DataFrame(
                    {"Non-churners": [actual_counts[0], predicted_counts[0]], "Churners": [actual_counts[1], predicted_counts[1]]},
                    index=["Actual", "Predicted"],
                )
                roc_auc_value = float(latest["roc_auc"])

                try:
                    false_pos, true_pos, _ = roc_curve(valid["true_label"], valid["predicted_prob"])
                    stored_auc = float(auc(false_pos, true_pos))
                except ValueError:
                    false_pos, true_pos, stored_auc = [], [], roc_auc_value

                eval_col1, eval_col2 = st.columns(2, gap="large")
                with eval_col1:
                    with st.container(border=True):
                        st.caption("Actual vs. Predicted Distribution")
                        st.bar_chart(classification_df, height=260)
                with eval_col2:
                    with st.container(border=True):
                        st.caption("ROC Curve & Discrimination")
                        roc_fig, roc_ax = plt.subplots(figsize=(4.8, 3.0))
                        if len(false_pos):
                            roc_ax.plot(false_pos, true_pos, color="#d95f02", linewidth=2, label=f"ROC (AUC {stored_auc:.2f})")
                        roc_ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
                        roc_ax.set_xlabel("False positive rate")
                        roc_ax.set_ylabel("True positive rate")
                        roc_ax.legend(loc="lower right", fontsize=8)
                        roc_fig.tight_layout()
                        st.pyplot(roc_fig, use_container_width=True)
                        plt.close(roc_fig)

                show_confusion_matrix(valid["true_label"], valid["predicted_label"], f"Evaluation Confusion Matrix (Run #{selected_run})")
                tp = int(((valid["predicted_label"] == 1) & (valid["true_label"] == 1)).sum())
                tn = int(((valid["predicted_label"] == 0) & (valid["true_label"] == 0)).sum())
                fp = int(((valid["predicted_label"] == 1) & (valid["true_label"] == 0)).sum())
                fn = int(((valid["predicted_label"] == 0) & (valid["true_label"] == 1)).sum())
                st.caption(f"📊 **Metrics Summary:** True Positives: {tp} | True Negatives: {tn} | False Positives: {fp} | False Negatives: {fn}")

    st.divider()

    # Audited Historical Prediction Log
    st.subheader("⏱️ Audited Prediction History Log")
    filter_option = st.selectbox("Date filter:", ["All", "Today", "Yesterday", "This week", "Last week", "Custom range"])

    def compute_range(option):
        today = datetime.now().date()
        if option == "Today":
            return today, today
        if option == "Yesterday":
            day = today - timedelta(days=1)
            return day, day
        if option == "This week":
            return today - timedelta(days=today.weekday()), today
        if option == "Last week":
            start = today - timedelta(days=today.weekday() + 7)
            end = today - timedelta(days=today.weekday() + 1)
            return start, end
        return None, None

    start_date, end_date = compute_range(filter_option)
    if filter_option == "Custom range":
        s_col, e_col = st.columns(2)
        with s_col:
            custom_start = st.date_input("Start date", value=datetime.now().date())
        with e_col:
            custom_end = st.date_input("End date", value=datetime.now().date())
        start_date, end_date = custom_start, custom_end

    if start_date is not None and end_date is not None and start_date > end_date:
        st.error("Start date must be on or before end date.")
    else:
        prediction_history = get_prediction_history(
            start_date.isoformat() if start_date is not None else None,
            end_date.isoformat() if end_date is not None else None,
        )
        if prediction_history.empty:
            st.info("No saved predictions match this date range.")
        else:
            h_metrics = st.columns(3)
            h_metrics[0].metric("Total Predictions", f"{len(prediction_history):,}")
            h_metrics[1].metric("Avg Churn Probability", f"{prediction_history['predicted_prob'].mean():.1%}")
            h_metrics[2].metric("Predicted Churners", f"{int(prediction_history['predicted_label'].sum()):,}")
            st.dataframe(
                prediction_history[[
                    "employee_id", "employee_name", "predicted_prob", "predicted_label", "true_label", "prediction_source", "created_at"
                ]],
                hide_index=True,
                use_container_width=True,
                height=280,
            )
