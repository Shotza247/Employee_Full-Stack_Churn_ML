# Databricks notebook source
from datetime import datetime, timedelta
import importlib
import os
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from sklearn.metrics import auc, confusion_matrix, roc_curve
import streamlit as st

import churn_app.config
import churn_app.database
import churn_app.faker_producer
import churn_app.feature_engineering
import churn_app.prediction
import churn_app.training

# Ensure clean hot-reloading across Streamlit reruns
importlib.reload(churn_app.database)
importlib.reload(churn_app.faker_producer)
importlib.reload(churn_app.feature_engineering)
importlib.reload(churn_app.prediction)
importlib.reload(churn_app.training)

from churn_app.config import MODEL_PATH
from churn_app.database import (
    diagnose_dataframe_quality,
    diagnose_row_quality,
    get_all_employees,
    get_data_lineage_records,
    get_employee_features,
    get_model_runs,
    get_prediction_history,
    get_predictions_by_run,
    get_training_data,
    import_employee_dataframe,
    init_db,
    insert_employee,
    is_source_imported,
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


def show_quality_cards(report: dict[str, int | float], location=st, diagnosed_df: pd.DataFrame | None = None):
    cards = location.columns(3)
    cards[0].metric("Rows", f"{report['rows']:,}")
    cards[1].metric("Duplicates", f"{report['duplicates']:,}")
    cards[2].metric("Missing", f"{report['missing']:,}")
    cards2 = location.columns(3)
    cards2[0].metric("Invalid", f"{report['invalid']:,}")
    cards2[1].metric("Columns", f"{report['columns']:,}")
    cards2[2].metric("Num Mean", f"{report['numeric_mean']:,.0f}")
    if diagnosed_df is not None and not diagnosed_df.empty and "data_quality_category" in diagnosed_df.columns:
        cat_counts = diagnosed_df["data_quality_category"].value_counts().reset_index()
        cat_counts.columns = ["Data Quality Category", "Count"]
        location.caption("Quality Category Breakdown:")
        location.dataframe(cat_counts, hide_index=True, use_container_width=True)


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
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=["Predicted Non-churner", "Predicted Churner"],
                y=["Actual Non-churner", "Actual Churner"],
                text=[[str(matrix[r][c]) for c in range(2)] for r in range(2)],
                texttemplate="<b>%{text}</b>",
                textfont={"size": 16, "color": "white"},
                colorscale="Blues",
                showscale=False,
                hovertemplate="<b>Actual:</b> %{y}<br><b>Predicted:</b> %{x}<br><b>Count:</b> %{z}<extra></extra>",
            )
        )
        fig.update_layout(
            title={"text": "Interactive Confusion Matrix", "font": {"size": 13}},
            margin={"l": 30, "r": 20, "t": 35, "b": 35},
            height=260,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)
    with counts_col:
        st.dataframe(matrix_df, use_container_width=True)
    return matrix


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, title: str = "Feature Correlation Heatmap"):
    if corr_matrix.empty:
        st.info("No correlation data available.")
        return
    text_matrix = [[f"{val:.2f}" if pd.notna(val) else "" for val in row] for row in corr_matrix.values]
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            zmin=-1.0,
            zmax=1.0,
            colorscale="RdBu_r",
            text=text_matrix,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar={"title": "Corr", "thickness": 14},
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": title, "font": {"size": 13}},
        margin={"l": 30, "r": 20, "t": 35, "b": 35},
        height=max(360, len(corr_matrix.index) * 32),
        template="plotly_white",
        xaxis={"tickangle": -35},
    )
    st.plotly_chart(fig, use_container_width=True)


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
    st.markdown("### 🏢 System Operations")
    st.caption("Active operational controls, inference routing, and serverless backends.")

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



# =========================================================
# Main Panel: 5-Tab Stakeholder Architecture
# =========================================================
tab_ingest, tab_insights, tab_roster, tab_signals, tab_model = st.tabs([
    "📥 Data Ingestion, Synthesis & Lineage Studio",
    "📊 Executive HR Insights & Demographics",
    "👥 Workforce Roster & Predictions",
    "🔍 Feature Signals & Retention Drivers",
    "⚙️ Model Studio & Evaluation History",
])

# Load base workforce dataset
full_training_data = get_training_data()

# =========================================================
# TAB 1: Data Ingestion, Synthesis & Lineage Studio
# =========================================================
with tab_ingest:
    st.header("Data Ingestion, Synthesis & Lineage Studio")
    st.caption("Enterprise ingestion channels, Faker identity auto-repair, data quality validation, and end-to-end data provenance.")

    # Segmented Ingestion Mode Selector
    ingest_mode = st.radio(
        "Select Ingestion Channel:",
        [
            "📥 Batch CSV Import & Auto-Repair",
            "🎲 Synthetic Batch Generator (Faker)",
            "👤 Single Employee Manual Entry",
        ],
        horizontal=True,
        label_visibility="collapsed",
        key="tab1_ingest_mode",
    )

    # -------------------------------------------------------------
    # MODE 1: Batch CSV Ingestion & 1-Click Auto-Repair
    # -------------------------------------------------------------
    if ingest_mode == "📥 Batch CSV Import & Auto-Repair":
        with st.container(border=True):
            st.subheader("📥 Batch CSV Ingestion & Quality Studio")
            st.caption("Upload workforce CSV files, inspect row-level quality diagnostics, and resolve duplicate or malformed records.")

            with st.expander("📂 Download Packaged Test Datasets (`data/Bulk_import_Data/`)", expanded=False):
                st.markdown("Ready-to-use sample CSV rosters with varied data quality characteristics:")
                bulk_data_dir = Path("data/Bulk_import_Data")
                sample_files = [
                    ("noisy_churn_experiment_900.csv", "🧪 900-Row Noisy Dataset (Outliers & Missing Values)"),
                    ("synthetic_churn_data_updated.csv", "✅ Clean Benchmark Dataset (100% Valid Records)"),
                    ("synthetical_churn_data_updated_raw.csv", "⚠️ Raw Stress-Test Dataset (Missing Fields)"),
                ]
                dl_cols = st.columns(len(sample_files))
                for idx, (fname, flabel) in enumerate(sample_files):
                    fpath = bulk_data_dir / fname
                    if fpath.exists():
                        with open(fpath, "rb") as f_in:
                            with dl_cols[idx]:
                                st.download_button(
                                    label=flabel,
                                    data=f_in.read(),
                                    file_name=fname,
                                    mime="text/csv",
                                    use_container_width=True,
                                    key=f"tab1_dl_{fname}",
                                )

            uploaded_file = st.file_uploader("Upload workforce CSV file", type="csv", key="tab1_uploader")
            if uploaded_file is not None:
                try:
                    csv_filename = uploaded_file.name
                    uploaded_df = pd.read_csv(uploaded_file)
                    if "income" in uploaded_df.columns and "annual_income" not in uploaded_df.columns:
                        uploaded_df["annual_income"] = uploaded_df["income"]
                    missing_columns = REQUIRED_COLUMNS - set(uploaded_df.columns)
                    if missing_columns:
                        st.error(f"Missing required columns: {', '.join(sorted(missing_columns))}")
                    else:
                        diagnosed_df = diagnose_dataframe_quality(uploaded_df)
                        
                        # Full-width quality assessment cards
                        show_quality_cards(quality_report(uploaded_df), st, diagnosed_df=diagnosed_df)

                        with st.expander(f"📋 Preview First 10 Rows of `{csv_filename}` (with Quality Diagnostic Badges)", expanded=True):
                            core_cols = ["first_name", "last_name", "department", "data_quality_issue", "data_quality_category", "age", "contract_type", "annual_income", "job_title", "tenure_months"]
                            display_cols = [c for c in core_cols if c in diagnosed_df.columns] + [c for c in diagnosed_df.columns if c not in core_cols]
                            st.dataframe(diagnosed_df[display_cols].head(10), hide_index=True, use_container_width=True)

                        # Duplicate Source Check
                        already_imported, existing_count = is_source_imported(csv_filename)
                        if already_imported:
                            st.warning(
                                f"⚠️ **File Already Ingested**: `{csv_filename}` has already been imported into the database "
                                f"({existing_count:,} records currently logged with source `{csv_filename}`). "
                                "Re-importing the same CSV is disabled to protect data integrity and prevent duplicate records."
                            )
                            st.button(f"🚫 '{csv_filename}' Already Ingested (Re-import Blocked)", disabled=True, use_container_width=True, key="tab1_import_btn_blocked")
                        else:
                            if st.button(f"🚀 Import Valid Rows from '{csv_filename}' to Database", type="primary", use_container_width=True, key="tab1_import_btn"):
                                accepted, rejected = import_employee_dataframe(uploaded_df, source_system=csv_filename, prevent_duplicate_source=True)
                                st.session_state["last_import"] = {
                                    "file_name": csv_filename,
                                    "accepted": accepted,
                                    "rejected": rejected,
                                    "predictions": None,
                                }
                                st.rerun()
                except (pd.errors.ParserError, UnicodeDecodeError) as error:
                    st.error(f"Could not parse CSV: {error}")

            # Post-import results & Faker auto-repair container
            last_import = st.session_state.get("last_import")
            if last_import:
                st.markdown("---")
                st.markdown("#### 📊 Latest Ingestion Summary")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.success(f"✅ **Accepted Records**: {len(last_import['accepted'])} records ingested from `{last_import.get('file_name', 'CSV')}`.")
                with res_col2:
                    if not last_import["rejected"].empty:
                        st.warning(f"⚠️ **Rejected Records**: {len(last_import['rejected'])} invalid rows filtered out.")
                    else:
                        st.info("🎉 100% of rows passed all validation and uniqueness gates.")

                if not last_import["rejected"].empty:
                    with st.expander("🔍 View Rejected Rows Quality Breakdown", expanded=False):
                        rej_df = last_import["rejected"]
                        if "data_quality_category" in rej_df.columns:
                            cat_summary = rej_df["data_quality_category"].value_counts().reset_index()
                            cat_summary.columns = ["Data Quality Category", "Count"]
                            st.dataframe(cat_summary, hide_index=True, use_container_width=True)
                        st.dataframe(rej_df, hide_index=True, use_container_width=True)

                    # 1-Click Faker Auto-Repair Section
                    with st.container(border=True):
                        r_col1, r_col2 = st.columns([3, 1])
                        with r_col1:
                            st.markdown("##### 🪄 1-Click Faker Auto-Repair")
                            st.caption("Synthesize valid identities, resolve duplicate collisions, and repair missing features for rejected rows.")
                        with r_col2:
                            if st.button("⚡ Auto-Repair & Ingest", type="primary", use_container_width=True, key="tab1_repair_btn"):
                                producer = FakerProducer()
                                repaired_df = producer.repair_rejected_rows(last_import["rejected"])
                                if not repaired_df.empty:
                                    cols_to_import = [c for c in repaired_df.columns if c not in ["source_row", "reason", "faker_repairs", "data_quality_category", "data_quality_issue"]]
                                    import_payload = repaired_df[cols_to_import].copy()
                                    origin_file = last_import.get("file_name", "csv_import")
                                    repaired_source = f"faker_repaired ({origin_file})"
                                    new_accepted, new_rejected = import_employee_dataframe(import_payload, source_system=repaired_source)

                                    combined_accepted = pd.concat([last_import["accepted"], new_accepted], ignore_index=True)
                                    st.session_state["last_import"]["accepted"] = combined_accepted
                                    st.session_state["last_import"]["rejected"] = new_rejected
                                    st.session_state["repaired_preview"] = repaired_df[["first_name", "last_name", "department", "job_title", "data_quality_issue", "faker_repairs"]]
                                    st.success(f"Repaired and imported {len(new_accepted)} rows via Faker Producer (source: `{repaired_source}`)!")
                                    st.rerun()

                    if "repaired_preview" in st.session_state:
                        with st.expander("✨ Recently Repaired Rows Audit", expanded=False):
                            st.dataframe(st.session_state["repaired_preview"], hide_index=True, use_container_width=True)

    # -------------------------------------------------------------
    # MODE 2: Synthetic Batch Generator (Faker)
    # -------------------------------------------------------------
    elif ingest_mode == "🎲 Synthetic Batch Generator (Faker)":
        with st.container(border=True):
            st.subheader("🎲 Synthetic Workforce Batch Producer")
            st.caption("Generate realistic, enterprise-compliant synthetic employee records with controllable churn signals using Python Faker.")

            s_col1, s_col2 = st.columns(2)
            with s_col1:
                synth_count = st.slider("Record batch size", min_value=5, max_value=100, value=20, step=5, key="tab1_synth_slider")
                st.caption(f"Generates **{synth_count}** unique synthetic employee profiles.")
            with s_col2:
                synth_churn_bias = st.slider("Baseline Churn Bias", min_value=0.05, max_value=0.80, value=0.25, step=0.05, format="%.0f%%", key="tab1_bias_slider")
                st.caption(f"Applies a **{synth_churn_bias:.0%}** statistical likelihood of churn across generated cohorts.")

            if st.button("🎲 Generate & Ingest Synthetic Batch to Database", type="primary", use_container_width=True, key="tab1_synth_btn"):
                synth_producer = FakerProducer()
                synth_batch = synth_producer.generate_synthetic_batch(count=synth_count, churn_prob=synth_churn_bias)
                synth_accepted, synth_rejected = import_employee_dataframe(synth_batch, source_system="faker_synthetic")
                st.success(f"🎉 Successfully generated and ingested {len(synth_accepted)} synthetic employee records (Provenance: `faker_synthetic`).")
                st.rerun()

    # -------------------------------------------------------------
    # MODE 3: Single Employee Manual Entry
    # -------------------------------------------------------------
    elif ingest_mode == "👤 Single Employee Manual Entry":
        with st.container(border=True):
            st.subheader("👤 Single Employee Ingestion & Instant Scoring")
            st.caption("Manually insert an individual employee record into SQLite with localized South African Rand (R / ZAR) compensation and receive immediate ML risk inference.")

            with st.form("tab1_employee_form"):
                f_col1, f_col2 = st.columns(2)
                with f_col1:
                    first_name = st.text_input("First Name", placeholder="e.g. Sipho")
                    age = st.number_input("Age", min_value=18, max_value=99, value=30)
                    department = st.selectbox("Department", sorted(VALID_DEPARTMENTS), key="tab1_form_dept")
                    annual_income = st.number_input("Annual Income (R / ZAR)", min_value=0.0, value=650000.0, step=10000.0, format="%.2f")
                with f_col2:
                    last_name = st.text_input("Last Name", placeholder="e.g. Dlamini")
                    contract_type = st.selectbox("Contract Type", sorted(VALID_CONTRACTS), key="tab1_form_contract")
                    job_title = st.text_input("Job Title", value="Specialist")
                    tenure_months = st.number_input("Tenure (Months)", min_value=0, max_value=240, value=12)

                submit_btn = st.form_submit_button("🚀 Insert & Score Employee", type="primary", use_container_width=True)
                if submit_btn:
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
                            source_system="manual_entry",
                        )
                        if new_id is None:
                            st.error("Employee was inserted, but ID was not returned.")
                        else:
                            st.success(f"✅ Inserted employee #{new_id} ({first_name} {last_name}) with provenance `manual_entry`.")
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
                                    risk_col = "🟢" if result["risk_band"] == "low" else "🟡" if result["risk_band"] == "medium" else "🔴"
                                    st.info(f"{risk_col} Instant Churn Risk Assessment: **{result['risk_band'].upper()} RISK** (Churn Probability: **{result['churn_probability']:.1%}**)")
                            except Exception as ex:
                                st.warning(f"Scoring note: {ex}")
                            st.rerun()

    # Data Lineage & Provenance Explorer
    st.markdown("---")
    st.subheader("🔍 Data Lineage & Provenance Explorer")
    st.caption("Trace every employee record from creation source to database storage, feature engineering, and model inference.")

    lineage_records = get_data_lineage_records()
    if not lineage_records.empty:
        available_sources = ["All Sources"] + sorted(list(lineage_records["source_system"].dropna().unique()))
        l_col1, l_col2, l_col3 = st.columns([2, 2, 2])
        with l_col1:
            source_filter = st.selectbox("Filter by Source Lineage System:", available_sources)

        filtered_lineage = lineage_records.copy()
        if source_filter != "All Sources":
            filtered_lineage = filtered_lineage[filtered_lineage["source_system"] == source_filter]

        with l_col2:
            st.metric("Total Lineage Records", f"{len(lineage_records):,}")
        with l_col3:
            st.metric(f"Filtered Records ({source_filter})", f"{len(filtered_lineage):,}")

        # Provenance source breakdown
        st.markdown("##### Provenance Source Distribution:")
        src_counts = lineage_records["source_system"].value_counts().reset_index()
        src_counts.columns = ["Source System", "Record Count"]
        src_counts["Proportion (%)"] = (src_counts["Record Count"] / len(lineage_records) * 100).round(1)
        st.dataframe(src_counts, hide_index=True, use_container_width=True)

        st.markdown("##### Audited Lineage Records:")
        display_lineage_cols = ["employee_id", "employee_name", "department", "contract_type", "annual_income", "tenure_months", "churn", "source_system", "created_at"]
        st.dataframe(filtered_lineage[display_lineage_cols], hide_index=True, use_container_width=True, height=280)
    else:
        st.info("No lineage records found in SQLite database.")

    # SQLite Database Health
    st.markdown("---")
    st.subheader("📊 SQLite Database Health & Schema Inspector")
    with st.expander("Database Health Scorecard & Active Schemas", expanded=False):
        current_data = get_training_data()
        current_diagnosed = diagnose_dataframe_quality(current_data) if not current_data.empty else None
        show_quality_cards(quality_report(current_data), st, diagnosed_df=current_diagnosed)

# Load base workforce dataset
full_training_data = get_training_data()

# =========================================================
# TAB 2: Executive HR Insights & Demographics
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

                departments = dept_counts.index.tolist()
                fig_dept = go.Figure()
                fig_dept.add_trace(go.Bar(
                    x=departments,
                    y=dept_counts["Retained"].tolist(),
                    name="Retained",
                    marker_color="#0f766e",
                    hovertemplate="<b>Department:</b> %{x}<br><b>Status:</b> Retained<br><b>Employees:</b> %{y}<extra></extra>",
                ))
                fig_dept.add_trace(go.Bar(
                    x=departments,
                    y=dept_counts["Churned"].tolist(),
                    name="Churned",
                    marker_color="#dc2626",
                    hovertemplate="<b>Department:</b> %{x}<br><b>Status:</b> Churned<br><b>Employees:</b> %{y}<extra></extra>",
                ))
                fig_dept.update_layout(
                    barmode="stack",
                    title={"text": "Employee Count by Department & Status", "font": {"size": 13}},
                    margin={"l": 30, "r": 20, "t": 40, "b": 40},
                    height=340,
                    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
                    template="plotly_white",
                    xaxis={"tickangle": -30},
                    yaxis={"title": "Employees"},
                )
                st.plotly_chart(fig_dept, use_container_width=True)

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

            # Age Cohorts & Contract Dynamics
            col_age, col_contract = st.columns([1, 1], gap="large")

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

                fig_age = go.Figure()
                fig_age.add_trace(go.Bar(
                    x=cohort_stats.index.tolist(),
                    y=cohort_stats["Retained"].tolist(),
                    name="Retained",
                    marker_color="#2563eb",
                    hovertemplate="<b>Cohort:</b> %{x}<br><b>Status:</b> Retained<br><b>Employees:</b> %{y}<extra></extra>",
                ))
                fig_age.add_trace(go.Bar(
                    x=cohort_stats.index.tolist(),
                    y=cohort_stats["Churned"].tolist(),
                    name="Churned",
                    marker_color="#ea580c",
                    hovertemplate="<b>Cohort:</b> %{x}<br><b>Status:</b> Churned<br><b>Employees:</b> %{y}<extra></extra>",
                ))
                fig_age.update_layout(
                    barmode="group",
                    title={"text": "Turnover Across Age Cohorts", "font": {"size": 13}},
                    margin={"l": 30, "r": 20, "t": 40, "b": 40},
                    height=320,
                    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
                    template="plotly_white",
                    xaxis={"tickangle": -20},
                    yaxis={"title": "Employee Count"},
                )
                st.plotly_chart(fig_age, use_container_width=True)

            with col_contract:
                st.subheader("📝 Contract Type & Employment Dynamics")
                contract_summary = filtered_df.groupby("contract_type")["churn"].agg(
                    Headcount="count",
                    Churners="sum",
                    Churn_Rate=lambda x: f"{x.mean():.1%}" if len(x) else "0.0%",
                    Avg_Annual_Salary=lambda x: f"R {filtered_df.loc[x.index, 'income'].mean():,.0f}",
                ).reset_index()

                st.markdown("##### Retention Breakdown by Contract Type")
                st.dataframe(contract_summary, hide_index=True, use_container_width=True)

                # Contract type retention visual (Interactive Plotly Horizontal Bar)
                contract_counts = filtered_df.groupby(["contract_type", "churn"]).size().unstack(fill_value=0)
                if 0 not in contract_counts.columns: contract_counts[0] = 0
                if 1 not in contract_counts.columns: contract_counts[1] = 0
                contract_counts = contract_counts.rename(columns={0: "Retained", 1: "Churned"})

                fig_ct = go.Figure()
                fig_ct.add_trace(go.Bar(
                    y=contract_counts.index.tolist(),
                    x=contract_counts["Retained"].tolist(),
                    name="Retained",
                    orientation="h",
                    marker_color="#0f766e",
                    hovertemplate="<b>Contract:</b> %{y}<br><b>Status:</b> Retained<br><b>Headcount:</b> %{x}<extra></extra>",
                ))
                fig_ct.add_trace(go.Bar(
                    y=contract_counts.index.tolist(),
                    x=contract_counts["Churned"].tolist(),
                    name="Churned",
                    orientation="h",
                    marker_color="#dc2626",
                    hovertemplate="<b>Contract:</b> %{y}<br><b>Status:</b> Churned<br><b>Headcount:</b> %{x}<extra></extra>",
                ))
                fig_ct.update_layout(
                    barmode="stack",
                    title={"text": "Contract Type Attrition Comparison", "font": {"size": 12}},
                    margin={"l": 30, "r": 20, "t": 35, "b": 25},
                    height=200,
                    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
                    template="plotly_white",
                    xaxis={"title": "Headcount"},
                )
                st.plotly_chart(fig_ct, use_container_width=True)

            st.divider()

            # Dedicated Row: Salary Distribution (Retained vs. Churned)
            st.subheader("💵 Salary Distribution (Retained vs. Churned)")
            st.caption("Distributional comparison of annual compensation (ZAR / R) across retained vs. churned employees to diagnose salary-driven turnover.")

            sal_col1, sal_col2 = st.columns([2, 1], gap="large")

            with sal_col1:
                ret_sal = filtered_df[filtered_df["churn"] == 0]["income"].dropna()
                chu_sal = filtered_df[filtered_df["churn"] == 1]["income"].dropna()

                fig_sal = go.Figure()
                if not ret_sal.empty:
                    fig_sal.add_trace(go.Histogram(
                        x=ret_sal,
                        name=f"Retained (n={len(ret_sal):,})",
                        opacity=0.65,
                        marker_color="#0f766e",
                        hovertemplate="<b>Retained</b><br>Salary: R %{x:,.0f}<br>Count: %{y}<extra></extra>",
                    ))
                if not chu_sal.empty:
                    fig_sal.add_trace(go.Histogram(
                        x=chu_sal,
                        name=f"Churned (n={len(chu_sal):,})",
                        opacity=0.65,
                        marker_color="#dc2626",
                        hovertemplate="<b>Churned</b><br>Salary: R %{x:,.0f}<br>Count: %{y}<extra></extra>",
                    ))

                fig_sal.update_layout(
                    barmode="overlay",
                    title={"text": "Annual Salary Distribution Overlay (ZAR / R)", "font": {"size": 13}},
                    margin={"l": 30, "r": 20, "t": 40, "b": 40},
                    height=340,
                    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
                    template="plotly_white",
                    xaxis={"title": "Annual Salary (ZAR / R)"},
                    yaxis={"title": "Employee Count"},
                )
                st.plotly_chart(fig_sal, use_container_width=True)

            with sal_col2:
                st.markdown("##### 📊 Salary Benchmark Statistics")
                mean_ret = float(ret_sal.mean()) if not ret_sal.empty else 0.0
                mean_chu = float(chu_sal.mean()) if not chu_sal.empty else 0.0
                median_ret = float(ret_sal.median()) if not ret_sal.empty else 0.0
                median_chu = float(chu_sal.median()) if not chu_sal.empty else 0.0
                diff_mean = mean_ret - mean_chu

                sal_stats_df = pd.DataFrame([
                    {"Metric": "Mean Annual Salary", "Retained": f"R {mean_ret:,.0f}", "Churned": f"R {mean_chu:,.0f}"},
                    {"Metric": "Median Annual Salary", "Retained": f"R {median_ret:,.0f}", "Churned": f"R {median_chu:,.0f}"},
                    {"Metric": "Min Annual Salary", "Retained": f"R {ret_sal.min():,.0f}" if not ret_sal.empty else "R 0", "Churned": f"R {chu_sal.min():,.0f}" if not chu_sal.empty else "R 0"},
                    {"Metric": "Max Annual Salary", "Retained": f"R {ret_sal.max():,.0f}" if not ret_sal.empty else "R 0", "Churned": f"R {chu_sal.max():,.0f}" if not chu_sal.empty else "R 0"},
                ])
                st.dataframe(sal_stats_df, hide_index=True, use_container_width=True)

                if diff_mean > 0:
                    st.info(f"💡 Retained staff earn **R {diff_mean:,.0f} (+{(diff_mean/mean_chu*100):.1f}%)** higher average annual compensation than departing employees.")
                elif diff_mean < 0:
                    st.warning(f"⚠️ Departing staff had higher average compensation by **R {abs(diff_mean):,.0f}**.")


# =========================================================
# TAB 3: Workforce Roster & Predictions
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
                risk_colors = {"High risk (≥70%)": "#dc2626", "Medium risk (45–69%)": "#f59e0b", "Low risk (<45%)": "#10b981"}
                fig_risk = go.Figure(go.Bar(
                    x=risk_chart.index.tolist(),
                    y=risk_chart["Employees"].tolist(),
                    marker_color=[risk_colors.get(idx, "#3b82f6") for idx in risk_chart.index],
                    text=[f"{int(count):,} ({pct:.1f}%)" for count, pct in zip(risk_chart["Employees"], risk_chart["Share (%)"])],
                    textposition="auto",
                    hovertemplate="<b>Risk Tier:</b> %{x}<br><b>Employees:</b> %{y:,}<extra></extra>",
                ))
                fig_risk.update_layout(
                    title={"text": "Workforce Churn Risk Distribution", "font": {"size": 13}},
                    margin={"l": 30, "r": 20, "t": 35, "b": 30},
                    height=240,
                    template="plotly_white",
                    yaxis={"title": "Employees"},
                )
                st.plotly_chart(fig_risk, use_container_width=True)
        with r_col2:
            with st.container(border=True):
                st.dataframe(risk_chart, hide_index=False, use_container_width=True, height=240)


# =========================================================
# TAB 4: Feature Signals & Retention Drivers
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
                    colors = ["#dc2626" if "Increases" in str(d) else "#0f766e" for d in top_feats["Direction"]]
                    fig_imp = go.Figure(go.Bar(
                        y=top_feats["Feature"].tolist(),
                        x=top_feats["Absolute Impact"].tolist(),
                        orientation="h",
                        marker_color=colors,
                        customdata=top_feats["Direction"].tolist(),
                        hovertemplate="<b>Feature:</b> %{y}<br><b>Impact Weight:</b> %{x:.4f}<br><b>Direction:</b> %{customdata}<extra></extra>",
                    ))
                    fig_imp.update_layout(
                        title={"text": "Top 10 Most Influential Features on Churn Model", "font": {"size": 13}},
                        margin={"l": 30, "r": 20, "t": 40, "b": 40},
                        height=340,
                        template="plotly_white",
                        xaxis={"title": "Relative Feature Importance / Weight"},
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)

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
# TAB 5: Model Studio & Evaluation History
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
                        fig_roc = go.Figure()
                        if len(false_pos):
                            fig_roc.add_trace(go.Scatter(
                                x=false_pos,
                                y=true_pos,
                                mode="lines",
                                name=f"ROC (AUC = {stored_auc:.3f})",
                                line={"color": "#d95f02", "width": 3},
                                hovertemplate="<b>FPR:</b> %{x:.3f}<br><b>TPR:</b> %{y:.3f}<extra></extra>",
                            ))
                        fig_roc.add_trace(go.Scatter(
                            x=[0, 1],
                            y=[0, 1],
                            mode="lines",
                            name="Random Baseline (AUC = 0.50)",
                            line={"color": "#9ca3af", "width": 1.5, "dash": "dash"},
                            hoverinfo="skip",
                        ))
                        fig_roc.update_layout(
                            title={"text": f"ROC Curve (AUC: {stored_auc:.2f})", "font": {"size": 12}},
                            margin={"l": 30, "r": 20, "t": 35, "b": 35},
                            height=250,
                            legend={"orientation": "h", "yanchor": "bottom", "y": -0.3, "xanchor": "center", "x": 0.5},
                            template="plotly_white",
                            xaxis={"title": "False Positive Rate", "range": [0, 1]},
                            yaxis={"title": "True Positive Rate", "range": [0, 1]},
                        )
                        st.plotly_chart(fig_roc, use_container_width=True)

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
