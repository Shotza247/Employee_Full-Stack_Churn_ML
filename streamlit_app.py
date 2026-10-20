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


st.set_page_config(page_title="Employee Churn Intelligence", layout="wide")
st.title("Employee Churn Intelligence")
st.caption("Onboard employees, inspect data quality, tune models, and monitor churn behavior.")

with st.sidebar:
    st.header("Settings & API")
    with st.expander("🌐 Cloud API (Vercel Backend)", expanded=False):
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

    st.header("Batch import")
    st.caption("Upload CSV dataset to import, inspect, and score employee records.")
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
                start_pred_time = time.perf_counter()
                try:
                    with st.spinner("Scoring imported employees with the churn model..."):
                        for employee_id in last_import["accepted"]["employee_id"]:
                            if use_remote_api and api_backend_url:
                                result = predict_employee_via_api(int(employee_id), api_backend_url)
                                if result.get("error"):
                                    result = predict_employee_locally(int(employee_id))
                            else:
                                result = predict_employee_locally(int(employee_id))
                            prediction_rows.append({"employee_id": int(employee_id), **result})
                    elapsed_pred_time = time.perf_counter() - start_pred_time
                    save_inference_predictions(
                        [
                            {
                                "employee_id": row["employee_id"],
                                "predicted_prob": row["churn_probability"],
                                "predicted_label": int(row["churn_probability"] >= 0.5),
                            }
                            for row in prediction_rows
                            if "churn_probability" in row
                        ]
                    )
                except (FileNotFoundError, ValueError) as error:
                    st.error(f"Cannot predict or save prediction history: {error}")
                else:
                    last_import["predictions"] = pd.DataFrame(prediction_rows)
                    st.session_state["last_import"] = last_import
                    st.success(
                        f"⏱️ Scored {len(last_import['accepted'])} imported employees in {elapsed_pred_time:.3f}s "
                        f"({elapsed_pred_time / max(1, len(last_import['accepted'])) * 1000:.1f}ms/employee)."
                    )

        if not last_import["rejected"].empty:
            st.warning(f"Rejected {len(last_import['rejected'])} rows")
            with st.expander("Rejected rows summary", expanded=True):
                st.dataframe(last_import["rejected"], hide_index=True, use_container_width=True)

            # Python Faker Producer Section to repair unapproved / rejected records
            with st.container(border=True):
                st.markdown("#### 🪄 Faker Data Producer & Repair")
                st.caption(
                    "Trigger the Python Faker producer to synthesize missing first/last names "
                    "and resolve identity rejections so unapproved rows can be tested."
                )
                if st.button("⚡ Generate Fake Names & Repair Rejected Rows", use_container_width=True):
                    producer = FakerProducer()
                    repaired_df = producer.repair_rejected_rows(last_import["rejected"])
                    if not repaired_df.empty:
                        # Clean columns for insertion
                        cols_to_import = [c for c in repaired_df.columns if c not in ["source_row", "reason", "faker_repairs"]]
                        import_payload = repaired_df[cols_to_import].copy()
                        new_accepted, new_rejected = import_employee_dataframe(import_payload)

                        # Update last_import state
                        combined_accepted = pd.concat([last_import["accepted"], new_accepted], ignore_index=True)
                        st.session_state["last_import"]["accepted"] = combined_accepted
                        st.session_state["last_import"]["rejected"] = new_rejected
                        st.session_state["repaired_preview"] = repaired_df[["first_name", "last_name", "department", "job_title", "faker_repairs"]]
                        st.success(f"Faker repaired and imported {len(new_accepted)} additional rows into the database!")
                        st.rerun()

    # Synthetic Producer Utility
    with st.expander("🧪 Synthetic Training Batch Producer (Faker)"):
        st.caption("Produce a fresh synthetic batch of labeled employee records with Python Faker.")
        synth_count = st.slider("Records to generate", min_value=5, max_value=50, value=10, step=5)
        if st.button("Generate & Import Synthetic Batch", use_container_width=True):
            synth_producer = FakerProducer()
            synth_batch = synth_producer.generate_synthetic_batch(count=synth_count)
            synth_accepted, synth_rejected = import_employee_dataframe(synth_batch)
            st.success(f"Generated and imported {len(synth_accepted)} synthetic employee records.")
            st.rerun()

last_import = st.session_state.get("last_import")
if st.session_state.get("repaired_preview") is not None:
    with st.expander("Recently Faker-repaired employee records", expanded=True):
        st.dataframe(st.session_state["repaired_preview"], hide_index=True, use_container_width=True)

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
                            save_inference_predictions(
                                [{
                                    "employee_id": new_id,
                                    "predicted_prob": result["churn_probability"],
                                    "predicted_label": int(result["churn_probability"] >= 0.5),
                                }]
                            )
                            st.write(f"Churn probability: {float(result['churn_probability']):.1%} | Risk: **{result['risk_band']}**")

employees = get_all_employees()
if employees:
    raw_employee_df = pd.DataFrame(employees, columns=["employee_id", "first_name", "last_name", "age", "department", "contract_type", "annual_income", "job_title", "tenure_months"])
    raw_employee_df["employee_name"] = raw_employee_df["first_name"] + " " + raw_employee_df["last_name"]

    # Apply feature engineering and imputation to guarantee zero null/None values
    employee_df = impute_and_engineer_features(raw_employee_df)

    st.header("Employee monitoring")
    with st.container(border=True):
        st.subheader("Employee roster")
        st.dataframe(employee_df[["employee_id", "employee_name", "age", "department", "contract_type", "job_title", "annual_income", "tenure_months"]], hide_index=True, use_container_width=True, height=320)

    st.subheader("Batch churn predictions")
    probabilities, bands = [], []
    prediction_errors = []
    elapsed_scoring_time = 0.0

    try:
        start_score_time = time.perf_counter()
        with st.spinner("Scoring employees with the churn model..."):
            for employee_id in employee_df["employee_id"]:
                if use_remote_api and api_backend_url:
                    result = predict_employee_via_api(int(employee_id), api_backend_url)
                    if result.get("error"):
                        result = predict_employee_locally(int(employee_id))
                else:
                    result = predict_employee_locally(int(employee_id))

                if result.get("error"):
                    prediction_errors.append(f"{employee_id}: {result['error']}")
                probability = None if result.get("error") else float(result["churn_probability"])
                probabilities.append(probability)
                bands.append(None if probability is None else result.get("risk_band", calculate_risk_band(probability)))
        elapsed_scoring_time = time.perf_counter() - start_score_time
    except FileNotFoundError as error:
        st.error(f"Cannot score employees until a trained model is available: {error}")
        probabilities = [None] * len(employee_df)
        bands = [None] * len(employee_df)

    if elapsed_scoring_time > 0:
        st.caption(
            f"⏱️ **Scoring completed in {elapsed_scoring_time:.3f}s** "
            f"({elapsed_scoring_time / max(1, len(employee_df)) * 1000:.1f}ms per employee across {len(employee_df)} records). "
            f"Feature imputation applied — 0 null values."
        )

    if prediction_errors:
        st.warning(f"Some employees could not be scored: {'; '.join(prediction_errors)}")

    employee_df["churn_probability"] = probabilities
    employee_df["risk_band"] = bands

    with st.container(border=True):
        st.dataframe(employee_df[["employee_id", "employee_name", "department", "job_title", "annual_income", "tenure_months", "churn_probability", "risk_band"]].sort_values("churn_probability", ascending=False, na_position="last"), hide_index=True, use_container_width=True, height=360)
        if st.button("Save current batch predictions to history", use_container_width=True):
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
            st.success("Batch predictions saved to prediction history.")
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

# ---------------------------------------------------------
# Feature Selection & Engineering Heatmaps & Influence
# ---------------------------------------------------------
st.divider()
st.header("Feature Selection & Engineering Heatmap")
st.caption("Analyze feature correlations with churn risk and inspect model feature weights to guide feature engineering.")

feat_tab1, feat_tab2 = st.tabs(["🔥 Correlation Heatmap", "📊 Model Feature Influence & Importance"])

with feat_tab1:
    training_data = get_training_data()
    corr_matrix = compute_feature_correlations(training_data)
    if not corr_matrix.empty and "Churn (Target)" in corr_matrix.columns:
        corr_subset_mode = st.radio(
            "Heatmap view:",
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

        st.subheader("Key correlation signals with Churn")
        corr_summary = pd.DataFrame({
            "Feature": churn_corrs.index,
            "Correlation with Churn": churn_corrs.values,
            "Impact": [
                "Increases Churn Risk (Positive Corr)" if v > 0 else "Protects Against Churn (Negative Corr)"
                for v in churn_corrs.values
            ],
            "Absolute Strength": churn_corrs.abs().values,
        }).sort_values("Absolute Strength", ascending=False)
        st.dataframe(corr_summary[["Feature", "Correlation with Churn", "Impact"]].head(12), hide_index=True, use_container_width=True)
    else:
        st.info("No labeled training data available to compute feature correlations.")

with feat_tab2:
    try:
        active_model = get_prediction_model()
        importances_df = get_feature_importances(active_model)
        if not importances_df.empty:
            st.subheader("Model Learned Weights / Feature Influence")
            st.caption("Features with higher absolute impact influence the prediction probability most significantly.")

            st.dataframe(
                importances_df[["Feature", "Absolute Impact", "Direction"]].head(15),
                hide_index=True,
                use_container_width=True,
            )

            # Impact chart
            top_feats = importances_df.head(10).sort_values("Absolute Impact", ascending=True)
            fig_imp, ax_imp = plt.subplots(figsize=(6.5, 3.8))
            colors = ["#d95f02" if "Increases" in str(d) else "#2f6fed" for d in top_feats["Direction"]]
            ax_imp.barh(top_feats["Feature"], top_feats["Absolute Impact"], color=colors)
            ax_imp.set_xlabel("Relative Feature Importance / Weight")
            ax_imp.set_title("Top 10 Most Influential Features on Churn Model")
            fig_imp.tight_layout()
            st.pyplot(fig_imp, use_container_width=True)
            plt.close(fig_imp)
        else:
            st.info("Feature importance data is not available for this model type.")
    except Exception as e:
        st.warning(f"Could not load feature importance: {e}")

# ---------------------------------------------------------
# Hyperparameter Tuning & Model Retraining
# ---------------------------------------------------------
st.divider()
st.header("Hyperparameter Tuning & Retraining")
st.caption("Tune hyperparameters and train new models (Logistic Regression, Random Forest, Gradient Boosting) to improve predictions.")

with st.container(border=True):
    col_alg, col_params = st.columns([1, 2], gap="large")

    with col_alg:
        st.subheader("1. Choose Algorithm")
        algo_choice = st.selectbox(
            "Model Architecture",
            ["Logistic Regression", "Random Forest", "Gradient Boosting"],
        )

    with col_params:
        st.subheader("2. Tune Parameters")
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
            with st.spinner(f"Fitting and evaluating {algo_choice} pipeline..."):
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
st.header("Model dashboard")
with st.container(border=True):
    filter_option = st.selectbox("Date filter", ["All", "Today", "Yesterday", "This week", "Last week", "Custom range"])

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
    start_column, end_column = st.columns(2)
    with start_column:
        custom_start = st.date_input("Start date", value=datetime.now().date())
    with end_column:
        custom_end = st.date_input("End date", value=datetime.now().date())
    start_date, end_date = custom_start, custom_end

if start_date is not None and end_date is not None and start_date > end_date:
    st.error("Start date must be on or before end date.")
else:
    prediction_history = get_prediction_history(
        start_date.isoformat() if start_date is not None else None,
        end_date.isoformat() if end_date is not None else None,
    )
    st.subheader("Prediction history")
    if prediction_history.empty:
        st.info("No saved predictions match this date range. Generate predictions in the app or through the API, then they will appear here.")
    else:
        history_metrics = st.columns(3)
        history_metrics[0].metric("Predictions", f"{len(prediction_history):,}")
        history_metrics[1].metric("Average churn probability", f"{prediction_history['predicted_prob'].mean():.1%}")
        history_metrics[2].metric("Predicted churners", f"{int(prediction_history['predicted_label'].sum()):,}")
        st.dataframe(
            prediction_history[
                ["employee_id", "employee_name", "predicted_prob", "predicted_label", "true_label", "prediction_source", "created_at"]
            ],
            hide_index=True,
            use_container_width=True,
            height=320,
        )

st.subheader("Model evaluation history")
runs_df = get_model_runs()
if runs_df.empty:
    st.info("No model evaluation runs have been recorded yet. Prediction history above does not require a training run.")
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
