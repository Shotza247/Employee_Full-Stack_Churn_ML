# Implementation Plan: Stakeholder-Ready HR Analytics & Modular Tab Interface

## Top-Level Overview
Transform the Streamlit interface into an interactive, stakeholder-ready HR analytics platform. 
The layout will position the **Data Ingestion, Faker Producer, and Employee Entry tools in the persistent Sidebar**, enabling real-time reactivity across **4 clean main-panel tabs**:
1. **Executive HR Insights & Demographics** (with interactive cohort & department drill-down filters)
2. **Workforce Roster & Predictions**
3. **Feature Signals & Retention Drivers**
4. **Model Studio & Evaluation History**

---

## Dashboard Layout Overview

![Spacious Dashboard Layout Architecture](artifacts/images/Spacious%20Dashboard%20Layout%20Architecture.png)

---

## Sub-Tasks

### Sub-Task 1: Sidebar Ingestion & Realtime Faker Data Hub
- **Intent**: Centralize all data ingestion, row repair, synthetic data generation, and single-employee entry in the sidebar so stakeholders can test scenarios from any tab.
- **Expected Outcomes**:
  - Sidebar contains Batch CSV Import, Faker Auto-Repair button, Synthetic Batch Generator, Single Employee Entry form, and Database Quality scorecards.
  - Adding or repairing employees triggers immediate reactive updates across all dashboard tabs without navigating away.
- **Todo List**:
  1. Relocate CSV Batch Uploader and Schema Validation to `st.sidebar`.
  2. Embed the Python Faker Producer Repair button and Synthetic Batch slider in `st.sidebar`.
  3. Relocate the "Add a single employee" form into an expander within `st.sidebar`.
  4. Ensure database state mutations trigger `st.rerun()` to update main-panel tabs.
- **Relevant Context**:
  - `streamlit_app.py:174-340` (Existing sidebar and import logic)
  - `churn_app/faker_producer.py`
  - `churn_app/database.py:import_employee_dataframe`
- **Status**: `[ ] pending`

---

### Sub-Task 2: Build Tab 1 — Executive HR Insights & Demographics (with Interactive Filters)
- **Intent**: Deliver macro KPIs, interactive drill-down filters, and rich demographic exploratory charts comparing churners vs. non-churners across departments, age cohorts, and salary levels.
- **Expected Outcomes**:
  - **Interactive Drill-Down Filter Bar**: Multi-select Department filter, Contract Type filter, and Age/Tenure range sliders to dynamically isolate specific employee segments.
  - **Executive KPI Ribbon**: Filter-aware Active Headcount, Retention Rate (%), Churn Rate (%), Average Salary, and Average Tenure.
  - **Department Breakdown**: Grouped/stacked bar chart comparing retained vs. churned employees per department.
  - **Department Statistics Table**: Department headcount, churn rate %, average salary (retained vs churned), and average tenure (retained vs churned).
  - **Age Cohort Segmentation**: Binned age groups (`< 25`, `25–34`, `35–44`, `45–54`, `55+`) with headcount and churn distribution.
  - **Compensation & Contract Dynamics**: Salary distribution box/histogram and contract type retention impact.
- **Todo List**:
  1. Fetch complete workforce dataframe using `get_training_data()`.
  2. Implement interactive filter bar (Department multiselect, Contract select, Age slider) and filter the active demographic slice.
  3. Compute macro summary metrics for the executive KPI ribbon based on the filtered view.
  4. Build department distribution charts and the department comparison summary table.
  5. Implement age cohort binning (`pd.cut`) and compute age-based retention metrics.
  6. Build salary and contract type retention impact visualizations.
- **Relevant Context**:
  - `streamlit_app.py`
  - `churn_app/database.py:get_training_data`
- **Status**: `[ ] pending`

---

### Sub-Task 3: Build Tab 2 — Workforce Roster & Predictions
- **Intent**: Provide operational employee lookup, batch scoring, latency benchmarking, and risk band distribution.
- **Expected Outcomes**:
  - Searchable and filterable employee roster table.
  - "Predict all employees" button with execution time benchmark (ms/employee).
  - Risk distribution summary and chart (`Low Risk`, `Medium Risk`, `High Risk`).
  - Batch results export / display.
- **Todo List**:
  1. Render the active employee roster table with department and salary details.
  2. Implement batch scoring with the saved model pipeline and inference timing measurement.
  3. Display the risk band breakdown (`High`, `Medium`, `Low`) with the bar chart and proportion table.
- **Relevant Context**:
  - `streamlit_app.py:358-440`
  - `churn_app/prediction.py:predict_employee`
- **Status**: `[ ] pending`

---

### Sub-Task 4: Build Tab 3 — Feature Signals & Retention Drivers
- **Intent**: Visualize statistical correlations and model-learned feature influence weights to explain turnover factors.
- **Expected Outcomes**:
  - Interactive Feature Correlation Heatmap with churn.
  - Top 10 Most Influential Features horizontal bar chart with directional coloring (Increases vs Decreases churn).
  - Key correlation signals summary.
- **Todo List**:
  1. Move correlation matrix computation and heatmap visualization into Tab 3.
  2. Extract feature weights / coefficients via `get_feature_importances(model)`.
  3. Render directional impact chart and analytical takeaways.
- **Relevant Context**:
  - `streamlit_app.py:441-513`
  - `churn_app/training.py:compute_feature_correlations, get_feature_importances`
- **Status**: `[ ] pending`

---

### Sub-Task 5: Build Tab 4 — Model Studio & Evaluation History
- **Intent**: Provide model tuning controls, evaluation tracking across runs, confusion matrices, ROC curves, and audited prediction history.
- **Expected Outcomes**:
  - Algorithm selector (*Logistic Regression*, *Random Forest*, *Gradient Boosting*) with tuned parameter inputs.
  - Model evaluation run table and latest metric cards (Accuracy, Precision, Recall, F1, ROC-AUC).
  - Run inspector with Confusion Matrix, ROC Curve, and Probability Histogram.
  - Date-filtered prediction history log.
- **Todo List**:
  1. Move hyperparameter tuning controls and retrain button into Tab 4.
  2. Embed model evaluation history table and metric cards.
  3. Embed the selected run deep-dive (Confusion matrix, ROC AUC plot, score distributions).
  4. Embed date-filtered prediction history log.
- **Relevant Context**:
  - `streamlit_app.py:514-700`
  - `churn_app/training.py:train_model`
  - `churn_app/database.py:get_model_runs, get_predictions_by_run, get_prediction_history`
- **Status**: `[ ] pending`

---

### Sub-Task 6: Verification & End-to-End Validation
- **Intent**: Verify that all tabs render smoothly, charts update reactively upon adding Faker/synthetic rows, and no warnings/errors occur.
- **Expected Outcomes**:
  - Streamlit app runs without errors or broken components.
  - Sidebar actions immediately update demographic charts in Tab 1 and prediction tables in Tab 2.
- **Todo List**:
  1. Test database loading and quality checks.
  2. Test tab navigation, Tab 1 interactive filters, and chart responsiveness.
  3. Verify Python syntax and run validation checks.
- **Status**: `[ ] pending`
