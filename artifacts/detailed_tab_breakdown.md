# Detailed Tab Breakdown & Workforce Analytics Specification

This specification document outlines the comprehensive functional requirements, visual components, data schemas, and analytics metrics for the modular Streamlit interface and persistent Sidebar data hub.

---

## 1. Persistent Sidebar: Data Ingestion & Simulation Studio

The sidebar acts as a persistent ingestion and scenario testing cockpit available across all main-panel tabs.

### Components & Functional Requirements:
1. **API / Local Inference Switch**:
   - Radio toggle between *Local Predictor (In-Process)* and *FastAPI REST Client*.
   - Dynamic endpoint URL configuration input (`http://127.0.0.1:8000`).
2. **Batch CSV Import & Schema Validator**:
   - File uploader supporting `.csv` employee rosters.
   - Live schema verification checking for 8 mandatory columns: `first_name`, `last_name`, `age`, `department`, `contract_type`, `annual_income`, `job_title`, `tenure_months` (and optional `churn`).
   - Rejection reporting displaying detailed validation errors and duplicate flags.
3. **🪄 Python Faker Producer & 1-Click Auto-Repair**:
   - One-click trigger synthesizing valid identities and imputing missing fields for rejected rows.
   - Automatically inserts repaired records into the SQLite database.
4. **🧪 Synthetic Batch Generator**:
   - Slider control allowing generation of 5 to 50 synthetic employee records on demand using `FakerProducer`.
   - Generates realistic demographic profiles, salaries, and optional churn labels, immediately writing them to SQLite.
5. **➕ Add Single Employee Form**:
   - Expandable form with input validation for ad-hoc employee insertion.
   - Immediately returns the new assigned `employee_id` and triggers local risk scoring.
6. **📊 Live Data Quality Scorecards**:
   - Dual-card metric container comparing uploaded CSV quality against current SQLite database integrity (Total rows, Missing values, Duplicates, Invalid value counts).

---

## 2. Tab 1: 📊 Executive HR Insights & Demographics

Designed specifically for HR Executives and Business Stakeholders to explore workforce macro trends and understand turnover drivers.

### Visualizations & KPI Metrics:
1. **Interactive Cohort Drill-Down Filter Bar**:
   - Multiselect Department Filter (*Marketing, Sales, Engineering, HR, Support, Operations, Finance*).
   - Multiselect Contract Type Filter (*permanent, contract*).
   - Age Range Slider (`18` to `99`).
   - Tenure Range Slider (`0` to `240` months).
   - All downstream Tab 1 charts and metrics reactively update based on the active filter slice.
2. **Executive Macro KPI Ribbon**:
   - **Total Headcount**: Filtered active employee count.
   - **Company Retention Rate**: Percentage of non-churned employees (`(1 - churn_rate) * 100`).
   - **Churn Rate**: Percentage of labeled churned employees.
   - **Average Annual Salary**: Mean compensation across the cohort.
   - **Average Tenure**: Mean months of service.
3. **Department Turnover Distribution Chart**:
   - Grouped / stacked bar chart comparing **Retained Employees (churn = 0)** vs. **Churned Employees (churn = 1)** across all departments.
   - Department retention risk ranking (identifying highest-turnover departments).
4. **Department Benchmark Statistics Table**:
   - Comprehensive comparative summary table with columns:
     - `Department`
     - `Headcount`
     - `Churn Rate (%)`
     - `Avg Salary (Retained)`
     - `Avg Salary (Churned)`
     - `Salary Disparity (%)`
     - `Avg Tenure (Retained)`
     - `Avg Tenure (Churned)`
5. **Age Cohort Vulnerability Analysis**:
   - Demographic segmentation into 5 standardized cohorts:
     - `< 25` (Entry level)
     - `25–34` (Early career)
     - `35–44` (Mid-career)
     - `45–54` (Senior career)
     - `55+` (Pre-retirement)
   - Visualizing cohort headcount and churn rate per age bracket.
6. **Compensation & Contract Retention Dynamics**:
   - Comparative salary distribution analysis for churners vs. non-churners.
   - Retention rates segmented by Contract Type (*Permanent vs. Contract*).

---

## 3. Tab 2: 👥 Workforce Roster & Predictions

Operational employee lookup and inference scoring engine.

### Visualizations & Features:
1. **Live Workforce Roster**:
   - Interactive, searchable, and sortable table displaying all employees currently stored in the SQLite database.
2. **1-Click Batch Inference Engine**:
   - Button to score all imported/active employees with the current fitted `churn_model.pkl`.
   - Real-time scoring latency benchmark badge reporting total elapsed time (seconds) and average per-employee latency (milliseconds/employee).
3. **Batch Prediction Results Table**:
   - Displays `employee_id`, `employee_name`, `department`, `job_title`, `annual_income`, `churn_probability`, and assigned `risk_band`.
4. **Churn Risk Distribution Summary**:
   - Risk categorization breakdown:
     - **High Risk**: Churn probability $\ge 0.70$ (Red alert badge)
     - **Medium Risk**: Churn probability between $0.45$ and $0.69$ (Amber warning badge)
     - **Low Risk**: Churn probability $< 0.45$ (Green stable badge)
   - Interactive bar chart and share percentage summary table.

---

## 4. Tab 3: 🔍 Feature Signals & Retention Drivers

Explainability and statistical analysis of what features drive employee turnover.

### Visualizations & Explanations:
1. **Correlation Matrix Heatmap**:
   - Pearson correlation heatmap between numeric employee attributes (`age`, `annual_income`, `tenure_months`) and historical `churn`.
   - Color scale highlighting strong positive (turnover risk) and negative (retention factor) correlations.
2. **Model Feature Influence Weights (Feature Importance)**:
   - Directional horizontal bar chart showing top 10 most influential model features (extracted from logistic regression coefficients or tree importance).
   - Color-coded: Orange for features increasing churn risk; Blue for features promoting retention.
3. **Key Retention Drivers Takeaways**:
   - Structured analytical cards summarizing primary turnover risks (e.g., tenure vulnerability thresholds, salary market parity).

---

## 5. Tab 4: ⚙️ Model Studio & Evaluation History

Machine learning pipeline governance, hyperparameter tuning, and historical audit trail.

### Visualizations & Controls:
1. **Interactive Hyperparameter Tuning**:
   - Algorithm selection:
     - **Logistic Regression**: Regularization $C$, Class Weighting (`None` vs `balanced`), Solvers (`lbfgs`, `saga`, `liblinear`), Max Iterations.
     - **Random Forest**: Estimator trees, Max depth, Min samples split, Class Weighting.
     - **Gradient Boosting**: Learning rate, Estimator count, Max depth.
   - **🚀 Train & Evaluate Model**: Retrains the pipeline on updated SQLite records, creates a new `run_id`, updates `churn_model.pkl`, and immediately records metrics.
2. **Model Evaluation Runs Table**:
   - Top metric summary cards showing latest run: **Accuracy**, **Precision**, **Recall**, **F1 Score**, **ROC-AUC**.
   - Historical model run comparison table logging all previous training iterations.
3. **Run Deep-Dive Analytics**:
   - Run selection inspector dropdown.
   - Actual vs. Predicted classification bar chart.
   - Receiver Operating Characteristic (ROC) curve plot with holdout AUC metric.
   - Confusion Matrix breakdown with True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
   - Prediction probability distribution histogram.
4. **Audited Prediction History Log**:
   - Date range filter (*All, Today, Yesterday, This week, Last week, Custom range*).
   - Prediction history table logging timestamps, scored employee IDs, probabilities, predicted labels, true labels, and prediction source (*Local Streamlit* vs *FastAPI*).
