# Detailed 5-Tab Breakdown & Workforce Analytics Specification

This specification document outlines the comprehensive functional architecture, visual components, data schemas, and analytics metrics for the 5-tab Streamlit dashboard and streamlined operations sidebar.

---

## 1. ⚙️ Streamlined Operations Sidebar

The sidebar provides dedicated operational controls and system status monitoring without cluttering data workflows:

### Components & Functional Requirements:
1. **Cloud API Backend Switch**:
   - Radio toggle between *Local Predictor (In-Process)* and *FastAPI REST Client (Cloud / Vercel)*.
   - Dynamic backend endpoint URL configuration input (e.g. `https://your-app.vercel.app` or `http://127.0.0.1:8000`).
   - Connection test trigger with instant latency and status reporting.
2. **Active Model Status Badge**:
   - Displays current model architecture, holdout ROC-AUC score, and F1 performance.
3. **5-Tab Navigation Guide**:
   - Clean architectural overview of the platform's analytical capabilities.

---

## 2. 📥 Tab 1: Data Ingestion, Synthesis & Lineage Studio

Centralizes all enterprise data ingestion workflows, automated row-level data quality diagnosis, Faker auto-repair, and full-width lineage exploration:

### Visualizations & Interactive Workflows:
1. **Segmented Ingestion Mode Selector (`st.radio(horizontal=True)`)**:
   - **Mode 1: 📥 Batch CSV Import & Auto-Repair**:
     - Pre-packaged test dataset downloader (`data/Bulk_import_Data/`).
     - CSV workforce roster uploader with schema validation.
     - Full-width Data Quality Assessment scorecard (Clean Records, Missing Values, Duplicate Records, Invalid Range Outliers, Invalid Categories).
     - Diagnosed preview table displaying row-level issue badges.
     - Duplicate file re-import blocker (`is_source_imported`) protecting data integrity.
     - Post-import summary with rejected rows breakdown.
     - **🪄 1-Click Faker Auto-Repair**: Synthesizes clean identities and resolves duplicates, tagging repaired records with `faker_repaired (<filename>.csv)`.
   - **Mode 2: 🎲 Synthetic Batch Generator (Faker)**:
     - Configurable batch size (5–100) and churn signal bias sliders (5%–80%).
     - 1-click batch producer inserting directly into SQLite with provenance `faker_synthetic`.
   - **Mode 3: 👤 Single Employee Manual Entry**:
     - Streamlined 2-column manual form with localized South African Rand (R / ZAR) compensation.
     - On insertion: immediate local ML risk scoring and SQLite audit logging (`source_system="manual_entry"`).
2. **🔍 Data Lineage & Provenance Explorer (Full-Width)**:
   - Dynamic dropdown filter querying records by exact source system (`<filename>.csv`, `faker_repaired (<filename>.csv)`, `faker_synthetic`, `manual_entry`, `seed_baseline`).
   - Metric ribbon tracking total records and filtered count.
   - Source distribution breakdown table with percentage allocations.
   - Audited records table showing employee details, source origin, and ISO timestamp.
3. **📊 SQLite Database Health & Schema Inspector**:
   - Database health scorecard verifying database-wide data quality and schema constraints.

---

## 3. 📊 Tab 2: Executive HR Insights & Demographics

Executive analytics hub providing filter-aware demographic analysis, turnover drivers, and compensation benchmarking in South African Rands (ZAR / R):

### Visualizations & KPI Metrics:
1. **Interactive Cohort Drill-Down Filter Bar**:
   - Multiselect Department Filter (*Marketing, Sales, Engineering, HR, Support, Operations, Finance*).
   - Multiselect Contract Type Filter (*permanent, contract*).
   - Age Range Slider (`18` to `99`).
2. **Executive Macro KPI Ribbon**:
   - **👥 Headcount**: Filtered active employee cohort size.
   - **🛡️ Retention Rate**: Overall percentage of retained employees.
   - **⚠️ Churn Rate**: Overall percentage of departing employees.
   - **💵 Avg Annual Salary**: Mean annual compensation formatted in Rands (`R XX,XXX`).
   - **⏱️ Avg Tenure**: Mean months of service.
3. **🏢 Department Turnover Distribution (Interactive Plotly Stacked Bar)**:
   - Interactive stacked bar chart comparing retained vs. churned staff per department with exact headcounts on hover.
4. **📋 Department Benchmark Statistics Table**:
   - Comparative metrics table displaying Headcount, Churn %, Average Annual Salary (Retained vs. Churned in R), and Average Tenure.
5. **👥 Age Cohort Vulnerability Analysis (Interactive Plotly Grouped Bar)**:
   - Demographic segmentation across 5 cohorts (`< 25`, `25–34`, `35–44`, `45–54`, `55+`) with hover headcount statistics.
6. **📝 Contract Type & Employment Dynamics (Interactive Plotly Horizontal Bar)**:
   - Retention and attrition comparison across permanent and contract employment tiers.
7. **💵 Dedicated Salary Distribution Row (Retained vs. Churned)**:
   - **Interactive Plotly Histogram Overlay**: Comparative annual salary distribution in Rands with hover tooltips.
   - **📊 Salary Benchmark Statistics Card**: Side-by-side summary comparing Mean, Median, Min, and Max annual salaries, plus automatic compensation disparity analysis.

---

## 4. 👥 Tab 3: Workforce Roster & Predictions

Operational scoring cockpit featuring zero-null feature imputation, millisecond batch benchmarking, and real-time evaluation scorecards:

### Visualizations & Prediction Operations:
1. **Active Workforce Roster Table**:
   - Searchable, sortable roster table with zero-null feature imputation.
2. **1-Click Batch Churn Scoring**:
   - Millisecond latency benchmark badge (`ms/employee`).
3. **📊 Batch Prediction Evaluation Scorecard**:
   - Automatic evaluation against labeled ground-truth records in the active roster:
     - 🎯 **Batch Accuracy** ($TP + TN / Total$)
     - 🔍 **Batch Precision** ($TP / (TP + FP)$)
     - 🎣 **Batch Recall** ($TP / (TP + FN)$)
     - ⚖️ **Batch F1-Score** ($2 \cdot (P \cdot R) / (P + R)$)
     - 📈 **Batch ROC-AUC**
   - **Interactive Batch Confusion Matrix**: Plotly heatmap showing exact True Positives, True Negatives, False Positives, and False Negatives.
   - **Classification Outcomes Table**: Proportions and counts of classification results.
4. **Scored Predictions Table with Ground Truth**:
   - Displays `employee_id`, `employee_name`, `department`, `job_title`, `annual_income`, `tenure_months`, `churn_probability`, `risk_band`, and `ground_truth`.
   - **💾 Save Batch Predictions to History**: Persists scored inferences to SQLite `inference_predictions`.
5. **🎯 Churn Risk Distribution (Interactive Plotly Bar)**:
   - Categorization into **High Risk (≥70%)**, **Medium Risk (45–69%)**, and **Low Risk (<45%)** with exact headcount shares.

---

## 5. 🔍 Tab 4: Feature Signals & Retention Drivers

Statistical explainability and model-learned feature weight analysis:

### Visualizations & Explanations:
1. **🔥 Interactive Feature Correlation Heatmap**:
   - Interactive Plotly heatmap with cell-level correlation annotations and hover tooltips.
   - View modes: *Top Correlated with Churn* and *Numeric Features*.
2. **Key Retention Signals Table**:
   - Features ranked by absolute correlation magnitude with directional turnover indicators.
3. **📊 Model Learned Feature Influence Weights**:
   - Interactive horizontal Plotly bar chart displaying trained model coefficients / feature importances.
   - Color-coded: Red for features increasing churn risk; Teal for features promoting retention.

---

## 6. ⚙️ Tab 5: Model Studio & Evaluation History

Enterprise model governance, hyperparameter tuning, and historical prediction audit trails:

### Visualizations & Governance Tools:
1. **Interactive Hyperparameter Tuning**:
   - Multi-algorithm support:
     - **Logistic Regression**: Regularization strength $C$, class weighting (`None` vs `balanced`), solvers (`lbfgs`, `saga`, `liblinear`).
     - **Random Forest**: Estimator tree count ($20–300$), max depth ($2–15$), min samples split.
     - **Gradient Boosting**: Learning rate ($0.01–0.30$), estimators, max depth.
   - **🚀 Train & Evaluate Model**: Stratified $80/20$ holdout evaluation, SQLite run logging, and frozen artifact generation (`churn_model.pkl`).
2. **Model Evaluation Runs Comparison Table**:
   - Metric comparison table tracking `run_id`, `model_architecture`, `accuracy`, `precision`, `recall`, `f1`, and `roc_auc`.
3. **Run Deep-Dive Analytics**:
   - **Interactive Confusion Matrix Heatmap**: TP, TN, FP, FN counts.
   - **Interactive Plotly ROC Curve**: True Positive Rate vs False Positive Rate with stored AUC.
4. **⏱️ Audited Historical Prediction Log**:
   - Date range filter (*All, Today, Yesterday, This week, Last week, Custom range*).
   - Timestamped query log of all scoring events with predicted probabilities and labels.

---

*Documentation and architecture diagrams created via IBM Bob.*
