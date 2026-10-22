# Implementation Summary & Technical Architecture

This document details the complete 5-tab stakeholder architecture, end-to-end data pipeline lineage, storage/auditing governance, currency localization (South African Rand - ZAR / R), and model evaluation tracking across machine learning algorithms.

---

## 1. Executive Implementation Overview: 5-Tab Enterprise Architecture

The Employee Churn Intelligence platform is structured into a streamlined persistent operations sidebar and a 5-tab modular stakeholder hub:

```mermaid
flowchart TB
    SIDEBAR["⚙️ Operations Sidebar<br/>• 🌐 Cloud API / Vercel Switch<br/>• ⚡ Active Model Status Card<br/>• 🧭 5-Tab Quick Navigation"]

    SIDEBAR ==>|"Reactive State"| TABS

    subgraph TABS ["📑 Main Analytics Platform (5 Dedicated Tabs)"]
        direction TB
        
        TAB1["📥 Tab 1: Data Ingestion, Synthesis & Lineage Studio<br/>• Batch CSV Ingestion & Bulk Dataset Downloader<br/>• Real-time Data Quality Diagnosis (Missing, Duplicates, Outliers)<br/>• 🪄 Faker Auto-Repair for Rejected Records<br/>• 🧪 Synthetic Batch Producer (Controllable Churn Bias)<br/>• ➕ Single Employee Manual Ingestion & Instant Scoring<br/>• 🔍 Data Lineage & Provenance Dropdown Explorer<br/>• 📊 SQLite Database Health Scorecard"]

        TAB2["📊 Tab 2: Executive HR Insights & Demographics<br/>• Dynamic Cohort Filters (Department, Contract, Age Range)<br/>• Executive KPI Ribbon (Headcount, Retention Rate %, Churn Rate %, Avg Salary in R, Avg Tenure)<br/>• Department Turnover Breakdown (Retained vs Churned)<br/>• Department Benchmark Statistics Table (Comparative R Salary & Tenure)<br/>• Age Cohort Vulnerability Analysis (<25, 25–34, 35–44, 45–54, 55+)<br/>• Compensation Dynamics & Salary Distributions (R)"]

        TAB3["👥 Tab 3: Workforce Roster & Predictions<br/>• Searchable Active Workforce Roster (Zero-Null Imputation Active)<br/>• 1-Click Batch Scoring with Latency Benchmarking (ms/employee)<br/>• Churn Risk Distribution Breakdown (High ≥70%, Medium 45–69%, Low <45%)"]

        TAB4["🔍 Tab 4: Feature Signals & Retention Drivers<br/>• Feature Correlation Matrix Heatmap<br/>• Directional Model Feature Influence Weights Chart<br/>• Key Retention Driver Takeaways & Risk Factors"]

        TAB5["⚙️ Tab 5: Model Studio & Evaluation History<br/>• Interactive Hyperparameter Tuning (Logistic Regression, Random Forest, Gradient Boosting)<br/>• Multi-Model Evaluation Runs History (Tracking Model Architecture, Accuracy, Precision, Recall, F1, ROC-AUC)<br/>• Run Deep-Dive (Confusion Matrix TP/TN/FP/FN, ROC AUC Curve, Probability Histograms)<br/>• Audited Historical Prediction Log with Date Filtering"]
    end

    classDef sideNode fill:#0F172A,stroke:#38BDF8,stroke-width:2px,color:#ffffff;
    classDef t1Node fill:#0284C7,stroke:#38BDF8,stroke-width:2px,color:#ffffff;
    classDef t2Node fill:#042F2E,stroke:#2DD4BF,stroke-width:2px,color:#ffffff;
    classDef t3Node fill:#1E3A8A,stroke:#60A5FA,stroke-width:2px,color:#ffffff;
    classDef t4Node fill:#312E81,stroke:#A5B4FC,stroke-width:2px,color:#ffffff;
    classDef t5Node fill:#78350F,stroke:#FBBF24,stroke-width:2px,color:#ffffff;

    class SIDEBAR sideNode;
    class TAB1 t1Node;
    class TAB2 t2Node;
    class TAB3 t3Node;
    class TAB4 t4Node;
    class TAB5 t5Node;
```

---

## 2. End-to-End Pipeline Lineage, Storage & Auditing Governance

### Where Data is Saved, Tracked, and Audited Across Stages

```mermaid
flowchart LR
    subgraph STAGE1 ["1. Ingestion & Quality Gate"]
        SRC["Raw Sources:<br/>• Batch CSV<br/>• Faker Synthetic<br/>• Faker Repaired<br/>• Manual Entry"] --> DIAG["Quality Diagnoser<br/>(Categorizes Missing, Duplicates, Outliers)"]
        DIAG -->|"Accepted Records"| RAW_DB[("SQLite employees.db<br/>• employees (source_system, created_at)<br/>• salaries<br/>• jobs")]
    end

    subgraph STAGE2 ["2. Preprocessing & Feature Engineering"]
        RAW_DB --> PREP["Central Tendency Baselines<br/>& Zero-Null Imputation"]
        PREP --> PIPE["ColumnTransformer<br/>• Numeric: Impute + Scale<br/>• Categoric: Impute + OneHot"]
        PIPE --> SNAP["Training Snapshot<br/>(artifacts/training_data.csv)"]
    end

    subgraph STAGE3 ["3. Model Training & Serialization"]
        PIPE --> FIT["Fit Classifier<br/>(LR / RF / GB)"]
        FIT --> PKL["Serialized Pipeline<br/>(churn_model.pkl)"]
        FIT --> RUN_LOG[("SQLite model_runs<br/>+ artifacts/metrics.json")]
        FIT --> TEST_PRED[("SQLite predictions<br/>+ artifacts/batch_predictions.csv")]
    end

    subgraph STAGE4 ["4. Live Inference & Audit Trail"]
        INFER_REQ["Inference Request<br/>(Streamlit / REST API)"] --> PKL
        PKL --> SCORE["Predict Probability & Risk Band"]
        SCORE --> AUDIT_LOG[("SQLite inference_predictions<br/>(employee_id, prob, label, timestamp)")]
    end

    classDef s1Node fill:#0284C7,stroke:#38BDF8,stroke-width:2px,color:#ffffff;
    classDef s2Node fill:#4338CA,stroke:#818CF8,stroke-width:2px,color:#ffffff;
    classDef s3Node fill:#047857,stroke:#34D399,stroke-width:2px,color:#ffffff;
    classDef s4Node fill:#B45309,stroke:#FBBF24,stroke-width:2px,color:#ffffff;

    class SRC,DIAG,RAW_DB s1Node;
    class PREP,PIPE,SNAP s2Node;
    class FIT,PKL,RUN_LOG,TEST_PRED s3Node;
    class INFER_REQ,SCORE,AUDIT_LOG s4Node;
```

### Storage and Audit Tracking Matrix

| Pipeline Stage | Storage Location | Lineage & Audit Identifiers | How to Audit Later |
| :--- | :--- | :--- | :--- |
| **Ingestion** | SQLite `employees`, `salaries`, `jobs` | `employee_id`, `source_system` (`csv_import`, `faker_synthetic`, `faker_repaired`, `manual_entry`, `seed_baseline`), `created_at` | Filter records in **Tab 1: Lineage Explorer** by source system dropdown or run SQL queries on `employees.db`. |
| **Feature Engineering** | In-Memory `impute_and_engineer_features` & `artifacts/training_data.csv` | Feature schema versions, baseline median/mode values computed from database | Inspect `artifacts/training_data.csv` or compare features across historical runs. |
| **Model Evaluation** | SQLite `model_runs`, `predictions` & `artifacts/metrics.json` | `run_id`, `model_type`, `run_timestamp`, holdout precision/recall/F1/ROC-AUC | Query `model_runs` table in **Tab 5: Model Studio** or read `artifacts/metrics.json`. |
| **Model Artifact** | Root `./churn_model.pkl` | Serialized fitted pipeline with preprocessing + classifier steps | Loaded dynamically in `churn_app/prediction.py` with file timestamp check. |
| **Live Inference** | SQLite `inference_predictions` | `prediction_id`, `employee_id`, `predicted_prob`, `predicted_label`, `true_label`, `created_at` | Audit via the date-range filter in **Tab 5: Prediction Audit History** or query `inference_predictions`. |

---

## 3. Multi-Model Architecture & Evaluation Governance

The platform provides explicit model architecture tracking across all evaluation runs, allowing stakeholders to benchmark which classification algorithms achieve the highest discriminatory power for their workforce:

### Supported Algorithms & Hyperparameter Spaces:
1. **Logistic Regression (Linear / Interpretable)**:
   - **Regularization Strength ($C$)**: Tunable from $0.01$ to $10.0$.
   - **Class Weighting**: `None` or `balanced` (penalizes minority churn misclassifications).
   - **Solvers**: `lbfgs`, `saga`, `liblinear`.
2. **Random Forest (Ensemble / Non-Linear)**:
   - **Estimators ($n$)**: $20$ to $300$ trees.
   - **Max Depth**: $2$ to $15$ levels.
   - **Min Samples Split**: $2$, $5$, $10$.
3. **Gradient Boosting (Boosting / Gradient Trees)**:
   - **Learning Rate ($\eta$)**: $0.01$ to $0.30$.
   - **Estimators ($n$)**: $20$ to $200$.
   - **Max Tree Depth**: $2$ to $8$.

### Model Evaluation Run Tracking Schema:
Every retraining event logs the specific model architecture alongside its holdout test metrics:

| Field | Type | Description |
| :--- | :--- | :--- |
| `run_id` | `INTEGER PRIMARY KEY` | Auto-incrementing identifier for the training event. |
| `model_architecture` | `TEXT` | Algorithm used (`Logistic Regression`, `Random Forest`, `Gradient Boosting`). |
| `accuracy` | `REAL` | Overall classification correctness on holdout split ($20\%$). |
| `precision` | `REAL` | Proportion of predicted churners who actually churned ($TP / (TP + FP)$). |
| `recall` | `REAL` | Proportion of actual churners correctly caught ($TP / (TP + FN)$). |
| `f1` | `REAL` | Harmonic mean of precision and recall ($2 \cdot (P \cdot R) / (P + R)$). |
| `roc_auc` | `REAL` | Area Under the Receiver Operating Characteristic curve. |
| `run_timestamp` | `TIMESTAMP` | ISO-8601 audit timestamp. |

---

## 4. South African Rand (ZAR / R) Localization

All financial computations, data quality rules, salary analytics, and prediction inputs are localized for the South African market:

1. **Compensation Metrics**:
   - Macro average compensation formatted as `R XX,XXX`.
   - Department salary benchmarks: `Avg Sal (Retained)` and `Avg Sal (Churned)` displayed in Rands.
   - Salary Disparity analysis across departments and contract tiers.
2. **Input Validation**:
   - Single-employee and batch annual income accepts South African Rand figures (e.g. `R 52,000.00` to `R 1,500,000.00+`).
3. **Visual Analytics**:
   - Histogram distribution charts calibrated to Rand salary ranges.
