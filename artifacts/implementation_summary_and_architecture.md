# Implementation Summary & Technical Architecture

This document serves as the implementation companion to `detailed_tab_breakdown.md`, detailing the architecture, currency localization (South African Rand - ZAR / R), model evaluation tracking by algorithm, and the complete 4-tab stakeholder interface.

---

## 1. Executive Implementation Overview

The Employee Churn Intelligence platform has been engineered specifically for HR analysts, people managers, and executive stakeholders in South Africa. All currency metrics, salary visualizations, and baseline compensation models are standardized in **South African Rands (R / ZAR)**.

```mermaid
flowchart TB
    SIDEBAR["📥 Persistent Sidebar (Data Hub & Faker Studio)<br/>• Batch CSV Ingestion & Schema Checks<br/>• 🪄 Faker Auto-Repair & Synthetic Batch Generator<br/>• ➕ Single Employee Entry with Instant Scoring<br/>• 📊 Live Database Health Tracker<br/>• 🌐 Cloud API / Vercel Switch"]

    SIDEBAR ==>|"⚡ Real-time Reactive State Updates"| TABS

    subgraph TABS ["📑 Main Analytics Platform (4 Modular Tabs)"]
        direction TB
        
        TAB1["📊 Tab 1: Executive HR Insights & Demographics<br/>• Dynamic Cohort Filters (Department, Contract, Age Range)<br/>• Executive KPI Ribbon (Headcount, Retention Rate %, Churn Rate %, Avg Salary in R, Avg Tenure)<br/>• Department Turnover Breakdown (Retained vs Churned)<br/>• Department Benchmark Statistics Table (Comparative R Salary & Tenure)<br/>• Age Cohort Vulnerability Analysis (<25, 25–34, 35–44, 45–54, 55+)<br/>• Compensation Dynamics & Salary Distributions (R)"]

        TAB2["👥 Tab 2: Workforce Roster & Predictions<br/>• Searchable Active Workforce Roster (Zero-Null Imputation Active)<br/>• 1-Click Batch Scoring with Latency Benchmarking (ms/employee)<br/>• Churn Risk Distribution Breakdown (High ≥70%, Medium 45–69%, Low <45%)"]

        TAB3["🔍 Tab 3: Feature Signals & Retention Drivers<br/>• Feature Correlation Matrix Heatmap<br/>• Directional Model Feature Influence Weights Chart<br/>• Key Retention Driver Takeaways & Risk Factors"]

        TAB4["⚙️ Tab 4: Model Studio & Evaluation History<br/>• Interactive Hyperparameter Tuning (Logistic Regression, Random Forest, Gradient Boosting)<br/>• Multi-Model Evaluation Runs History (Tracking Model Architecture, Accuracy, Precision, Recall, F1, ROC-AUC)<br/>• Run Deep-Dive (Confusion Matrix TP/TN/FP/FN, ROC AUC Curve, Probability Histograms)<br/>• Audited Historical Prediction Log with Date Filtering"]
    end

    classDef sideNode fill:#1E293B,stroke:#38BDF8,stroke-width:2px,color:#ffffff;
    classDef t1Node fill:#0F766E,stroke:#2DD4BF,stroke-width:2px,color:#ffffff;
    classDef t2Node fill:#1D4ED8,stroke:#60A5FA,stroke-width:2px,color:#ffffff;
    classDef t3Node fill:#4338CA,stroke:#818CF8,stroke-width:2px,color:#ffffff;
    classDef t4Node fill:#B45309,stroke:#FBBF24,stroke-width:2px,color:#ffffff;

    class SIDEBAR sideNode;
    class TAB1 t1Node;
    class TAB2 t2Node;
    class TAB3 t3Node;
    class TAB4 t4Node;
```

---

## 2. Multi-Model Architecture & Evaluation Governance

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

## 3. South African Rand (ZAR / R) Localization

All financial computations, data quality rules, salary analytics, and prediction inputs are localized for the South African market:

1. **Compensation Metrics**:
   - Macro average compensation formatted as `R XX,XXX`.
   - Department salary benchmarks: `Avg Sal (Retained)` and `Avg Sal (Churned)` displayed in Rands.
   - Salary Disparity analysis across departments and contract tiers.
2. **Input Validation**:
   - Single-employee and batch annual income accepts South African Rand figures (e.g. `R 52,000.00` to `R 1,500,000.00+`).
3. **Visual Analytics**:
   - Histogram distribution charts calibrated to Rand salary ranges.
