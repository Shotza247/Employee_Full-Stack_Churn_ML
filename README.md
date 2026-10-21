# 🏢 Employee Churn Intelligence & Full-Stack ML Serving Platform

An enterprise-ready machine learning and workforce analytics platform that predicts employee turnover risk, explores demographic drivers, and serves predictions through an interactive 4-tab Streamlit dashboard and a serverless FastAPI backend.

All financial metrics, salary benchmarks, and compensation modeling are standardized in **South African Rands (R / ZAR)**.

---

## 📌 Project at a Glance

```mermaid
flowchart TB
    subgraph INGEST ["📥 Ingestion & Data Hub (Sidebar)"]
        direction LR
        CSV["📄 Batch CSV Upload"]
        FAKER["🪄 Python Faker Auto-Repair"]
        MANUAL["➕ Single Employee Entry"]
    end

    subgraph STORAGE ["🗄️ Persistence Layer (SQLite)"]
        DB[("employees.db<br/>• Employee Demographics & Salaries (ZAR / R)<br/>• Model Runs & Evaluation History<br/>• Audited Prediction Logs")]
    end

    subgraph ML_CORE ["🧠 Machine Learning Engine"]
        direction LR
        IMPUTE["🛡️ Zero-Null Feature Imputation"]
        TRAIN["🏋️ Stratified Training Pipeline<br/>(Logistic Regression / Random Forest / Gradient Boosting)"]
        MODEL["📦 Versioned Frozen Artifact<br/>(churn_model.pkl)"]
    end

    subgraph SERVING ["⚡ Dual Inference & Serving Infrastructure"]
        direction LR
        ST_PREDICT["💻 Streamlit Local Predictor<br/>(In-Memory Scoring)"]
        FAST_API["🚀 Serverless FastAPI Backend<br/>(REST POST /predict on Vercel)"]
        RESULT["🎯 Risk Probability & Latency<br/>(High ≥70% / Med 45-69% / Low <45%)"]
    end

    subgraph DASHBOARD ["📊 Modular 4-Tab Analytics Platform"]
        direction TB
        T1["📊 Tab 1: Executive HR Insights & Demographics (R / ZAR)"]
        T2["👥 Tab 2: Workforce Roster & Batch Scoring (ms/emp)"]
        T3["🔍 Tab 3: Feature Signals & Retention Drivers"]
        T4["⚙️ Tab 4: Model Studio & Evaluation History"]
    end

    CSV -->|"Schema Validate & Import"| DB
    FAKER -->|"Stress Test / Auto-Fix"| DB
    MANUAL -->|"Direct Ingestion"| DB

    DB -->|"Labeled Historical Records"| TRAIN
    TRAIN -->|"Export Trained Pipeline"| MODEL
    TRAIN -->|"Log Metrics by Architecture"| DB

    DB -->|"Workforce Records"| IMPUTE
    IMPUTE --> ST_PREDICT
    IMPUTE --> FAST_API
    MODEL --> ST_PREDICT
    MODEL --> FAST_API

    ST_PREDICT --> RESULT
    FAST_API --> RESULT

    RESULT --> DASHBOARD
    DB --> DASHBOARD

    classDef ingestNode fill:#0F172A,stroke:#38BDF8,stroke-width:2px,color:#ffffff;
    classDef dbNode fill:#042F2E,stroke:#2DD4BF,stroke-width:2px,color:#ffffff;
    classDef mlNode fill:#1E3A8A,stroke:#60A5FA,stroke-width:2px,color:#ffffff;
    classDef modelNode fill:#78350F,stroke:#FBBF24,stroke-width:2px,color:#ffffff;
    classDef serveNode fill:#312E81,stroke:#A5B4FC,stroke-width:2px,color:#ffffff;
    classDef resultNode fill:#064E3B,stroke:#34D399,stroke-width:2px,color:#ffffff;
    classDef tabNode fill:#1E293B,stroke:#94A3B8,stroke-width:2px,color:#ffffff;

    class CSV,FAKER,MANUAL ingestNode;
    class DB dbNode;
    class IMPUTE,TRAIN mlNode;
    class MODEL modelNode;
    class ST_PREDICT,FAST_API serveNode;
    class RESULT resultNode;
    class T1,T2,T3,T4 tabNode;
```

---

## ✨ Core Features & Platform Capabilities

- **📊 Modular 4-Tab HR Analytics Dashboard:** Cleanly organized into Executive HR Insights, Workforce Roster Scoring, Feature Signals, and Model Governance.
- **🎯 Dynamic Cohort Drill-Down Filters:** Filter demographic charts and KPI cards by Department, Contract Type, Age, and Tenure in real time.
- **🏢 Department Turnover & Salary Disparity Analytics:** Side-by-side benchmark tables and distribution charts comparing retained vs. churned employees across departments.
- **👥 Age Cohort Vulnerability Analysis:** Cohort analysis across 5 age brackets (`< 25`, `25–34`, `35–44`, `45–54`, `55+`) identifying turnover patterns.
- **🪄 Python Faker Producer & 1-Click Repair:** Synthesizes realistic identities and repairs rejected CSV records automatically without poisoning model weights.
- **🧪 Synthetic Batch Generator:** Injects configurable stress-test batches (5 to 50 records) directly into SQLite on demand.
- **🛡️ Zero-Null Feature Imputation:** Imputes missing numerical and categorical features prior to scoring using database-learned central tendencies.
- **🎛️ Multi-Model Hyperparameter Tuning:** Interactively tune and compare **Logistic Regression**, **Random Forest**, and **Gradient Boosting** pipelines with full metric history tracking by model architecture.
- **⚡ Millisecond Prediction Latency:** Measures real-time scoring benchmark latency (ms/employee) during batch scoring.
- **🚀 Dual-Serving Infrastructure:** Score employees locally via Streamlit or serve predictions over public REST API via FastAPI on Vercel Serverless.

---

## 🚀 Quick Start Guide

### 1. Environment Setup

Open PowerShell in the project directory:

```powershell
py -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Initialize and Seed SQLite Database

```powershell
python -c "from churn_app.database import init_db, populate_sample_data; init_db(); populate_sample_data()"
```

This creates the SQLite tables (`employees`, `salaries`, `jobs`, `model_runs`, `predictions`, `inference_predictions`) and inserts baseline sample records.

### 3. Train the Model Pipeline

```powershell
python -m churn_app.train_and_evaluate
```

This trains the scikit-learn pipeline, generates `churn_model.pkl`, and exports evaluation reports to `artifacts/`:

| Artifact | Description |
| :--- | :--- |
| `churn_model.pkl` | Serialized preprocessing pipeline and fitted classifier |
| `artifacts/metrics.json` | Holdout accuracy, precision, recall, F1, and ROC-AUC |
| `artifacts/training_data.csv` | Labeled rows used in supervised training |
| `artifacts/batch_predictions.csv` | Holdout predictions generated during evaluation |

### 4. Run the Streamlit Dashboard

```powershell
python -m streamlit run streamlit_app.py
```

Open `http://localhost:8501` to access the interactive dashboard.

### 5. (Optional) Run the FastAPI Service Locally

```powershell
python -m uvicorn churn_app.api:app --reload --host 127.0.0.1 --port 8000
```

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

---

## 📑 4-Tab Dashboard Architecture

![Spacious Dashboard Layout Architecture](artifacts/images/Spacious%20Dashboard%20Layout%20Architecture.png)

### 📥 Persistent Sidebar: Ingestion & Simulation Studio
- **Cloud API Switch:** Route predictions locally or to a deployed Vercel FastAPI backend.
- **Batch CSV Import:** Previews uploaded rosters, verifies 8 required column schemas, provides direct download buttons for repository test datasets (`data/Bulk_import_Data/`), and displays data quality scorecards.
- **🪄 Faker Auto-Repair:** One-click synthetic identity fix for unapproved/rejected rows.
- **🧪 Synthetic Batch Generator:** Generates 5–50 realistic employee rows for testing.
- **➕ Add Single Employee:** Instant employee creation with automated real-time scoring.
- **📊 SQLite Database Health:** Real-time scorecard tracking duplicate, missing, and invalid records.

### 📊 Tab 1: Executive HR Insights & Demographics
- **Dynamic Filter Bar:** Interactive multi-select filters for Departments, Contract Types, and Age Range.
- **Macro KPI Ribbon:** Filter-aware Headcount, Retention Rate (%), Churn Rate (%), Average Salary (R), and Average Tenure (months).
- **Department Turnover Distribution:** Stacked bar chart comparing Retained vs. Churned employees by department.
- **Department Benchmark Table:** Side-by-side comparative table showing Headcount, Churn %, Average Salary (Retained vs. Churned), and Average Tenure.
- **Age Cohort Vulnerability Analysis:** Cohort segmentation (`< 25`, `25–34`, `35–44`, `45–54`, `55+`) with retention vs. churn bars.
- **Compensation & Contract Dynamics:** Comparative salary distribution histograms (R) and contract type retention summaries.

### 👥 Tab 2: Workforce Roster & Predictions
- **Active Workforce Roster:** Searchable and sortable roster table with zero-null feature imputation.
- **1-Click Batch Scoring:** Score all roster employees with millisecond-per-employee latency benchmarking.
- **Risk Categorization Breakdown:** Categorization into **High Risk** ($\ge 70\%$), **Medium Risk** ($45-69\%$), and **Low Risk** ($< 45\%$) with interactive distribution chart and proportion table.

### 🔍 Tab 3: Feature Signals & Retention Drivers
- **Correlation Heatmap Matrix:** Visual correlation matrix between features and turnover risk.
- **Ranked Key Signals Table:** Features ranked by absolute influence with directional risk tags.
- **Model Learned Feature Influence:** Directional horizontal bar chart indicating features promoting retention vs. increasing turnover risk.

### ⚙️ Tab 4: Model Studio & Evaluation History
- **Interactive Hyperparameter Tuning:** Tune and train **Logistic Regression**, **Random Forest**, or **Gradient Boosting** models in real time.
- **Model Evaluation Runs History:** Audit table logging `run_id`, `model_architecture`, `accuracy`, `precision`, `recall`, `f1`, and `roc_auc`.
- **Run Deep-Dive Analytics:** Confusion Matrix (TP, TN, FP, FN), Stored ROC Curve with holdout AUC, and Prediction Distribution Histogram.
- **Audited Prediction Log:** Date-filtered historical log of all scoring events with ground-truth verification.

---

## 📐 Data Pipeline & Model Architecture

### How the Data Pipeline Works

```mermaid
flowchart TD
    A["Select CSV"] --> B["Check required columns"]
    B --> C{"All columns present?"}
    C -- No --> D["Show schema error"]
    C -- Yes --> E["Preview data & quality summary"]
    E --> F["User selects Import valid rows"]
    F --> G["Validate each record"]
    G --> H{"Valid & unique?"}
    H -- No --> I["Flag rejected row with reason"]
    I --> J["Repair via Python Faker Producer"]
    J --> F
    H -- Yes --> K["Insert employee, salary, and job data"]
    K --> L["Display accepted summary & score predictions"]

    classDef actionNode fill:#1D4ED8,stroke:#60A5FA,stroke-width:2px,color:#ffffff;
    classDef decisionNode fill:#B45309,stroke:#FBBF24,stroke-width:2px,color:#ffffff;
    classDef errorNode fill:#B91C1C,stroke:#F87171,stroke-width:2px,color:#ffffff;
    classDef successNode fill:#047857,stroke:#34D399,stroke-width:2px,color:#ffffff;

    class A,B,E,F,G,J,K actionNode;
    class C,H decisionNode;
    class D,I errorNode;
    class L successNode;
```

<details>
<summary><strong>📊 Click to toggle Data Pipeline UML Activity Diagram</strong></summary>

```mermaid
flowchart TB
    start((Start)) --> upload["Upload CSV"]
    upload --> schema{"Required columns present"}
    schema -- No --> error["Display schema error"]
    schema -- Yes --> preview["Preview rows and quality"]
    preview --> confirm["User confirms import"]
    confirm --> validate["Validate row values"]
    validate --> duplicate{"Valid and unique"}
    duplicate -- No --> reject["Record rejection reason"]
    duplicate -- Yes --> normalize["Convert missing feature values to NULL"]
    normalize --> persist["Write to SQLite tables"]
    reject --> summary["Display accepted and rejected rows"]
    persist --> summary
    summary --> finish((Done))
    error --> finish

    classDef terminalNode fill:#0F172A,stroke:#38BDF8,stroke-width:2px,color:#ffffff;
    classDef processNode fill:#1E3A8A,stroke:#60A5FA,stroke-width:2px,color:#ffffff;
    classDef decisionNode fill:#78350F,stroke:#FBBF24,stroke-width:2px,color:#ffffff;
    classDef errorNode fill:#7F1D1D,stroke:#F87171,stroke-width:2px,color:#ffffff;
    classDef successNode fill:#064E3B,stroke:#34D399,stroke-width:2px,color:#ffffff;

    class start,finish terminalNode;
    class upload,preview,confirm,validate,normalize,persist processNode;
    class schema,duplicate decisionNode;
    class error,reject errorNode;
    class summary successNode;
```

</details>

---

### How the Model Learns

```mermaid
flowchart TD
    DB[("SQLite database")] --> JOIN["Join employee tables"]
    JOIN --> LABELED["Filter labeled records"]
    LABELED --> SPLIT["Stratified train/test split"]
    SPLIT --> NUM["Numeric branch: impute & scale"]
    SPLIT --> CAT["Categorical branch: impute & one-hot encode"]
    NUM --> COMBINE["Feature union"]
    CAT --> COMBINE
    COMBINE --> FIT["Fit classifier (LR / RF / GB)"]
    FIT --> EVAL["Evaluate on holdout set"]
    EVAL --> SAVE["Save artifact as churn_model.pkl"]

    classDef dbNode fill:#0F766E,stroke:#2DD4BF,stroke-width:2px,color:#ffffff;
    classDef prepNode fill:#0284C7,stroke:#38BDF8,stroke-width:2px,color:#ffffff;
    classDef splitNode fill:#4338CA,stroke:#818CF8,stroke-width:2px,color:#ffffff;
    classDef transformNode fill:#1D4ED8,stroke:#60A5FA,stroke-width:2px,color:#ffffff;
    classDef modelNode fill:#B45309,stroke:#FBBF24,stroke-width:2px,color:#ffffff;
    classDef saveNode fill:#047857,stroke:#34D399,stroke-width:2px,color:#ffffff;

    class DB dbNode;
    class JOIN,LABELED prepNode;
    class SPLIT splitNode;
    class NUM,CAT,COMBINE transformNode;
    class FIT,EVAL modelNode;
    class SAVE saveNode;
```

<details>
<summary><strong>📐 Click to toggle Model Pipeline UML Component Diagram</strong></summary>

```mermaid
classDiagram
    class TrainingData {
        +employee_id
        +age
        +income
        +tenure_months
        +department
        +contract_type
        +job_title
        +churn
    }
    class NumericTransformer {
        +SimpleImputer(mean)
        +StandardScaler
    }
    class CategoricalTransformer {
        +SimpleImputer(most_frequent)
        +OneHotEncoder(ignore_unknown)
    }
    class ColumnTransformer {
        +combine numeric and categorical branches
    }
    class Classifier {
        +LogisticRegression / RandomForest / GradientBoosting
        +fit(features, churn)
        +predict_proba(features)
    }
    class FittedPipeline {
        +preprocessor
        +classifier
        +predict_proba(employee)
    }
    TrainingData --> ColumnTransformer : model features
    ColumnTransformer *-- NumericTransformer
    ColumnTransformer *-- CategoricalTransformer
    ColumnTransformer --> Classifier : transformed features
    FittedPipeline *-- ColumnTransformer
    FittedPipeline *-- Classifier

    style TrainingData fill:#0284C7,stroke:#38BDF8,stroke-width:2px,color:#ffffff
    style NumericTransformer fill:#4338CA,stroke:#818CF8,stroke-width:2px,color:#ffffff
    style CategoricalTransformer fill:#4338CA,stroke:#818CF8,stroke-width:2px,color:#ffffff
    style ColumnTransformer fill:#1D4ED8,stroke:#60A5FA,stroke-width:2px,color:#ffffff
    style Classifier fill:#B45309,stroke:#FBBF24,stroke-width:2px,color:#ffffff
    style FittedPipeline fill:#047857,stroke:#34D399,stroke-width:2px,color:#ffffff
```

</details>

---

## ☁️ Architecture Comparison: Local vs. Cloud Deployment

```mermaid
flowchart LR
    subgraph Local_Stack ["💻 Local Development (Single Machine)"]
        direction TB
        L_USER["User Browser<br/>(localhost:8501)"] --> L_ST["Local Streamlit App<br/>(Python Process)"]
        L_ST -->|"Direct In-Process Call / HTTP"| L_API["Local Predictor / FastAPI<br/>(localhost:8000)"]
        L_API -->|"Direct File Read"| L_MODEL["Local Model File<br/>(./churn_model.pkl)"]
        L_API -->|"Direct SQL Query/Write"| L_DB[("Local SQLite File<br/>(./employees.db)")]
        L_MODEL --> L_API
        L_API -->|"In-Process Return"| L_ST
        L_ST -->|"Render Dashboard"| L_USER
    end

    subgraph Cloud_Stack ["☁️ Cloud Deployment (Streamlit Cloud + Vercel)"]
        direction TB
        C_USER["End User Browser<br/>(Web / Mobile)"] --> C_ST["Streamlit Cloud<br/>(share.streamlit.io)"]
        C_ST -->|"HTTPS REST Request<br/>POST /predict"| C_VERCEL["FastAPI on Vercel<br/>(Serverless /api/index.py)"]
        C_VERCEL -->|"In-Memory Pipeline"| C_MODEL["Bundled Artifact<br/>(churn_model.pkl)"]
        C_VERCEL -->|"Ephemeral SQLite Read/Write"| C_DB[("Serverless SQLite<br/>(/tmp/employees.db)")]
        C_MODEL --> C_VERCEL
        C_VERCEL -->|"HTTPS JSON Response<br/>(probability, risk, latency)"| C_ST
        C_ST -->|"Render Live Cloud UI"| C_USER
    end

    classDef userNode fill:#1E293B,stroke:#64748B,stroke-width:2px,color:#ffffff;
    classDef stNode fill:#1D4ED8,stroke:#60A5FA,stroke-width:2px,color:#ffffff;
    classDef apiNode fill:#4338CA,stroke:#818CF8,stroke-width:2px,color:#ffffff;
    classDef modelNode fill:#B45309,stroke:#FBBF24,stroke-width:2px,color:#ffffff;
    classDef dbNode fill:#0F766E,stroke:#2DD4BF,stroke-width:2px,color:#ffffff;

    class L_USER,C_USER userNode;
    class L_ST,C_ST stNode;
    class L_API,C_VERCEL apiNode;
    class L_MODEL,C_MODEL modelNode;
    class L_DB,C_DB dbNode;
```

| Dimension | Local Development | Streamlit Cloud + Vercel Deployment |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit running locally on `http://localhost:8501` | Hosted globally on **Streamlit Community Cloud** |
| **Backend & API** | Local FastAPI process on `http://127.0.0.1:8000` | Serverless Python function on **Vercel** (`https://your-app.vercel.app`) |
| **Communication** | Direct in-memory Python function call or local loopback | Public **HTTPS REST API** (`POST /predict`) with CORS enabled |
| **Database** | Persistent SQLite read/write to root `./employees.db` | Read-only bundle copied to ephemeral `/tmp/employees.db` on cold start |
| **Model Artifact** | Local file access from root `./churn_model.pkl` | Bundled in serverless package and loaded into memory on cold start |
| **Scalability** | Single machine CPU process | **Auto-scaling serverless instances** on-demand |

---

## ⚡ Decoupled Inference vs. Retraining Pipeline

### Why Evaluation Metrics Don't Change Automatically During Inference

In this architecture, **Inference (Prediction)** and **Model Evaluation (Training)** are deliberately decoupled:

```mermaid
flowchart LR
    subgraph INFER ["⚡ Inference Path (Predicting)"]
        direction TB
        I1["1. Faker Data Generated / Repaired"] --> I2["2. Saved to SQLite"]
        I2 --> I3["3. Click 'Predict Imported Employees'"]
        I3 --> I4["4. Scored with EXISTING churn_model.pkl"]
        I4 --> I5["5. Logged to 'Prediction History' Table"]
    end

    subgraph TRAIN ["🧠 Evaluation Path (Retraining)"]
        direction TB
        T1["1. Labeled Dataset (churn = 0 / 1)"] --> T2["2. Stratified 80/20 Holdout Split"]
        T2 --> T3["3. Trigger Retrain / Hyperparameter Tuning"]
        T3 --> T4["4. Compute Precision, Recall, F1, ROC-AUC"]
        T4 --> T5["5. Log Run ID to 'Model Evaluation History'"]
    end

    I2 -.->|"Supplies New Ground Truth"| T1

    classDef inferNode fill:#0284C7,stroke:#38BDF8,stroke-width:2px,color:#ffffff;
    classDef trainNode fill:#047857,stroke:#34D399,stroke-width:2px,color:#ffffff;

    class I1,I2,I3,I4,I5 inferNode;
    class T1,T2,T3,T4,T5 trainNode;
```

#### Production Best Practice
> **Decoupling prediction from retraining** prevents silent model drift and accidental model poisoning from untested data.
> It guarantees sub-millisecond prediction latency and ensures every set of evaluation metrics is cleanly tied to a versioned **Run ID** in the database.

---

## 📁 Repository Structure

| File / Folder | Purpose |
| :--- | :--- |
| `streamlit_app.py` | 4-tab interactive HR dashboard and persistent sidebar simulation studio |
| `churn_app/api.py` | FastAPI application and `POST /predict` inference endpoint |
| `churn_app/database.py` | SQLite schema, import validation, feature lookups, and run audit persistence |
| `churn_app/faker_producer.py` | Python Faker Producer for synthetic batch generation and 1-click row repair |
| `churn_app/feature_engineering.py` | Central tendency baseline calculations and zero-null feature imputation layer |
| `churn_app/training.py` | Pipeline architectures (LogReg, Random Forest, GradBoost), tuning, and metrics |
| `churn_app/prediction.py` | Model loader, single-employee scoring, and risk band classifier |
| `churn_app/train_and_evaluate.py` | CLI training runner and evaluation artifact generator |
| `api/index.py` | Serverless entrypoint for Vercel cloud deployment |
| `vercel.json` | Vercel deployment routing and python serverless configuration |
| `artifacts/` | Holdout evaluation metrics, specifications, and architecture diagrams |
| `data/` | Historical datasets, bulk import files, and drift experiment records |
| `notebooks/` | Jupyter notebooks for exploratory data analysis and model experimentation |

---

*Documentation and architecture diagrams created via IBM Bob.*
