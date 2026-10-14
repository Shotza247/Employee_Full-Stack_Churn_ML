# Employee Churn Intelligence

An interactive employee-churn project that stores employee records in SQLite, trains a scikit-learn classification pipeline, and makes predictions through a Streamlit app or a FastAPI endpoint.

> **New here?** Follow the quick start first. Then explore the diagrams to see how an uploaded row travels through the database, training pipeline, and prediction flow.

## Project at a glance

```mermaid
flowchart LR
    CSV["Employee CSV"] --> UI["Streamlit app"]
    UI -->|validate and import| DB[("SQLite database")]
    DB -->|labeled examples| TRAIN["Training pipeline"]
    TRAIN -->|save fitted pipeline| MODEL["churn_model.pkl"]
    UI -->|predict button| PREDICT["Local prediction"]
    API["FastAPI client"] -->|employee_id| SERVICE["FastAPI"]
    SERVICE --> DB
    DB -->|employee features| PREDICT
    MODEL --> PREDICT
    PREDICT --> RESULT["Churn probability and risk band"]
    SERVICE --> RESULT
```

### What the app does

- **Imports:** previews a CSV, checks required columns and row values, and reports accepted and rejected records.
- **Stores:** saves employee identity, features, and optional observed churn labels in SQLite.
- **Trains:** fits numeric and categorical preprocessing together with a logistic-regression classifier.
- **Predicts:** scores imported records from Streamlit or an individual employee through FastAPI.
- **Evaluates:** records model metrics and predictions for labeled employees in SQLite.

**Important:** uploading records does not automatically retrain the model. Use the explicit **Predict imported employees** button to score accepted rows with the current model. Retraining is a separate step.

## Quick start

### 1. Set up Python

Open PowerShell in the project folder and run:

```powershell
py -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation in this terminal, allow it for the current process and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
venv\Scripts\Activate.ps1
```

### 2. Initialize and seed the database

```powershell
python -c "from churn_app.database import init_db, populate_sample_data; init_db(); populate_sample_data()"
```

This creates the SQLite tables and adds example employees if the employee table is empty. Existing employee records are not replaced.

### 3. Train the model

```powershell
python -m churn_app.train_and_evaluate
```

The command reads employee data from SQLite, trains and evaluates the model, saves the fitted model artifact, and writes reports to the `artifacts` folder:

| Output | What it contains |
|---|---|
| `churn_model.pkl` | Fitted preprocessing and classification pipeline used by the apps |
| `artifacts/metrics.json` | Holdout evaluation metrics and model run ID |
| `artifacts/training_data.csv` | Labeled rows used in supervised training |
| `artifacts/batch_predictions.csv` | Predictions generated for labeled training rows |

The database and model paths can be changed with `CHURN_DB_PATH` and `CHURN_MODEL_PATH`. For example, in PowerShell:

```powershell
$env:CHURN_DB_PATH = "C:\data\employees.db"
$env:CHURN_MODEL_PATH = "C:\data\churn_model.pkl"
python -m churn_app.train_and_evaluate
```

### 4. Run the Streamlit app

```powershell
python -m streamlit run streamlit_app.py
```

In the browser:

1. Choose a CSV in the **Batch import** section.
2. Review the preview and data-quality summary.
3. Select **Import valid rows**.
4. Review how many rows were imported or rejected.
5. Select **Predict imported employees** to score the accepted rows.

Streamlit loads the local `churn_model.pkl` and predicts directly; FastAPI does not need to be running for Streamlit predictions.

### 5. (Optional) Run the FastAPI service

In a separate terminal, activate the same environment and run:

```powershell
python -m uvicorn churn_app.api:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to try the API interactively. The prediction endpoint accepts an employee ID:

```http
POST http://127.0.0.1:8000/predict
Content-Type: application/json

{
  "employee_id": 1
}
```

The API returns the churn probability, risk band, and available numeric feature values for that employee.

## Follow a row through the data pipeline

The CSV importer expects these columns: `first_name`, `last_name`, `age`, `department`, `contract_type`, `annual_income`, `job_title`, and `tenure_months`. A `churn` column is optional; if present, accepted values are `0`, `1`, or missing.

First and last names are required to identify a record. Missing model features are allowed and stored as SQL `NULL`. Non-missing age and tenure values must be whole numbers in the accepted ranges; income must be non-negative. Department and contract values are checked against the app’s known values. Invalid rows and duplicate records are reported rather than imported.

```mermaid
flowchart TD
    A["Select CSV"] --> B["Check required columns"]
    B --> C{"All required columns present?"}
    C -- No --> D["Show missing column names"]
    C -- Yes --> E["Preview data and quality summary"]
    E --> F["User selects Import valid rows"]
    F --> G["Validate each row"]
    G --> H{"Valid and not a duplicate?"}
    H -- No --> I["Add row and reason to rejected list"]
    H -- Yes --> J["Insert employee, salary, and job data"]
    J --> K["Keep missing feature values as NULL"]
    I --> L["Show import results"]
    K --> L
```

### Data pipeline UML activity diagram

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
```

## How the model learns

The model is trained from the database tables joined into one employee-level dataset. Only rows with a known `churn` label are eligible for supervised training. Missing churn labels stay missing; the project does not invent outcomes.

| Feature group | Columns | Training preprocessing |
|---|---|---|
| Numeric | `age`, `income`, `tenure_months` | Fill missing values with each feature’s training-set mean, then standardize |
| Categorical | `department`, `contract_type`, `job_title` | Fill missing values with the most frequent training-set value, then one-hot encode |
| Target | `churn` | Binary label; unlabeled rows are excluded from fitting |

The labeled examples are split into training and holdout sets. Imputation and scaling/encoding are fitted as part of the pipeline on the training split, so the holdout set and future predictions receive the learned transformations without fitting preprocessing on those rows. Logistic regression classifies the transformed features.

```mermaid
flowchart TD
    DB[("SQLite employee, salary, and job tables")] --> JOIN["Join rows into employee dataset"]
    JOIN --> LABELED["Keep rows with known churn label"]
    LABELED --> SPLIT["Stratified train / holdout split"]
    SPLIT --> TRAIN["Training rows"]
    SPLIT --> TEST["Holdout rows"]
    TRAIN --> NUM["Numeric branch: mean impute, standardize"]
    TRAIN --> CAT["Categorical branch: mode impute, one-hot encode"]
    NUM --> COMBINE["Combine transformed features"]
    CAT --> COMBINE
    COMBINE --> FIT["Fit logistic regression"]
    FIT --> EVAL["Evaluate on holdout rows"]
    EVAL --> SAVE["Save full fitted pipeline as churn_model.pkl"]
    SAVE --> EXPORT["Write metrics and batch prediction reports"]
```

### Model pipeline UML component diagram

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
    class LogisticRegression {
        +fit(features, churn)
        +predict_proba(features)
    }
    class FittedPipeline {
        +preprocessor
        +classifier
        +predict_proba(new_employee)
    }
    TrainingData --> ColumnTransformer : six model features
    ColumnTransformer *-- NumericTransformer
    ColumnTransformer *-- CategoricalTransformer
    ColumnTransformer --> LogisticRegression : transformed features
    FittedPipeline *-- ColumnTransformer
    FittedPipeline *-- LogisticRegression
```

## How predictions work

Streamlit and FastAPI share the same database feature lookup and prediction function. The saved artifact includes its fitted imputers, scaler, encoder, and classifier. Missing values are transformed using training-time statistics; a category not seen during training is ignored by the one-hot encoder rather than causing a prediction error.

The returned risk band is based on the probability: `low` below `0.4`, `medium` from `0.4` to below `0.7`, and `high` at `0.7` or above.

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit or API client
    participant App as Prediction handler
    participant DB as SQLite
    participant Model as Saved fitted pipeline

    User->>UI: Choose imported employees or submit employee_id
    UI->>App: Request prediction
    App->>DB: Read employee feature row
    DB-->>App: Numeric and categorical values (may be missing)
    App->>Model: predict_proba(features)
    Note over Model: Apply fitted imputers, scaling, and one-hot encoding
    Model-->>App: Churn probability
    App->>App: Map probability to risk band
    App-->>UI: Probability, band, and available numeric values
    UI-->>User: Display prediction
```

### Prediction flow UML sequence diagram

```mermaid
sequenceDiagram
    actor Analyst
    participant Streamlit
    participant FeatureLookup as Database feature lookup
    participant SQLite
    participant Predictor as predict_employee
    participant Artifact as churn_model.pkl

    Analyst->>Streamlit: Click "Predict imported employees"
    loop Each accepted employee ID
        Streamlit->>FeatureLookup: Request feature row
        FeatureLookup->>SQLite: SELECT employee and joined features
        SQLite-->>FeatureLookup: Feature values
        FeatureLookup-->>Streamlit: Row for employee
        Streamlit->>Predictor: Predict with row
        Predictor->>Artifact: Transform and calculate probability
        Artifact-->>Predictor: Probability of churn
        Predictor-->>Streamlit: Probability and risk band
    end
    Streamlit-->>Analyst: Show predictions for imported employees
```

### UML class diagram: application and data

```mermaid
classDiagram
    class Employees {
        +employee_id PK
        +first_name
        +last_name
        +age nullable
        +department nullable
        +contract_type nullable
        +churn nullable
    }
    class Salaries {
        +salary_id PK
        +employee_id FK
        +annual_income nullable
    }
    class Jobs {
        +job_id PK
        +employee_id FK
        +job_title nullable
        +tenure_months nullable
    }
    class ModelRuns {
        +run_id PK
        +run_timestamp
        +accuracy
        +precision
        +recall
        +f1
        +roc_auc
    }
    class Predictions {
        +prediction_id PK
        +run_id FK
        +employee_id FK
        +predicted_prob
        +predicted_label
        +true_label nullable
        +created_at
    }
    class StreamlitApp {
        +import_csv()
        +predict_imported_employees()
    }
    class FastAPI {
        +POST /predict
    }
    class TrainingModule {
        +train_model()
        +build_model()
    }
    class PredictionModule {
        +load_model()
        +predict_employee()
    }

    Employees "1" --> "0..1" Salaries : employee_id
    Employees "1" --> "0..1" Jobs : employee_id
    Employees "1" --> "0..*" Predictions : scored employee
    ModelRuns "1" --> "0..*" Predictions : run results
    StreamlitApp --> Employees : imports and looks up
    FastAPI --> Employees : looks up
    TrainingModule --> Employees : reads labeled data
    TrainingModule --> ModelRuns : records evaluation
    TrainingModule --> Predictions : saves batch scores
    PredictionModule --> Employees : reads feature row
    PredictionModule --> Predictions : shared model logic
```

## Retrain after changing the data or model

Newly uploaded rows are available in SQLite, but the already-fitted model does not learn from them automatically. To include new observed labels in training:

1. Import the updated dataset, including `churn` values where known.
2. Run `python -m churn_app.train_and_evaluate`.
3. Review `artifacts/metrics.json` and the generated batch predictions.
4. Use the newly generated `churn_model.pkl` with the app.

Rows without churn labels remain stored and can be predicted, but are left out of model fitting. If a numeric feature or categorical feature has no observed training values, training stops with an explanatory error because the imputer cannot learn a replacement value.

When deploying, package the code, dependencies, trained model artifact, and a database containing the employee data you expect to query. SQLite is a local file: separate deployments or instances will not automatically share its contents. On hosting platforms with ephemeral filesystems, database changes may not survive restarts, so use persistent storage or a shared database if that is required. After replacing the model artifact, restart/redeploy the app so Streamlit’s cached model is loaded again; FastAPI loads its model when the service starts.

## Explore the code

| File | Purpose |
|---|---|
| `streamlit_app.py` | Upload UI, row import, dashboards, and local prediction button |
| `churn_app/database.py` | SQLite schema, import validation, feature lookup, and training-data queries |
| `churn_app/training.py` | Feature definitions, preprocessing pipeline, training, evaluation, and artifact saving |
| `churn_app/prediction.py` | Model loading, employee scoring, and risk-band mapping |
| `churn_app/api.py` | FastAPI application and `POST /predict` route |
| `churn_app/train_and_evaluate.py` | Command-line training and report export |
| `train_churn_model.ipynb` | Notebook-based exploration using the shared model pipeline |

## Try the drift experiment

The Streamlit app includes an **Import noisy experiment dataset** section for `noisy_churn_experiment.csv`. The example dataset introduces shifted feature distributions, extreme but valid values, scenario labels, and a duplicate to explore import validation and changes in model behavior. Review its quality summary and rejected rows, then explicitly retrain if you want labeled experiment rows to affect the model.
