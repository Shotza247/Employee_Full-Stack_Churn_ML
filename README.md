# FastAPI For Inferencing

This project stores employee records in SQLite, trains a churn model from joined database tables, and exposes both a FastAPI prediction API and a Streamlit onboarding UI.

## Setup

1. Create the Python environment:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
   py -m venv venv
   venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

## Database initialization

Seed the database and create the SQLite tables:

```powershell
py -c "from database import init_db, populate_sample_data; init_db(); populate_sample_data()"
```

## Train the churn model from the database

Run training and evaluation non-interactively:

```powershell
py -m churn_app.train_and_evaluate
```

The notebook `train_churn_model.ipynb` remains an experimental workflow over the same reusable package.

After training, this notebook creates:

- `churn_model.pkl`
- `batch_predictions.csv`
- `training_data_from_db.csv`

After changing the feature set or preprocessing pipeline, retrain the model and deploy the newly generated `churn_model.pkl`. The model uses numeric features (`age`, `income`, and `tenure_months`) and categorical features (`department`, `contract_type`, and `job_title`). Its fitted pipeline mean-imputes numeric values, most-frequent-imputes categorical values, scales numeric values, and one-hot encodes categories. Unknown categories at prediction time are safely ignored by the encoder.

## Run the FastAPI service

```powershell
py -m uvicorn churn_app.api:app --reload --host 0.0.0.0 --port 8000
```

Then call the prediction endpoint:

```json
POST http://127.0.0.1:8000/predict
{
  "employee_id": 1
}
```

## Run the Streamlit app

```powershell
streamlit run streamlit_app.py
```

The Streamlit app allows you to insert new employees into the database and score employees locally with `churn_model.pkl`. After importing a CSV, click **Predict imported employees** to score only the rows accepted from that import. The Streamlit app does not need the FastAPI service running for these predictions. Make sure the trained model artifact is included in the deployment.

## Run a drift experiment

Upload `noisy_churn_experiment.csv` from the Streamlit **Import noisy experiment dataset** section. The file contains shifted age, income, and tenure distributions, extreme but valid values, contract and department shifts, explicit scenario labels, and one duplicate row.

The importer previews feature means against the baseline, rejects invalid or duplicate rows, preserves missing feature values as `NULL`, and adds accepted employees to SQLite. First and last names remain required to identify employee records. The training pipeline fits numeric mean imputation and categorical most-frequent imputation using labeled training rows, then scales numeric features and one-hot encodes categorical features. Missing `churn` outcomes remain `NULL` and are excluded from supervised training; they are not fabricated. The fitted pipeline applies the same imputation and encoding to prediction inputs. Streamlit predictions run locally and do not require the API service:

```powershell
py -m streamlit run streamlit_app.py
```

The **Predict imported employees** button scores the accepted imported records with the current model. The roster batch view also uses that local model. Uploaded labels are observed outcomes only; prediction uses the feature columns, not the uploaded churn labels.
