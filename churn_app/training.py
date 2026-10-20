# Databricks notebook source
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import MODEL_PATH
from .database import get_training_data, save_model_run

NUMERIC_FEATURES = ["age", "income", "tenure_months"]
CATEGORICAL_FEATURES = ["department", "contract_type", "job_title"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_model(
    model_type: str = "logistic_regression",
    C: float = 1.0,
    penalty: str = "l2",
    solver: str = "lbfgs",
    class_weight: str | None = None,
    max_iter: int = 300,
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    learning_rate: float = 0.1,
    random_state: int = 42,
):
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
    ])

    if model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth) if max_depth is not None else None,
            min_samples_split=int(min_samples_split),
            class_weight=class_weight,
            random_state=random_state,
        )
    elif model_type == "gradient_boosting":
        classifier = GradientBoostingClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth) if max_depth is not None else 3,
            learning_rate=float(learning_rate),
            random_state=random_state,
        )
    else:  # default to logistic_regression
        lr_kwargs = {
            "C": float(C),
            "solver": solver,
            "class_weight": class_weight,
            "max_iter": int(max_iter),
            "random_state": random_state,
        }
        if penalty != "l2":
            lr_kwargs["penalty"] = penalty
        classifier = LogisticRegression(**lr_kwargs)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def train_model(
    db_path=None,
    model_path=MODEL_PATH,
    model_type: str = "logistic_regression",
    **hyperparameters,
):
    data = get_training_data(db_path) if db_path else get_training_data()
    data = data.loc[data["churn"].notna()].copy()
    if data.empty or data["churn"].nunique() < 2:
        raise ValueError("Training data must contain both churn classes (0 and 1).")

    X, y = data[FEATURE_COLUMNS], data["churn"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    missing_numeric = X_train[NUMERIC_FEATURES].columns[X_train[NUMERIC_FEATURES].isna().all(axis=0)].tolist()
    if missing_numeric:
        raise ValueError(f"Cannot mean-impute features with no observed training values: {', '.join(missing_numeric)}.")

    missing_categorical = X_train[CATEGORICAL_FEATURES].columns[X_train[CATEGORICAL_FEATURES].isna().all(axis=0)].tolist()
    if missing_categorical:
        raise ValueError(f"Cannot infer most-frequent values for categorical features with no observed training values: {', '.join(missing_categorical)}.")

    model = build_model(model_type=model_type, **hyperparameters).fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    labels = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, labels)),
        "precision": float(precision_score(y_test, labels, zero_division=0)),
        "recall": float(recall_score(y_test, labels, zero_division=0)),
        "f1": float(f1_score(y_test, labels, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    artifact_path = Path(model_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)

    all_probabilities = model.predict_proba(X)[:, 1]
    predictions = pd.DataFrame({
        "employee_id": data["employee_id"],
        "predicted_prob": all_probabilities,
        "predicted_label": (all_probabilities >= 0.5).astype(int),
        "true_label": y,
    })

    run_id = save_model_run(metrics, predictions, db_path) if db_path else save_model_run(metrics, predictions)
    logging.info("Saved model to %s as run %s (%s)", artifact_path, run_id, model_type)
    return model, metrics, predictions, run_id


def get_feature_importances(model) -> pd.DataFrame:
    """Extract feature importance or regression coefficients from the fitted pipeline."""
    try:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        feature_names = preprocessor.get_feature_names_out()

        # Clean names for presentation
        clean_names = []
        for name in feature_names:
            clean = name.replace("numeric__", "").replace("categorical__", "")
            clean = clean.replace("_", " ").title()
            clean_names.append(clean)

        if hasattr(classifier, "coef_"):
            coefs = classifier.coef_[0]
            df = pd.DataFrame({
                "Feature": clean_names,
                "Raw Feature": feature_names,
                "Coefficient": coefs,
                "Absolute Impact": np.abs(coefs),
                "Direction": ["Increases Churn Risk" if c > 0 else "Decreases Churn Risk" for c in coefs],
            }).sort_values("Absolute Impact", ascending=False)
            return df
        elif hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
            df = pd.DataFrame({
                "Feature": clean_names,
                "Raw Feature": feature_names,
                "Importance": importances,
                "Absolute Impact": importances,
                "Direction": ["Relative Tree Importance" for _ in importances],
            }).sort_values("Importance", ascending=False)
            return df
    except Exception as error:
        logging.warning("Could not extract feature importances: %s", error)
    return pd.DataFrame()


def compute_feature_correlations(data: pd.DataFrame | None = None, db_path=None) -> pd.DataFrame:
    """
    Compute correlation matrix for numerical features and one-hot encoded categorical features
    relative to churn and each other for feature selection and engineering heatmaps.
    """
    if data is None:
        data = get_training_data(db_path) if db_path else get_training_data()

    labeled = data.loc[data["churn"].notna()].copy()
    if labeled.empty:
        return pd.DataFrame()

    # Create dummy-encoded features for correlation analysis
    df_for_corr = pd.DataFrame()
    for col in NUMERIC_FEATURES:
        if col in labeled:
            df_for_corr[col.title()] = pd.to_numeric(labeled[col], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        if col in labeled:
            dummies = pd.get_dummies(labeled[col], prefix=col.replace("_", " ").title(), drop_first=False, dtype=float)
            df_for_corr = pd.concat([df_for_corr, dummies], axis=1)

    df_for_corr["Churn (Target)"] = labeled["churn"].astype(float)
    corr_matrix = df_for_corr.corr()
    return corr_matrix
