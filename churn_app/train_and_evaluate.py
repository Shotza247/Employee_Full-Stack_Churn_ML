# Databricks notebook source
"""Train, evaluate, and persist the employee churn model without a notebook."""
import argparse
import json
import logging
from pathlib import Path
from .config import DB_PATH, MODEL_PATH
from .database import get_training_data, init_db, populate_sample_data
from .training import train_model

LOGGER = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--no-seed", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    init_db(args.db_path)
    if not args.no_seed:
        populate_sample_data(args.db_path)
    data = get_training_data(args.db_path)
    if data.empty:
        raise ValueError("No training data found. Add employee records or remove --no-seed.")
    _, metrics, predictions, run_id = train_model(db_path=args.db_path, model_path=args.model_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "batch_predictions.csv", index=False)
    data.to_csv(args.output_dir / "training_data.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps({"run_id": run_id, **metrics}, indent=2), encoding="utf-8")
    LOGGER.info("Training data rows: %d", len(data))
    LOGGER.info("Model artifact: %s", args.model_path)
    print(json.dumps({"run_id": run_id, **metrics}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())