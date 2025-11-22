from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---- Config ----
RANDOM_STATE = 42
TEST_SIZE = 0.2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -------------------
# Data loading
# -------------------
def load_diabetes_data() -> pd.DataFrame:
    """Load the sklearn diabetes regression dataset as a DataFrame."""
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df

def load_housing_data() -> pd.DataFrame:
    """
    Load housing data from a local CSV at: <project_root>/data/housing.csv

    The CSV must contain a numeric 'target' column for the label.
    """
    csv_path = PROJECT_ROOT / "data" / "housing.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Local housing CSV not found at {csv_path}.\n\n"
            "Create it (see src/create_dummy_housing.py helper) or "
            "put your own housing dataset there with a numeric 'target' column."
        )

    df = pd.read_csv(csv_path)

    if "target" not in df.columns:
        raise ValueError(
            f"Local housing CSV {csv_path} must contain a 'target' column "
            "for the regression label."
        )

    return df

def load_data(dataset: str = "diabetes") -> pd.DataFrame:
    dataset = dataset.lower()
    if dataset == "diabetes":
        return load_diabetes_data()
    elif dataset == "housing":
        return load_housing_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use 'diabetes' or 'housing'.")

        

    



def load_data(dataset: str = "diabetes") -> pd.DataFrame:
    """
    Load and return the requested dataset as a DataFrame with a 'target' column.
    """
    dataset = dataset.lower()
    if dataset == "diabetes":
        return load_diabetes_data()
    elif dataset == "housing":
        return load_housing_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use 'diabetes' or 'housing'.")

# -------------------
# Pipeline / model
# -------------------
def build_pipeline(X: pd.DataFrame, random_state: int = RANDOM_STATE) -> Pipeline:
    """
    Build a preprocessing + model pipeline that works for both:
      - all-numeric (diabetes)
      - mixed numeric + categorical (housing)
    """
    # Infer numeric vs categorical columns from dtypes
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipe


# -------------------
# Evaluation utilities
# -------------------
def evaluate(y_true, y_pred) -> dict:
    """Compute regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)      # plain MSE
    rmse = float(np.sqrt(mse))                    # take square root manually
    r2 = r2_score(y_true, y_pred)

    return {"mae": float(mae), "rmse": rmse, "r2": float(r2)}


def pretty_print_metrics(metrics: dict) -> None:
    print("Evaluation metrics:")
    print(f"  MAE : {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²  : {metrics['r2']:.4f}")


# -------------------
# Train entry point
# -------------------
def train(dataset: str = "diabetes") -> None:
    dataset = dataset.lower()
    print(f"→ Loading data for dataset: {dataset}")
    df = load_data(dataset=dataset)

    X = df.drop(columns=["target"])
    y = df["target"]

    print("→ Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print("→ Building pipeline...")
    pipe = build_pipeline(X_train)

    print("→ Training model...")
    pipe.fit(X_train, y_train)

    print("→ Evaluating on test set...")
    y_pred = pipe.predict(X_test)
    metrics = evaluate(y_test, y_pred)
    pretty_print_metrics(metrics)

    model_path = MODEL_DIR / f"{dataset}_rf_pipeline.joblib"
    print(f"→ Saving model to: {model_path}")
    joblib.dump({"pipeline": pipe, "metrics": metrics}, model_path)

    print("✅ Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a RandomForest regression pipeline."
    )
    parser.add_argument(
        "--dataset",
        "-d",
        choices=["diabetes", "housing"],
        default="diabetes",
        help="Which dataset to train on (default: diabetes).",
    )

    args = parser.parse_args()
    train(dataset=args.dataset)