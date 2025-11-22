from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---- Config ----
RANDOM_STATE = 42
TEST_SIZE = 0.2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """
    Load and return the diabetes dataset as a DataFrame.

    This is a regression dataset built into scikit-learn,
    so it does NOT require any network download.
    """
    data = load_diabetes()

    # data.data -> features (NumPy array)
    # data.feature_names -> list of column names
    # data.target -> target values (NumPy array)
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    return df


def build_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """Build a preprocessing + model pipeline."""
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )
    return pipe


def evaluate(y_true, y_pred) -> dict:
    """Compute regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)  # plain MSE
    rmse = np.sqrt(mse)                       # take square root manually
    r2 = r2_score(y_true, y_pred)

    return {"mae": mae, "rmse": rmse, "r2": r2}


def pretty_print_metrics(metrics: dict) -> None:
    print("Evaluation metrics:")
    print(f"  MAE : {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²  : {metrics['r2']:.4f}")


def train() -> None:
    print("→ Loading data...")
    df = load_data()

    X = df.drop(columns=["target"])
    y = df["target"]

    print("→ Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print("→ Building pipeline...")
    pipe = build_pipeline()

    print("→ Training model...")
    pipe.fit(X_train, y_train)

    print("→ Evaluating on test set...")
    y_pred = pipe.predict(X_test)
    metrics = evaluate(y_test, y_pred)
    pretty_print_metrics(metrics)

    model_path = MODEL_DIR / "housing_rf_pipeline.joblib"
    print(f"→ Saving model to: {model_path}")
    joblib.dump({"pipeline": pipe, "metrics": metrics}, model_path)

    print("✅ Training complete.")


if __name__ == "__main__":
    train()
