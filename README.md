## ML Predictive Pipeline – Housing & Diabetes Regression

A clean, extensible machine-learning pipeline built in Python for training and evaluating regression models using scikit-learn, pipelines, and joblib.
Supports multiple datasets (currently Diabetes from sklearn and a local housing dataset), automatic preprocessing, model persistence, and reproducible evaluation.

This project is structured as a practical, production-style ML workflow suitable for portfolio demonstration.

---

## ✨ Features

✔ Reproducible ML training pipeline

    Built using sklearn.pipeline.Pipeline

    Includes preprocessing:

        StandardScaler for numeric features

        OneHotEncoder for categorical features

    Model: RandomForestRegressor (easy to swap)

✔ Multiple dataset support

    diabetes dataset from scikit-learn

    housing dataset from a local CSV file

    Synthetic housing generator (src/create_dummy_housing.py) included

✔ Clean model saving

    Models are stored under:    
    
        models/<dataset>_rf_pipeline.joblib

    Example:

        models/diabetes_rf_pipeline.joblib

        models/housing_rf_pipeline.joblib


---

## 🧰 Tech Stack

- **Python 3.10+**
- **Pandas**
- **NumPy**
- **scikit-learn**
- **Matplotlib & Seaborn**
- **Joblib**
- **Jupyter Notebook**

---

## 📂 Project Structure

ml-predictive-pipeline-housing/
│
├── src/
│   ├── train.py                   # Main training script (supports multiple datasets)
│   ├── create_dummy_housing.py    # Generates offline synthetic housing dataset
│   └── ... (future modules)
│
├── models/                        # Saved trained models
│   ├── diabetes_rf_pipeline.joblib
│   └── housing_rf_pipeline.joblib
│
└── data/
    └── housing.csv                # Local housing dataset used for training

---

## 📊 Datasets

1. Diabetes Dataset (scikit-learn)

    Loaded via load_diabetes()

    All-numeric features

    Roadmap-friendly baseline model

Command:

    // python -m src.train --dataset diabetes

2. Housing Dataset (local CSV)

    Loaded from:

        data/housing.csv

    Requirements:

        Must contain a column named target

        Other columns may be numeric or categorical

    If you have no housing CSV yet, generate a synthetic one:

        // python -m src.create_dummy_housing

    Then train:

        // python -m src.train --dataset housing

---

## 🏋️ Training Models

    Train using either dataset:

        Diabetes:

            // python -m src.train --dataset diabetes

        Housing:

            // python -m src.train --dataset housing

    Each run will:

        Load the dataset

        Split into train/test

        Build an ML pipeline

        Train a RandomForestRegressor

        Evaluate using MAE, RMSE, and R²

        Save the model + metrics using joblib

---

## 📈 Example Output

→ Loading data for dataset: diabetes
→ Splitting train/test...
→ Building pipeline...
→ Training model...
→ Evaluating on test set...
Evaluation metrics:
  MAE : 44.4154
  RMSE: 54.5943
  R²  : 0.4374
→ Saving model to: models/diabetes_rf_pipeline.joblib
✅ Training complete.

---

## 🔧 Configuration

Modify defaults at top of src/train.py:

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 200

---

## 🧪 Future Extensions

    Feel free to expand this project with:

        Hyperparameter tuning (GridSearchCV)

        Cross-validation reports

        REST inference server (FastAPI)

        Model explainability (SHAP values)

        Additional datasets

        Automated training via Makefile or bash scripts

---

## 💡 Why this project is great for a portfolio

    Shows understanding of real-world ML engineering patterns

    Not just notebooks — proper project structure

    Clean CLI + modular code

    Demonstrates ability to build maintainable ML pipelines

    ffline dataset option makes the repo self-contained

---









