"""Model training, feature schema and prediction logic."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from backend.core.state import state


class TrainingError(Exception):
    """Invalid training request (maps to HTTP 400)."""


def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers)


def train_models(df: pd.DataFrame, target: str) -> dict:
    target = target.strip()

    if target not in df.columns:
        raise TrainingError(
            f"Target column '{target}' not found. Available: {df.columns.tolist()}"
        )

    df = df.dropna(subset=[target])
    y = df[target]
    X = df.drop(columns=[target])

    if y.nunique() < 2:
        raise TrainingError("Target column must contain at least 2 classes.")

    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        raise TrainingError(
            f"'{target}' looks continuous ({y.nunique()} unique values). "
            "This trainer handles classification. Pick a column with a few classes."
        )

    if X.shape[1] == 0:
        raise TrainingError("Dataset needs at least one feature column besides the target.")

    # Stratify only when every class has at least 2 rows
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    preprocessor = create_preprocessor(X)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=42),
    }

    results = []
    best_f1 = -1.0
    best_pipeline = None
    best_name = None

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[("preprocessor", clone(preprocessor)), ("model", model)]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        results.append(
            {
                "model": name,
                "accuracy": round(float(accuracy_score(y_test, preds)), 4),
                "precision": round(
                    float(precision_score(y_test, preds, average="weighted", zero_division=0)), 4
                ),
                "recall": round(
                    float(recall_score(y_test, preds, average="weighted", zero_division=0)), 4
                ),
                "f1_score": round(float(f1), 4),
            }
        )

        if f1 > best_f1:
            best_f1, best_pipeline, best_name = f1, pipeline, name

    state.save_model(best_pipeline, best_name, target)

    return {
        "status": "success",
        "message": "Models trained successfully.",
        "target_column": target,
        "best_model": best_name,
        "best_f1_score": round(float(best_f1), 4),
        "results": results,
    }


def build_feature_schema(df: pd.DataFrame) -> list:
    """Describes every input column so the frontend can build the form."""
    features = []
    for col in df.columns:
        if col == state.target_column:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].dropna()
            features.append(
                {
                    "name": str(col),
                    "type": "number",
                    "min": float(s.min()) if not s.empty else None,
                    "max": float(s.max()) if not s.empty else None,
                    "median": float(s.median()) if not s.empty else None,
                }
            )
        else:
            features.append(
                {
                    "name": str(col),
                    "type": "category",
                    "options": [str(v) for v in df[col].dropna().unique()[:50]],
                }
            )
    return features


def predict_one(data: dict) -> dict:
    model = state.model
    preprocessor = model.named_steps["preprocessor"]
    expected = list(preprocessor.feature_names_in_)

    numeric_cols = []
    for name, _, cols in preprocessor.transformers_:
        if name == "numeric":
            numeric_cols = list(cols)

    row = {}
    for col in expected:
        value = data.get(col)
        row[col] = np.nan if value is None or value == "" else value

    input_df = pd.DataFrame([row])
    for col in numeric_cols:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    prediction = model.predict(input_df)[0]
    if isinstance(prediction, np.generic):
        prediction = prediction.item()

    response = {
        "status": "success",
        "prediction": prediction,
        "model": state.model_name,
        "target_column": state.target_column,
    }

    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(input_df)[0]
            response["probabilities"] = [
                {"class": str(c), "probability": round(float(p), 4)}
                for c, p in zip(model.classes_, probs)
            ]
        except Exception:
            pass

    return response
