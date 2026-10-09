import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder
)


# Columns that are identifiers/constants
# and are not useful for prediction.
DEFAULT_DROP_COLUMNS = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours"
]


def prepare_features_and_target(
    df,
    target_column,
    drop_columns=None
):
    """
    Separate features and target.

    For Attrition:
        No  -> 0
        Yes -> 1
    """

    data = df.copy()

    # Check target column
    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    # Remove rows with missing target
    data = data.dropna(
        subset=[target_column]
    )

    # Separate target
    X = data.drop(
        columns=[target_column]
    )

    y = data[target_column].copy()

    # --------------------------------------------------
    # REMOVE UNNECESSARY COLUMNS
    # --------------------------------------------------

    columns_to_drop = set(
        DEFAULT_DROP_COLUMNS
    )

    if drop_columns:
        columns_to_drop.update(
            drop_columns
        )

    columns_to_drop = [
        column
        for column in columns_to_drop
        if column in X.columns
    ]

    if columns_to_drop:
        X = X.drop(
            columns=columns_to_drop
        )

    # --------------------------------------------------
    # TARGET ENCODING
    # --------------------------------------------------

    # Convert target to string temporarily.
    target_values = (
        y.astype(str)
        .str.strip()
        .str.lower()
    )

    unique_values = set(
        target_values.unique()
    )

    # Attrition / Yes-No classification
    if unique_values.issubset(
        {"yes", "no"}
    ):

        y = target_values.map(
            {
                "no": 0,
                "yes": 1
            }
        )

        y = y.astype(int)

    else:

        # Generic categorical target
        if (
            y.dtype == "object"
            or str(y.dtype) == "category"
        ):

            encoder = LabelEncoder()

            y = encoder.fit_transform(
                y.astype(str)
            )

    return X, y


def create_preprocessor(X):
    """
    Automatic preprocessing.

    Numerical:
        Missing values -> Median
        Scaling -> StandardScaler

    Categorical:
        Missing values -> Most Frequent
        Encoding -> OneHotEncoder
    """

    numerical_columns = (
        X.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        X.select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        )
        .columns
        .tolist()
    )

    # --------------------------------------------------
    # NUMERICAL PIPELINE
    # --------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # --------------------------------------------------
    # CATEGORICAL PIPELINE
    # --------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # --------------------------------------------------
    # COMPLETE PREPROCESSOR
    # --------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),

            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ],

        remainder="drop"
    )

    return preprocessor