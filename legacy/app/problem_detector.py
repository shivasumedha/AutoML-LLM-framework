import pandas as pd
import numpy as np


def detect_problem_type(y):
    """
    Automatically detect whether the ML problem is
    Classification or Regression.
    """

    # Remove missing values for analysis
    y_clean = y.dropna()

    # -------------------------------------------------
    # 1. String / categorical target
    # -------------------------------------------------
    if y_clean.dtype == "object" or pd.api.types.is_string_dtype(y_clean):

        unique_values = y_clean.nunique()

        if unique_values <= 20:
            return "CLASSIFICATION"

        return "CLASSIFICATION"

    # -------------------------------------------------
    # 2. Boolean target
    # -------------------------------------------------
    if pd.api.types.is_bool_dtype(y_clean):
        return "CLASSIFICATION"

    # -------------------------------------------------
    # 3. Numeric target
    # -------------------------------------------------
    if pd.api.types.is_numeric_dtype(y_clean):

        unique_values = y_clean.nunique()
        total_values = len(y_clean)

        # Numeric target with only a few unique values
        # is usually classification.
        if unique_values <= 20:
            return "CLASSIFICATION"

        # Integer target with relatively few classes
        if pd.api.types.is_integer_dtype(y_clean):
            if unique_values / total_values < 0.05:
                return "CLASSIFICATION"

        # Otherwise regression
        return "REGRESSION"

    # -------------------------------------------------
    # 4. Fallback
    # -------------------------------------------------
    return "CLASSIFICATION"


def get_target_column(df):
    """
    Automatically suggest a target column.
    """

    # Common target column names
    target_names = [
        "target",
        "Target",
        "label",
        "Label",
        "class",
        "Class",
        "y",
        "Y",
        "attrition",
        "Attrition",
        "churn",
        "Churn",
        "outcome",
        "Outcome"
    ]

    for name in target_names:
        if name in df.columns:
            return name

    return None