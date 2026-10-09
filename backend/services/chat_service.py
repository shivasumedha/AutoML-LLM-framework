"""Rule-based dataset chatbot."""

import numpy as np

from backend.core.state import state
from backend.services.data_service import load_current_df, records

EXAMPLES = (
    "Try asking:\n"
    "• How many rows are there?\n• Show missing values\n"
    "• What are the data types?\n• Are there duplicate rows?\n"
    "• Give me a dataset summary"
)


def _find_column(df, words):
    for column in df.columns:
        name = str(column).lower()
        if any(word in name for word in words):
            return column
    return None


def _has(message, *phrases):
    return any(p in message for p in phrases)


def answer(user_message: str) -> str:
    if state.dataset_path is None:
        return (
            "Please upload a CSV dataset first. Once uploaded, "
            "I can answer questions about your dataset."
        )

    df = load_current_df()
    text = user_message.strip()
    if not text:
        return "Please enter a question."

    m = text.lower()
    numeric = df.select_dtypes(include=np.number)

    if m in ["hi", "hello", "hey", "hi bot", "hello bot"]:
        return (
            "Hello! 👋 I am your AutoML dataset assistant. Ask me about rows, "
            "columns, missing values, duplicates, data types, or statistics."
        )

    if _has(m, "how many rows", "number of rows") or m == "rows":
        return f"Your dataset has {df.shape[0]} rows."

    if _has(m, "how many columns", "number of columns") or m == "columns":
        return f"Your dataset has {df.shape[1]} columns."

    if _has(m, "summary", "dataset info", "dataset information", "about dataset", "describe dataset"):
        return (
            f"Your dataset contains {df.shape[0]} rows and {df.shape[1]} columns. "
            f"There are {int(df.duplicated().sum())} duplicate rows and "
            f"{int(df.isnull().sum().sum())} missing values."
        )

    if _has(m, "column names", "what columns", "list columns", "show columns"):
        return "The dataset columns are: " + ", ".join(str(c) for c in df.columns)

    if _has(m, "missing", "null", "empty values"):
        missing = df.isnull().sum()
        total = int(missing.sum())
        if total == 0:
            return "There are no missing values in your dataset. ✓"
        cols = {str(k): int(v) for k, v in missing.items() if v > 0}
        return f"There are {total} missing values. Columns with missing values: {cols}"

    if _has(m, "duplicate"):
        return f"Your dataset contains {int(df.duplicated().sum())} duplicate rows."

    if _has(m, "data types", "datatype", "types of columns"):
        return f"Column data types: { {str(k): str(v) for k, v in df.dtypes.items()} }"

    if _has(m, "numeric columns", "numerical columns"):
        return f"Numeric columns: {numeric.columns.tolist()}"

    if _has(m, "categorical"):
        return f"Categorical columns: {df.select_dtypes(exclude=np.number).columns.tolist()}"

    if _has(m, "first rows", "first 5", "top rows", "sample data"):
        return f"Here are the first 5 rows:\n{records(df.head(5))}"

    if _has(m, "attrition", "target", "class distribution"):
        col = state.target_column if state.target_column in df.columns else _find_column(
            df, ["attrition", "target", "label", "class", "outcome", "species"]
        )
        if col:
            return f"Target column: '{col}'. Class distribution: {df[col].value_counts().to_dict()}"

    if _has(m, "unique values"):
        return f"Unique value counts: { {str(c): int(df[c].nunique()) for c in df.columns} }"

    if _has(m, "average", "mean"):
        return f"Average values of numeric columns: {numeric.mean().round(2).to_dict()}"

    if _has(m, "minimum", "min value"):
        return f"Minimum values: {numeric.min().to_dict()}"

    if _has(m, "maximum", "max value"):
        return f"Maximum values: {numeric.max().to_dict()}"

    if _has(m, "model", "trained", "training"):
        if state.model is not None:
            return (
                f"The current trained model is {state.model_name}. "
                f"The target column is '{state.target_column}'."
            )
        return "No model has been trained yet. Go to the Models page, select a target column and train."

    if _has(m, "help", "what can you do"):
        return (
            "I can help you with:\n"
            "• Dataset size and summary\n• Column names and data types\n"
            "• Missing values and duplicates\n• Numeric / categorical columns\n"
            "• Unique values, averages, min and max\n"
            "• Target / class distribution\n• Model training status"
        )

    return f"I understand your question: '{text}'\n\n{EXAMPLES}"
