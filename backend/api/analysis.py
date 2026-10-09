import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.core.state import state
from backend.services.data_service import load_current_df, records

router = APIRouter(tags=["Analysis"])


@router.get("/analyze")
def analyze_dataset():
    try:
        df = load_current_df()

        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()

        missing_values = {
            str(k): int(v)
            for k, v in df.isnull().sum().sort_values(ascending=False).items()
        }

        numeric_summary = {}
        if numeric_columns:
            desc = df[numeric_columns].describe()
            numeric_summary = {
                str(col): {
                    str(stat): (None if pd.isna(val) else float(val))
                    for stat, val in desc[col].items()
                }
                for col in desc.columns
            }

        return {
            "status": "success",
            "filename": state.dataset_name,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": df.columns.tolist(),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "missing_values": missing_values,
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric_summary": numeric_summary,
            "preview": records(df.head(10)),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@router.get("/charts")
def chart_data(bins: int = 10, max_categories: int = 10):
    try:
        df = load_current_df()
        bins = max(3, min(bins, 30))
        numeric_df = df.select_dtypes(include=np.number)

        histograms = {}
        for col in numeric_df.columns:
            s = numeric_df[col].dropna()
            if s.empty:
                continue
            counts, edges = np.histogram(s, bins=bins)
            histograms[str(col)] = [
                {"range": f"{edges[i]:.2f} - {edges[i + 1]:.2f}", "count": int(counts[i])}
                for i in range(len(counts))
            ]

        categories = {}
        for col in df.select_dtypes(exclude=np.number).columns:
            vc = df[col].value_counts().head(max_categories)
            categories[str(col)] = [{"name": str(k), "count": int(v)} for k, v in vc.items()]

        missing = [
            {"column": str(c), "missing": int(v)}
            for c, v in df.isnull().sum().items()
            if v > 0
        ]

        correlation = []
        corr_cols = numeric_df.columns[:12]
        if len(corr_cols) >= 2:
            corr = numeric_df[corr_cols].corr().round(3)
            for x in corr.columns:
                for y in corr.columns:
                    value = corr.loc[x, y]
                    correlation.append(
                        {
                            "x": str(x),
                            "y": str(y),
                            "value": None if pd.isna(value) else float(value),
                        }
                    )

        return {
            "status": "success",
            "histograms": histograms,
            "categories": categories,
            "missing": missing,
            "correlation": correlation,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart data failed: {e}")
