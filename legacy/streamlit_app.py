
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AutoML-LLM Framework",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "trained": False,
    "best_pipeline": None,
    "best_model_name": None,
    "problem_type": None,
    "target_column": None,
    "label_encoder": None,
    "feature_columns": [],
    "numeric_features": [],
    "categorical_features": [],
    "results_df": None,
    "best_predictions": None,
    "X_test": None,
    "y_test": None,
    "y_all": None,
    "file_name": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 27px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .small-note {
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🤖 AutoML-LLM Framework</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Automated Machine Learning • Cross-Validation • Analytics • Prediction'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ AutoML Settings")

test_size = st.sidebar.slider(
    "Test Data Size",
    min_value=0.10,
    max_value=0.40,
    value=0.20,
    step=0.05
)

random_state = st.sidebar.number_input(
    "Random State",
    min_value=0,
    max_value=1000,
    value=42
)

cv_folds = st.sidebar.slider(
    "Cross-Validation Folds",
    min_value=3,
    max_value=10,
    value=5,
    step=1
)

top_features = st.sidebar.slider(
    "Top Features to Display",
    min_value=5,
    max_value=30,
    value=15,
    step=5
)


# =========================================================
# FILE UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">📂 Upload Dataset</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.info("👆 Upload a CSV dataset to start your AutoML experiment.")
    st.stop()


# =========================================================
# READ DATA
# =========================================================

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"❌ Error reading dataset: {e}")
    st.stop()

df.columns = df.columns.astype(str).str.strip()

if df.empty:
    st.error("❌ The uploaded CSV is empty.")
    st.stop()

st.success(
    f"✅ Dataset loaded successfully — "
    f"{df.shape[0]} rows × {df.shape[1]} columns"
)


# =========================================================
# RESET WHEN FILE CHANGES
# =========================================================

current_file_name = uploaded_file.name

if st.session_state.file_name != current_file_name:
    for key, value in defaults.items():
        if key != "file_name":
            st.session_state[key] = value
    st.session_state.file_name = current_file_name


# =========================================================
# DATASET OVERVIEW
# =========================================================

st.markdown(
    '<div class="section-title">📊 Dataset Overview</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Rows", df.shape[0])

with c2:
    st.metric("Columns", df.shape[1])

with c3:
    st.metric("Missing Values", int(df.isnull().sum().sum()))

with c4:
    st.metric("Duplicate Rows", int(df.duplicated().sum()))


# =========================================================
# DATA PREVIEW
# =========================================================

with st.expander("👀 View Dataset Preview"):
    st.dataframe(df.head(15), use_container_width=True)


# =========================================================
# DATA QUALITY REPORT
# =========================================================

with st.expander("🧹 Data Quality Report"):

    quality_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(dtype) for dtype in df.dtypes],
        "Missing Values": [
            int(df[col].isnull().sum()) for col in df.columns
        ],
        "Missing %": [
            round(df[col].isnull().mean() * 100, 2)
            for col in df.columns
        ],
        "Unique Values": [
            int(df[col].nunique(dropna=True)) for col in df.columns
        ],
        "Duplicates": [
            int(df[col].duplicated().sum()) for col in df.columns
        ]
    })

    st.dataframe(
        quality_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# TARGET SELECTION
# =========================================================

st.markdown(
    '<div class="section-title">🎯 Target Selection</div>',
    unsafe_allow_html=True
)

categorical_candidates = []

for column in df.columns:
    dtype = df[column].dtype

    if (
        dtype == "object"
        or str(dtype) == "category"
        or dtype == "bool"
    ):
        categorical_candidates.append(column)

if categorical_candidates:
    default_target = categorical_candidates[-1]
else:
    default_target = df.columns[-1]

target_column = st.selectbox(
    "Select the column you want to predict:",
    df.columns.tolist(),
    index=df.columns.tolist().index(default_target)
)

st.success(f"🎯 Target Column: **{target_column}**")


# =========================================================
# PROBLEM TYPE DETECTION
# =========================================================

target_dtype = df[target_column].dtype
unique_values = df[target_column].nunique(dropna=True)

if (
    target_dtype == "object"
    or str(target_dtype) == "category"
    or target_dtype == "bool"
):
    detected_problem_type = "CLASSIFICATION"
elif unique_values <= 10:
    detected_problem_type = "CLASSIFICATION"
else:
    detected_problem_type = "REGRESSION"

st.info(
    f"🔍 Automatically detected: **{detected_problem_type}**"
)

problem_type = st.radio(
    "Select Problem Type",
    ["CLASSIFICATION", "REGRESSION"],
    index=0 if detected_problem_type == "CLASSIFICATION" else 1,
    horizontal=True
)


# =========================================================
# TARGET DISTRIBUTION
# =========================================================

st.markdown(
    '<div class="section-title">🎯 Target Distribution</div>',
    unsafe_allow_html=True
)

fig_target, ax_target = plt.subplots(figsize=(9, 4))

if problem_type == "CLASSIFICATION":

    target_counts = df[target_column].astype(str).value_counts()

    ax_target.bar(
        target_counts.index.astype(str),
        target_counts.values
    )

    ax_target.set_title("Target Class Distribution")
    ax_target.set_xlabel("Class")
    ax_target.set_ylabel("Number of Samples")
    plt.setp(ax_target.get_xticklabels(), rotation=30, ha="right")

else:

    numeric_target = pd.to_numeric(
        df[target_column],
        errors="coerce"
    )

    ax_target.hist(
        numeric_target.dropna(),
        bins=20
    )

    ax_target.set_title("Target Value Distribution")
    ax_target.set_xlabel(target_column)
    ax_target.set_ylabel("Frequency")

plt.tight_layout()
st.pyplot(fig_target)
plt.close(fig_target)


# =========================================================
# DATASET EXPLORER
# =========================================================

with st.expander("🔎 Dataset Explorer"):

    explorer_column = st.selectbox(
        "Choose a column to explore",
        df.columns,
        key="explorer_column"
    )

    selected_series = df[explorer_column]

    ec1, ec2, ec3 = st.columns(3)

    with ec1:
        st.metric(
            "Unique",
            int(selected_series.nunique(dropna=True))
        )

    with ec2:
        st.metric(
            "Missing",
            int(selected_series.isnull().sum())
        )

    with ec3:
        st.metric(
            "Non-null",
            int(selected_series.notna().sum())
        )

    if pd.api.types.is_numeric_dtype(selected_series):

        fig_explore, ax_explore = plt.subplots(figsize=(9, 4))

        ax_explore.hist(
            selected_series.dropna(),
            bins=25
        )

        ax_explore.set_title(
            f"Distribution: {explorer_column}"
        )
        ax_explore.set_xlabel(explorer_column)
        ax_explore.set_ylabel("Frequency")

        plt.tight_layout()
        st.pyplot(fig_explore)
        plt.close(fig_explore)

        st.write(
            selected_series.describe().to_frame("Value")
        )

    else:

        counts = (
            selected_series.astype(str)
            .value_counts()
            .head(20)
        )

        fig_explore, ax_explore = plt.subplots(figsize=(9, 4))

        ax_explore.bar(
            counts.index,
            counts.values
        )

        ax_explore.set_title(
            f"Top Categories: {explorer_column}"
        )
        ax_explore.set_xlabel("Category")
        ax_explore.set_ylabel("Count")

        plt.setp(
            ax_explore.get_xticklabels(),
            rotation=35,
            ha="right"
        )

        plt.tight_layout()
        st.pyplot(fig_explore)
        plt.close(fig_explore)


# =========================================================
# CORRELATION HEATMAP
# =========================================================

numeric_df = df.select_dtypes(include=np.number)

if numeric_df.shape[1] >= 2:

    with st.expander("🔥 Explore Feature Correlations"):

        correlation = numeric_df.corr()

        fig_corr, ax_corr = plt.subplots(
            figsize=(10, 7)
        )

        image = ax_corr.imshow(
            correlation,
            aspect="auto"
        )

        ax_corr.set_xticks(
            range(len(correlation.columns))
        )
        ax_corr.set_yticks(
            range(len(correlation.columns))
        )

        ax_corr.set_xticklabels(
            correlation.columns,
            rotation=45,
            ha="right"
        )
        ax_corr.set_yticklabels(
            correlation.columns
        )

        ax_corr.set_title(
            "🔥 Feature Correlation Heatmap"
        )

        fig_corr.colorbar(
            image,
            ax=ax_corr
        )

        plt.tight_layout()
        st.pyplot(fig_corr)
        plt.close(fig_corr)


# =========================================================
# REMOVE MISSING TARGET
# =========================================================

df = df.dropna(
    subset=[target_column]
).copy()

if len(df) < 10:
    st.error(
        "❌ Not enough rows after removing missing target values."
    )
    st.stop()


# =========================================================
# FEATURES AND TARGET
# =========================================================

X = df.drop(columns=[target_column])
y = df[target_column]


if X.shape[1] == 0:
    st.error("❌ Dataset must contain at least one feature column.")
    st.stop()


# =========================================================
# TARGET ENCODING
# =========================================================

label_encoder = None

if problem_type == "CLASSIFICATION":

    if (
        y.dtype == "object"
        or str(y.dtype) == "category"
        or y.dtype == "bool"
    ):

        label_encoder = LabelEncoder()

        y = label_encoder.fit_transform(
            y.astype(str)
        )

    else:
        y = y.to_numpy()

    if len(np.unique(y)) < 2:
        st.error(
            "❌ Classification requires at least 2 target classes."
        )
        st.stop()

else:

    try:
        y = pd.to_numeric(
            y,
            errors="raise"
        ).to_numpy()

    except Exception:
        st.error(
            "❌ Regression target must be numerical."
        )
        st.stop()


# =========================================================
# FEATURE TYPES
# =========================================================

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


# =========================================================
# SESSION INFORMATION
# =========================================================

st.session_state.feature_columns = X.columns.tolist()
st.session_state.numeric_features = numeric_features
st.session_state.categorical_features = categorical_features
st.session_state.target_column = target_column
st.session_state.problem_type = problem_type
st.session_state.label_encoder = label_encoder


# =========================================================
# PREPROCESSING
# =========================================================

transformers = []

if numeric_features:

    transformers.append(
        (
            "numeric",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            ),
            numeric_features
        )
    )

if categorical_features:

    transformers.append(
        (
            "categorical",
            Pipeline(
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
            ),
            categorical_features
        )
    )

if not transformers:
    st.error(
        "❌ No usable feature columns were found."
    )
    st.stop()

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

try:

    if problem_type == "CLASSIFICATION":

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

    else:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )

except Exception:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )


# =========================================================
# MODELS
# =========================================================

if problem_type == "CLASSIFICATION":

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            random_state=random_state
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=random_state
        )
    }

else:

    models = {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=random_state
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            random_state=random_state
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=random_state
        )
    }


# =========================================================
# AUTOML
# =========================================================

st.markdown(
    '<div class="section-title">🚀 AutoML Training</div>',
    unsafe_allow_html=True
)

run_automl = st.button(
    "🚀 Run AutoML",
    type="primary",
    use_container_width=True
)


if run_automl:

    results = []

    best_model_name = None
    best_pipeline = None
    best_predictions = None
    best_score = -np.inf

    progress = st.progress(0)
    status = st.empty()

    total_models = len(models)

    for index, (model_name, model) in enumerate(models.items()):

        status.write(
            f"🔄 Training and validating **{model_name}**..."
        )

        try:

            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        preprocessor
                    ),
                    (
                        "model",
                        model
                    )
                ]
            )

            # -------------------------------------------------
            # 5-FOLD / USER-SELECTED CROSS VALIDATION
            # -------------------------------------------------

            if problem_type == "CLASSIFICATION":

                cv_scores = cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv_folds,
                    scoring="accuracy"
                )

            else:

                cv_scores = cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv_folds,
                    scoring="r2"
                )

            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))

            # -------------------------------------------------
            # FINAL TRAINING ON TRAINING SET
            # -------------------------------------------------

            pipeline.fit(
                X_train,
                y_train
            )

            predictions = pipeline.predict(
                X_test
            )

            # -------------------------------------------------
            # REGRESSION
            # -------------------------------------------------

            if problem_type == "REGRESSION":

                r2 = r2_score(
                    y_test,
                    predictions
                )

                mae = mean_absolute_error(
                    y_test,
                    predictions
                )

                rmse = np.sqrt(
                    mean_squared_error(
                        y_test,
                        predictions
                    )
                )

                results.append({
                    "Model": model_name,
                    "R² Score": round(r2, 4),
                    "CV Mean": round(cv_mean, 4),
                    "CV Std": round(cv_std, 4),
                    "MAE": round(mae, 4),
                    "RMSE": round(rmse, 4)
                })

                # Select model using CV score
                score = cv_mean

            # -------------------------------------------------
            # CLASSIFICATION
            # -------------------------------------------------

            else:

                accuracy = accuracy_score(
                    y_test,
                    predictions
                )

                precision = precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )

                recall = recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )

                f1 = f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )

                results.append({
                    "Model": model_name,
                    "Accuracy": round(accuracy, 4),
                    "CV Mean": round(cv_mean, 4),
                    "CV Std": round(cv_std, 4),
                    "Precision": round(precision, 4),
                    "Recall": round(recall, 4),
                    "F1 Score": round(f1, 4)
                })

                # Select model using CV score
                score = cv_mean

            # -------------------------------------------------
            # BEST MODEL
            # -------------------------------------------------

            if score > best_score:

                best_score = score
                best_model_name = model_name
                best_pipeline = pipeline
                best_predictions = predictions

        except Exception as e:

            st.warning(
                f"⚠️ {model_name} failed: {e}"
            )

        progress.progress(
            (index + 1) / total_models
        )

    status.empty()

    if best_pipeline is None:

        st.error(
            "❌ No model could be trained successfully."
        )
        st.stop()

    # ---------------------------------------------------------
    # SAVE TO SESSION STATE
    # ---------------------------------------------------------

    st.session_state.trained = True
    st.session_state.best_pipeline = best_pipeline
    st.session_state.best_model_name = best_model_name
    st.session_state.results_df = pd.DataFrame(results)
    st.session_state.best_predictions = best_predictions
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.y_all = y

    st.success(
        "🎉 AutoML pipeline completed successfully!"
    )


# =========================================================
# DISPLAY RESULTS
# =========================================================

if st.session_state.trained:

    best_pipeline = st.session_state.best_pipeline
    best_model_name = st.session_state.best_model_name
    results_df = st.session_state.results_df
    best_predictions = st.session_state.best_predictions
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    y_all = st.session_state.y_all


    # =====================================================
    # LEADERBOARD
    # =====================================================

    st.markdown(
        '<div class="section-title">🏆 AutoML Leaderboard</div>',
        unsafe_allow_html=True
    )

    leaderboard = results_df.copy()

    if problem_type == "CLASSIFICATION":
        leaderboard = leaderboard.sort_values(
            "CV Mean",
            ascending=False
        ).reset_index(drop=True)

        leaderboard.insert(
            0,
            "Rank",
            range(1, len(leaderboard) + 1)
        )

    else:
        leaderboard = leaderboard.sort_values(
            "CV Mean",
            ascending=False
        ).reset_index(drop=True)

        leaderboard.insert(
            0,
            "Rank",
            range(1, len(leaderboard) + 1)
        )

    st.dataframe(
        leaderboard,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Models are ranked using the mean cross-validation score."
    )


    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    st.markdown(
        '<div class="section-title">🤖 Model Comparison</div>',
        unsafe_allow_html=True
    )

    if problem_type == "CLASSIFICATION":

        metric_to_plot = "Accuracy"
        ylabel = "Test Accuracy"
        title = "🏆 Model Accuracy Comparison"

    else:

        metric_to_plot = "R² Score"
        ylabel = "Test R² Score"
        title = "🏆 Model R² Comparison"

    fig_model, ax_model = plt.subplots(
        figsize=(10, 5)
    )

    bars = ax_model.bar(
        results_df["Model"],
        results_df[metric_to_plot]
    )

    ax_model.set_ylabel(ylabel)
    ax_model.set_title(title)

    if problem_type == "CLASSIFICATION":
        ax_model.set_ylim(0, 1.05)

    for bar in bars:

        height = bar.get_height()

        ax_model.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom"
        )

    plt.xticks(
        rotation=25,
        ha="right"
    )

    plt.tight_layout()
    st.pyplot(fig_model)
    plt.close(fig_model)


    # =====================================================
    # CROSS VALIDATION GRAPH
    # =====================================================

    st.subheader("🔁 Cross-Validation Performance")

    fig_cv, ax_cv = plt.subplots(
        figsize=(10, 5)
    )

    ax_cv.bar(
        results_df["Model"],
        results_df["CV Mean"]
    )

    ax_cv.set_ylabel(
        "Mean CV Score"
    )

    ax_cv.set_title(
        f"{cv_folds}-Fold Cross-Validation Comparison"
    )

    if problem_type == "CLASSIFICATION":
        ax_cv.set_ylim(0, 1.05)

    plt.xticks(
        rotation=25,
        ha="right"
    )

    plt.tight_layout()
    st.pyplot(fig_cv)
    plt.close(fig_cv)


    # =====================================================
    # BEST MODEL
    # =====================================================

    st.markdown(
        '<div class="section-title">🏆 Best Model</div>',
        unsafe_allow_html=True
    )

    best_row = leaderboard.iloc[0]

    st.success(
        f"🏆 **{best_model_name}**"
    )

    b1, b2, b3 = st.columns(3)

    with b1:
        st.metric(
            "CV Mean",
            f"{best_row['CV Mean']:.4f}"
        )

    with b2:
        st.metric(
            "CV Std",
            f"{best_row['CV Std']:.4f}"
        )

    with b3:

        if problem_type == "CLASSIFICATION":
            st.metric(
                "Test Accuracy",
                f"{best_row['Accuracy']:.4f}"
            )
        else:
            st.metric(
                "Test R²",
                f"{best_row['R² Score']:.4f}"
            )


    # =====================================================
    # FINAL METRICS
    # =====================================================

    st.markdown(
        '<div class="section-title">📈 Final Performance</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # REGRESSION
    # =====================================================

    if problem_type == "REGRESSION":

        final_r2 = r2_score(
            y_test,
            best_predictions
        )

        final_mae = mean_absolute_error(
            y_test,
            best_predictions
        )

        final_rmse = np.sqrt(
            mean_squared_error(
                y_test,
                best_predictions
            )
        )

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric(
                "R² Score",
                f"{final_r2:.4f}"
            )

        with m2:
            st.metric(
                "MAE",
                f"{final_mae:.4f}"
            )

        with m3:
            st.metric(
                "RMSE",
                f"{final_rmse:.4f}"
            )


        # -------------------------------------------------
        # ACTUAL VS PREDICTED
        # -------------------------------------------------

        st.subheader("🎯 Actual vs Predicted")

        fig_actual, ax_actual = plt.subplots(
            figsize=(9, 5)
        )

        ax_actual.scatter(
            y_test,
            best_predictions,
            alpha=0.7
        )

        min_value = min(
            np.min(y_test),
            np.min(best_predictions)
        )

        max_value = max(
            np.max(y_test),
            np.max(best_predictions)
        )

        ax_actual.plot(
            [min_value, max_value],
            [min_value, max_value],
            linestyle="--"
        )

        ax_actual.set_xlabel(
            "Actual Values"
        )

        ax_actual.set_ylabel(
            "Predicted Values"
        )

        ax_actual.set_title(
            "Actual vs Predicted Values"
        )

        plt.tight_layout()
        st.pyplot(fig_actual)
        plt.close(fig_actual)


        # -------------------------------------------------
        # RESIDUAL PLOT
        # -------------------------------------------------

        st.subheader("📉 Residual Analysis")

        residuals = (
            y_test - best_predictions
        )

        fig_residual, ax_residual = plt.subplots(
            figsize=(9, 5)
        )

        ax_residual.scatter(
            best_predictions,
            residuals,
            alpha=0.7
        )

        ax_residual.axhline(
            0,
            linestyle="--"
        )

        ax_residual.set_xlabel(
            "Predicted Values"
        )

        ax_residual.set_ylabel(
            "Residuals"
        )

        ax_residual.set_title(
            "Residual / Error Plot"
        )

        plt.tight_layout()
        st.pyplot(fig_residual)
        plt.close(fig_residual)


        # -------------------------------------------------
        # ERROR DISTRIBUTION
        # -------------------------------------------------

        st.subheader("📊 Error Distribution")

        fig_error, ax_error = plt.subplots(
            figsize=(9, 4)
        )

        ax_error.hist(
            residuals,
            bins=20
        )

        ax_error.set_xlabel(
            "Prediction Error"
        )

        ax_error.set_ylabel(
            "Frequency"
        )

        ax_error.set_title(
            "Distribution of Prediction Errors"
        )

        plt.tight_layout()
        st.pyplot(fig_error)
        plt.close(fig_error)


    # =====================================================
    # CLASSIFICATION
    # =====================================================

    else:

        final_accuracy = accuracy_score(
            y_test,
            best_predictions
        )

        final_precision = precision_score(
            y_test,
            best_predictions,
            average="weighted",
            zero_division=0
        )

        final_recall = recall_score(
            y_test,
            best_predictions,
            average="weighted",
            zero_division=0
        )

        final_f1 = f1_score(
            y_test,
            best_predictions,
            average="weighted",
            zero_division=0
        )

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric(
                "Accuracy",
                f"{final_accuracy:.4f}"
            )

        with m2:
            st.metric(
                "Precision",
                f"{final_precision:.4f}"
            )

        with m3:
            st.metric(
                "Recall",
                f"{final_recall:.4f}"
            )

        with m4:
            st.metric(
                "F1 Score",
                f"{final_f1:.4f}"
            )


        # -------------------------------------------------
        # CLASSIFICATION METRIC GRAPH
        # -------------------------------------------------

        st.subheader("📊 Classification Metrics")

        metric_names = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]

        metric_values = [
            final_accuracy,
            final_precision,
            final_recall,
            final_f1
        ]

        fig_metrics, ax_metrics = plt.subplots(
            figsize=(9, 5)
        )

        bars = ax_metrics.bar(
            metric_names,
            metric_values
        )

        ax_metrics.set_ylim(
            0,
            1.05
        )

        ax_metrics.set_ylabel(
            "Score"
        )

        ax_metrics.set_title(
            "Final Classification Metrics"
        )

        for bar, value in zip(
            bars,
            metric_values
        ):

            ax_metrics.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom"
            )

        plt.tight_layout()
        st.pyplot(fig_metrics)
        plt.close(fig_metrics)


        # -------------------------------------------------
        # CONFUSION MATRIX
        # -------------------------------------------------

        st.subheader("🎯 Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            best_predictions
        )

        fig_cm, ax_cm = plt.subplots(
            figsize=(7, 6)
        )

        image = ax_cm.imshow(
            cm,
            interpolation="nearest"
        )

        ax_cm.set_title(
            "Confusion Matrix"
        )

        ax_cm.set_xlabel(
            "Predicted Label"
        )

        ax_cm.set_ylabel(
            "Actual Label"
        )

        for i in range(cm.shape[0]):

            for j in range(cm.shape[1]):

                ax_cm.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=12
                )

        if label_encoder is not None:

            labels = label_encoder.classes_

            ax_cm.set_xticks(
                range(len(labels))
            )

            ax_cm.set_yticks(
                range(len(labels))
            )

            ax_cm.set_xticklabels(
                labels,
                rotation=30,
                ha="right"
            )

            ax_cm.set_yticklabels(
                labels
            )

        fig_cm.colorbar(
            image,
            ax=ax_cm
        )

        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)


    # =====================================================
    # FEATURE IMPORTANCE / COEFFICIENT IMPORTANCE
    # =====================================================

    st.subheader("🔥 Feature Importance / Influence")

    try:

        model_object = (
            best_pipeline
            .named_steps["model"]
        )

        feature_names = (
            best_pipeline
            .named_steps["preprocessor"]
            .get_feature_names_out()
        )

        importances = None

        if hasattr(
            model_object,
            "feature_importances_"
        ):

            importances = model_object.feature_importances_

        elif hasattr(
            model_object,
            "coef_"
        ):

            coefficients = model_object.coef_

            if coefficients.ndim == 1:
                importances = np.abs(coefficients)
            else:
                importances = np.mean(
                    np.abs(coefficients),
                    axis=0
                )

        if importances is not None:

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            })

            importance_df = (
                importance_df
                .sort_values(
                    "Importance",
                    ascending=False
                )
                .head(top_features)
            )

            fig_imp, ax_imp = plt.subplots(
                figsize=(10, 6)
            )

            ax_imp.barh(
                importance_df["Feature"][::-1],
                importance_df["Importance"][::-1]
            )

            ax_imp.set_xlabel(
                "Importance / Absolute Influence"
            )

            ax_imp.set_title(
                f"Top {top_features} Features"
            )

            plt.tight_layout()
            st.pyplot(fig_imp)
            plt.close(fig_imp)

            with st.expander(
                "📋 View Feature Importance Table"
            ):
                st.dataframe(
                    importance_df,
                    use_container_width=True,
                    hide_index=True
                )

        else:

            st.info(
                "ℹ️ Feature importance is not directly available "
                "for this model."
            )

    except Exception:

        st.info(
            "ℹ️ Feature importance could not be calculated."
        )


    # =====================================================
    # PREDICTION PLAYGROUND
    # =====================================================

    st.markdown(
        '<div class="section-title">🔮 Prediction Playground</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter values for a new sample and use the trained "
        "best model to generate a prediction."
    )

    prediction_inputs = {}

    # Use columns to make the form cleaner
    input_columns = st.columns(2)

    for index, feature in enumerate(X.columns):

        target_container = input_columns[index % 2]

        with target_container:

            if feature in numeric_features:

                numeric_series = pd.to_numeric(
                    X[feature],
                    errors="coerce"
                )

                min_val = float(
                    numeric_series.min()
                )

                max_val = float(
                    numeric_series.max()
                )

                mean_val = float(
                    numeric_series.mean()
                )

                if not np.isfinite(mean_val):

                    prediction_inputs[feature] = st.number_input(
                        feature,
                        value=0.0,
                        key=f"num_{feature}"
                    )

                elif (
                    not np.isfinite(min_val)
                    or not np.isfinite(max_val)
                    or min_val == max_val
                ):

                    prediction_inputs[feature] = st.number_input(
                        feature,
                        value=mean_val,
                        key=f"num_{feature}"
                    )

                else:

                    prediction_inputs[feature] = st.number_input(
                        feature,
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        key=f"num_{feature}"
                    )

            else:

                categories = (
                    X[feature]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if categories:

                    prediction_inputs[feature] = st.selectbox(
                        feature,
                        categories,
                        key=f"cat_{feature}"
                    )

                else:

                    prediction_inputs[feature] = ""


    predict_button = st.button(
        "🔮 Predict New Sample",
        type="primary",
        use_container_width=True
    )


    if predict_button:

        try:

            new_data = pd.DataFrame(
                [prediction_inputs]
            )

            new_data = new_data[
                X.columns
            ]

            prediction = (
                best_pipeline
                .predict(new_data)
            )


            # -------------------------------------------------
            # CLASSIFICATION
            # -------------------------------------------------

            if problem_type == "CLASSIFICATION":

                predicted_value = prediction[0]

                if label_encoder is not None:

                    predicted_class = (
                        label_encoder
                        .inverse_transform(
                            [int(predicted_value)]
                        )[0]
                    )

                else:

                    predicted_class = predicted_value

                st.success(
                    f"🎯 Predicted Class: "
                    f"**{predicted_class}**"
                )


                if hasattr(
                    best_pipeline,
                    "predict_proba"
                ):

                    probabilities = (
                        best_pipeline
                        .predict_proba(new_data)[0]
                    )

                    if label_encoder is not None:

                        class_names = (
                            label_encoder.classes_
                        )

                    else:

                        class_names = (
                            best_pipeline
                            .named_steps["model"]
                            .classes_
                        )

                    probability_df = pd.DataFrame({
                        "Class": class_names,
                        "Probability": probabilities
                    })

                    probability_df[
                        "Probability"
                    ] = probability_df[
                        "Probability"
                    ].round(4)

                    st.subheader(
                        "📊 Prediction Confidence"
                    )

                    st.bar_chart(
                        probability_df.set_index(
                            "Class"
                        )
                    )

                    max_probability = float(
                        np.max(probabilities)
                    )

                    st.metric(
                        "Prediction Confidence",
                        f"{max_probability * 100:.2f}%"
                    )

                    with st.expander(
                        "📋 View Prediction Probabilities"
                    ):

                        st.dataframe(
                            probability_df,
                            use_container_width=True,
                            hide_index=True
                        )


            # -------------------------------------------------
            # REGRESSION
            # -------------------------------------------------

            else:

                predicted_value = float(
                    prediction[0]
                )

                st.success(
                    f"🎯 Predicted Value: "
                    f"**{predicted_value:.4f}**"
                )

                st.metric(
                    "Predicted Value",
                    f"{predicted_value:.4f}"
                )

                target_mean = float(
                    np.mean(y_all)
                )

                target_min = float(
                    np.min(y_all)
                )

                target_max = float(
                    np.max(y_all)
                )

                r1, r2, r3 = st.columns(3)

                with r1:
                    st.metric(
                        "Minimum Target",
                        f"{target_min:.3f}"
                    )

                with r2:
                    st.metric(
                        "Average Target",
                        f"{target_mean:.3f}"
                    )

                with r3:
                    st.metric(
                        "Maximum Target",
                        f"{target_max:.3f}"
                    )

        except Exception as e:

            st.error(
                f"❌ Prediction failed: {e}"
            )


    # =====================================================
    # DOWNLOAD TEST PREDICTIONS
    # =====================================================

    st.markdown(
        '<div class="section-title">📥 Export Predictions</div>',
        unsafe_allow_html=True
    )

    prediction_export = X_test.copy()

    prediction_export["Actual"] = y_test
    prediction_export["Predicted"] = best_predictions

    if problem_type == "CLASSIFICATION" and label_encoder is not None:

        try:
            prediction_export["Actual"] = (
                label_encoder
                .inverse_transform(
                    np.asarray(y_test).astype(int)
                )
            )

            prediction_export["Predicted"] = (
                label_encoder
                .inverse_transform(
                    np.asarray(best_predictions).astype(int)
                )
            )
        except Exception:
            pass

    prediction_csv = prediction_export.to_csv(
        index=False
    )

    st.download_button(
        label="📥 Download Test Predictions CSV",
        data=prediction_csv,
        file_name="automl_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )


    # =====================================================
    # SAVE MODEL
    # =====================================================

    model_data = {
        "model": best_pipeline,
        "target_column": target_column,
        "problem_type": problem_type,
        "label_encoder": label_encoder,
        "feature_columns": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "best_model_name": best_model_name,
        "cv_folds": cv_folds
    }

    model_file = "trained_model.pkl"

    with open(
        model_file,
        "wb"
    ) as file:

        pickle.dump(
            model_data,
            file
        )


    # =====================================================
    # DOWNLOAD MODEL
    # =====================================================

    st.markdown(
        '<div class="section-title">💾 Export Model</div>',
        unsafe_allow_html=True
    )

    with open(
        model_file,
        "rb"
    ) as file:

        st.download_button(
            label="⬇️ Download Trained Model (.pkl)",
            data=file,
            file_name="trained_model.pkl",
            mime="application/octet-stream",
            use_container_width=True
        )


    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    st.markdown(
        '<div class="section-title">📋 Final AutoML Summary</div>',
        unsafe_allow_html=True
    )

    summary = {
        "Dataset Rows": len(df),
        "Dataset Columns": len(df.columns),
        "Target Column": target_column,
        "Problem Type": problem_type,
        "Models Tested": len(results_df),
        "Cross-Validation Folds": cv_folds,
        "Best Model": best_model_name,
        "Best CV Mean": round(
            float(leaderboard.iloc[0]["CV Mean"]),
            4
        ),
        "Best CV Std": round(
            float(leaderboard.iloc[0]["CV Std"]),
            4
        )
    }

    if problem_type == "REGRESSION":

        summary["R² Score"] = round(
            final_r2,
            4
        )

        summary["MAE"] = round(
            final_mae,
            4
        )

        summary["RMSE"] = round(
            final_rmse,
            4
        )

    else:

        summary["Accuracy"] = round(
            final_accuracy,
            4
        )

        summary["Precision"] = round(
            final_precision,
            4
        )

        summary["Recall"] = round(
            final_recall,
            4
        )

        summary["F1 Score"] = round(
            final_f1,
            4
        )


    summary_df = pd.DataFrame(
        list(summary.items()),
        columns=[
            "Property",
            "Value"
        ]
    )

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # DOWNLOAD SUMMARY
    # =====================================================

    summary_csv = summary_df.to_csv(
        index=False
    )

    st.download_button(
        label="📥 Download AutoML Summary CSV",
        data=summary_csv,
        file_name="automl_summary.csv",
        mime="text/csv",
        use_container_width=True
    )


    # =====================================================
    # COMPLETION
    # =====================================================

    st.success(
        "✅ AutoML analysis completed. "
        "Model comparison, cross-validation, analytics, "
        "explainability, prediction playground and exports are ready."
    )
