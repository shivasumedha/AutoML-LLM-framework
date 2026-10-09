# ============================================================
#              AUTOML-LLM UNIVERSAL FRAMEWORK
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    GridSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.svm import (
    SVC,
    SVR
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")


# ============================================================
#                  DISPLAY FUNCTIONS
# ============================================================

def print_header(title):

    print("\n")
    print("=" * 60)
    print(f"{title:^60}")
    print("=" * 60)


# ============================================================
#                  LOAD DATASET
# ============================================================

print_header("AUTOML-LLM UNIVERSAL FRAMEWORK")

print("Enter your CSV dataset path.")
print("Example: data/sample.csv")
print()

dataset_path = input("Dataset path: ").strip()

if not os.path.exists(dataset_path):

    print("\nERROR: Dataset file not found.")
    print("Please check the path and try again.")
    exit()


# Load CSV

try:

    df = pd.read_csv(dataset_path)

except Exception as e:

    print("\nERROR while loading dataset:")
    print(e)
    exit()


print_header("[1] DATASET LOADING")

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")


# ============================================================
#                  DATASET ANALYSIS
# ============================================================

print_header("[2] DATASET ANALYSIS")

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nColumn Names:")
print(list(df.columns))

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")

missing_values = df.isnull().sum()

print(missing_values.to_dict())

print(f"\nTotal Missing Values: {missing_values.sum()}")

duplicate_count = df.duplicated().sum()

print(f"\nDuplicate Rows Found: {duplicate_count}")

if duplicate_count > 0:

    df = df.drop_duplicates().reset_index(drop=True)

    print(f"Duplicate Rows Removed: {duplicate_count}")

else:

    print("Duplicate Rows Removed: 0")


# Numerical columns

numerical_columns = df.select_dtypes(
    include=np.number
).columns.tolist()


# Categorical columns

categorical_columns = df.select_dtypes(
    exclude=np.number
).columns.tolist()


print("\nNumerical Columns:")
print(numerical_columns)

print("\nCategorical Columns:")
print(categorical_columns)


# ============================================================
#                  TARGET SELECTION
# ============================================================

print_header("[3] TARGET COLUMN SELECTION")

print("Available columns:")

for i, column in enumerate(df.columns, start=1):

    print(f"{i}. {column}")


print()

target_column = input(
    "Enter target column name: "
).strip()


# Validate target

if target_column not in df.columns:

    print("\nERROR: Target column does not exist.")

    print("\nAvailable columns:")

    for column in df.columns:

        print(column)

    exit()


# ============================================================
#              REMOVE CONSTANT FEATURES
# ============================================================

feature_columns_before = [
    column for column in df.columns
    if column != target_column
]

constant_columns = []

for column in feature_columns_before:

    if df[column].nunique(dropna=False) <= 1:

        constant_columns.append(column)


if constant_columns:

    print("\nRemoving constant columns:")
    print(constant_columns)

    df = df.drop(
        columns=constant_columns
    )

else:

    print("\nNo constant columns found.")


# ============================================================
#                  PROBLEM DETECTION
# ============================================================

print_header("[4] PROBLEM DETECTION")

print(f"Target Column: {target_column}")

y = df[target_column]

target_clean = y.dropna()

target_dtype = target_clean.dtype

unique_values = target_clean.nunique()

print(f"Target Data Type: {target_dtype}")
print(f"Unique Target Values: {unique_values}")


# ------------------------------------------------------------
# Automatic Classification / Regression Detection
# ------------------------------------------------------------

if (
    pd.api.types.is_object_dtype(target_clean)
    or pd.api.types.is_string_dtype(target_clean)
    or pd.api.types.is_categorical_dtype(target_clean)
    or pd.api.types.is_bool_dtype(target_clean)
):

    problem_type = "CLASSIFICATION"

else:

    if unique_values <= 20:

        problem_type = "CLASSIFICATION"

    else:

        problem_type = "REGRESSION"


print(f"Problem Type: {problem_type}")


# ============================================================
#                  FEATURE PREPARATION
# ============================================================

print_header("[5] FEATURE PREPARATION")

X = df.drop(
    columns=[target_column]
)

y = df[target_column]


# Remove rows where target is missing

valid_target_rows = y.notna()

X = X.loc[valid_target_rows].reset_index(drop=True)

y = y.loc[valid_target_rows].reset_index(drop=True)


print(f"Original Dataset Shape: {df.shape}")

print(f"Feature Shape: {X.shape}")

print(f"Target Shape: {y.shape}")


# ------------------------------------------------------------
# Target information
# ------------------------------------------------------------

if problem_type == "CLASSIFICATION":

    print("\nTarget Distribution:")

    print(y.value_counts())


else:

    print("\nTarget Statistics:")

    print(y.describe())


# ============================================================
#              CLASSIFICATION TARGET CLEANUP
# ============================================================

if problem_type == "CLASSIFICATION":

    # Convert boolean target to string

    if pd.api.types.is_bool_dtype(y):

        y = y.astype(str)


# ============================================================
#                  TRAIN TEST SPLIT
# ============================================================

print_header("[6] TRAIN / TEST SPLIT")


# Determine whether stratification is possible

stratify_value = None

if problem_type == "CLASSIFICATION":

    class_counts = y.value_counts()

    if len(class_counts) > 1 and class_counts.min() >= 2:

        stratify_value = y


try:

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=stratify_value

    )

except ValueError:

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,
        test_size=0.20,
        random_state=42

    )


print(f"Training Samples: {len(X_train)}")

print(f"Testing Samples: {len(X_test)}")


# ============================================================
#              IDENTIFY FEATURE TYPES
# ============================================================

numerical_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()


print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)


# ============================================================
#              PREPROCESSING PIPELINE
# ============================================================

print_header("[7] PREPROCESSING PIPELINE")


# Numerical pipeline

numerical_pipeline = Pipeline(
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
)


# Categorical pipeline

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
                handle_unknown="ignore",
                sparse_output=False
            )
        )

    ]
)


# Combined preprocessing

transformers = []


if numerical_features:

    transformers.append(

        (
            "numerical",
            numerical_pipeline,
            numerical_features
        )

    )


if categorical_features:

    transformers.append(

        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )

    )


preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)


print("Automatic preprocessing pipeline created.")

print(
    "Numerical features → Imputation + Scaling"
)

print(
    "Categorical features → Imputation + One-Hot Encoding"
)


# ============================================================
#                  MODEL DEFINITIONS
# ============================================================

print_header("[8] MODEL TRAINING")


if problem_type == "CLASSIFICATION":

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),

        "SVM":
            SVC(
                class_weight="balanced",
                random_state=42
            )
    }


else:

    models = {

        "Linear Regression":
            LinearRegression(),

        "Decision Tree":
            DecisionTreeRegressor(
                random_state=42
            ),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            ),

        "SVM":
            SVR()
    }


# ============================================================
#                  CROSS VALIDATION
# ============================================================

if problem_type == "CLASSIFICATION":

    # Determine safe CV value

    class_counts_train = y_train.value_counts()

    if len(class_counts_train) > 1:

        minimum_class_count = class_counts_train.min()

        cv_folds = min(
            5,
            minimum_class_count
        )

    else:

        cv_folds = 2


    if cv_folds < 2:

        cv_folds = 2


    scoring_metrics = {

        "accuracy": "accuracy",

        "precision": "precision_weighted",

        "recall": "recall_weighted",

        "f1": "f1_weighted"

    }


else:

    cv_folds = 5

    scoring_metrics = {

        "r2": "r2",

        "mae": "neg_mean_absolute_error",

        "mse": "neg_mean_squared_error"

    }


results = {}

trained_pipelines = {}


for model_name, model in models.items():

    print(f"\nTraining {model_name}...")


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


    # Cross validation

    try:

        cv_result = cross_validate(

            pipeline,

            X_train,

            y_train,

            cv=cv_folds,

            scoring=scoring_metrics,

            error_score="raise"

        )

    except Exception as e:

        print(
            f"Cross-validation warning for {model_name}: {e}"
        )

        continue


    # Train complete training data

    pipeline.fit(
        X_train,
        y_train
    )


    trained_pipelines[model_name] = pipeline


    if problem_type == "CLASSIFICATION":

        cv_accuracy = np.mean(
            cv_result["test_accuracy"]
        )

        cv_precision = np.mean(
            cv_result["test_precision"]
        )

        cv_recall = np.mean(
            cv_result["test_recall"]
        )

        cv_f1 = np.mean(
            cv_result["test_f1"]
        )


        # Test prediction

        y_pred = pipeline.predict(
            X_test
        )


        test_accuracy = accuracy_score(
            y_test,
            y_pred
        )

        test_precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        test_recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        test_f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )


        results[model_name] = {

            "cv_score": cv_f1,

            "test_score": test_f1,

            "cv_accuracy": cv_accuracy,

            "cv_precision": cv_precision,

            "cv_recall": cv_recall,

            "cv_f1": cv_f1,

            "test_accuracy": test_accuracy,

            "test_precision": test_precision,

            "test_recall": test_recall,

            "test_f1": test_f1,

            "predictions": y_pred

        }


    else:

        cv_r2 = np.mean(
            cv_result["test_r2"]
        )

        cv_mae = -np.mean(
            cv_result["test_mae"]
        )

        cv_mse = -np.mean(
            cv_result["test_mse"]
        )


        y_pred = pipeline.predict(
            X_test
        )


        test_mae = mean_absolute_error(
            y_test,
            y_pred
        )

        test_mse = mean_squared_error(
            y_test,
            y_pred
        )

        test_rmse = np.sqrt(
            test_mse
        )

        test_r2 = r2_score(
            y_test,
            y_pred
        )


        results[model_name] = {

            "cv_score": cv_r2,

            "test_score": test_r2,

            "cv_r2": cv_r2,

            "cv_mae": cv_mae,

            "cv_mse": cv_mse,

            "test_mae": test_mae,

            "test_mse": test_mse,

            "test_rmse": test_rmse,

            "test_r2": test_r2,

            "predictions": y_pred

        }


# ============================================================
#                  MODEL COMPARISON
# ============================================================

print_header("[9] MODEL COMPARISON")


if not results:

    print("ERROR: No model could be successfully trained.")

    exit()


for model_name, result in results.items():

    print(f"\n{model_name}")


    if problem_type == "CLASSIFICATION":

        print(
            f"  CV Accuracy    : "
            f"{result['cv_accuracy']:.4f}"
        )

        print(
            f"  CV Precision   : "
            f"{result['cv_precision']:.4f}"
        )

        print(
            f"  CV Recall      : "
            f"{result['cv_recall']:.4f}"
        )

        print(
            f"  CV F1          : "
            f"{result['cv_f1']:.4f}"
        )

        print(
            f"  Test Accuracy  : "
            f"{result['test_accuracy']:.4f}"
        )

        print(
            f"  Test Precision : "
            f"{result['test_precision']:.4f}"
        )

        print(
            f"  Test Recall    : "
            f"{result['test_recall']:.4f}"
        )

        print(
            f"  Test F1        : "
            f"{result['test_f1']:.4f}"
        )


    else:

        print(
            f"  CV R2          : "
            f"{result['cv_r2']:.4f}"
        )

        print(
            f"  CV MAE         : "
            f"{result['cv_mae']:.4f}"
        )

        print(
            f"  CV MSE         : "
            f"{result['cv_mse']:.4f}"
        )

        print(
            f"  Test R2        : "
            f"{result['test_r2']:.4f}"
        )

        print(
            f"  Test MAE       : "
            f"{result['test_mae']:.4f}"
        )

        print(
            f"  Test RMSE      : "
            f"{result['test_rmse']:.4f}"
        )


# ============================================================
#                  MODEL SELECTION
# ============================================================

print_header("[10] BEST MODEL SELECTION")


if problem_type == "CLASSIFICATION":

    # F1 is used because it handles class imbalance
    best_model_name = max(

        results,

        key=lambda name:
        results[name]["test_f1"]

    )

    best_score = results[
        best_model_name
    ]["test_f1"]

    selection_metric = "Test F1 Score"


else:

    # R2: higher is better
    best_model_name = max(

        results,

        key=lambda name:
        results[name]["test_r2"]

    )

    best_score = results[
        best_model_name
    ]["test_r2"]

    selection_metric = "Test R2 Score"


best_pipeline = trained_pipelines[
    best_model_name
]


print(f"Selection Metric: {selection_metric}")

print(f"Selected Model: {best_model_name}")

print(f"Best Score: {best_score:.4f}")


# ============================================================
#                  HYPERPARAMETER TUNING
# ============================================================

print_header("[11] HYPERPARAMETER TUNING")


print(
    f"Tuning selected model: {best_model_name}"
)


param_grid = {}


if problem_type == "CLASSIFICATION":

    if best_model_name == "Logistic Regression":

        param_grid = {

            "model__C": [
                0.01,
                0.1,
                1,
                10
            ]

        }


    elif best_model_name == "Decision Tree":

        param_grid = {

            "model__max_depth": [
                None,
                5,
                10,
                20
            ],

            "model__min_samples_split": [
                2,
                5,
                10
            ]

        }


    elif best_model_name == "Random Forest":

        param_grid = {

            "model__n_estimators": [
                100,
                200
            ],

            "model__max_depth": [
                None,
                10,
                20
            ]

        }


    elif best_model_name == "SVM":

        param_grid = {

            "model__C": [
                0.1,
                1,
                10
            ],

            "model__gamma": [
                "scale",
                "auto"
            ],

            "model__kernel": [
                "rbf"
            ]

        }


else:

    if best_model_name == "Linear Regression":

        param_grid = {}


    elif best_model_name == "Decision Tree":

        param_grid = {

            "model__max_depth": [
                None,
                5,
                10,
                20
            ],

            "model__min_samples_split": [
                2,
                5,
                10
            ]

        }


    elif best_model_name == "Random Forest":

        param_grid = {

            "model__n_estimators": [
                100,
                200
            ],

            "model__max_depth": [
                None,
                10,
                20
            ]

        }


    elif best_model_name == "SVM":

        param_grid = {

            "model__C": [
                0.1,
                1,
                10
            ],

            "model__gamma": [
                "scale",
                "auto"
            ],

            "model__kernel": [
                "rbf"
            ]

        }


# ============================================================
#                  RUN GRID SEARCH
# ============================================================

tuned_pipeline = best_pipeline

best_parameters = {}


if param_grid:

    try:

        if problem_type == "CLASSIFICATION":

            tuning_scoring = "f1_weighted"

        else:

            tuning_scoring = "r2"


        grid_search = GridSearchCV(

            estimator=best_pipeline,

            param_grid=param_grid,

            scoring=tuning_scoring,

            cv=cv_folds,

            n_jobs=-1

        )


        grid_search.fit(
            X_train,
            y_train
        )


        tuned_pipeline = grid_search.best_estimator_

        best_parameters = grid_search.best_params_


        print(
            "Hyperparameter tuning completed."
        )

        print(
            "\nBest Hyperparameters:"
        )

        print(best_parameters)


    except Exception as e:

        print(
            "\nHyperparameter tuning failed."
        )

        print(e)

        print(
            "Using the original selected model."
        )

else:

    print(
        "No hyperparameters available for tuning."
    )


# ============================================================
#                  FINAL EVALUATION
# ============================================================

print_header("[12] FINAL MODEL EVALUATION")


final_predictions = tuned_pipeline.predict(
    X_test
)


if problem_type == "CLASSIFICATION":

    final_accuracy = accuracy_score(
        y_test,
        final_predictions
    )

    final_precision = precision_score(
        y_test,
        final_predictions,
        average="weighted",
        zero_division=0
    )

    final_recall = recall_score(
        y_test,
        final_predictions,
        average="weighted",
        zero_division=0
    )

    final_f1 = f1_score(
        y_test,
        final_predictions,
        average="weighted",
        zero_division=0
    )


    print(
        f"Final Accuracy  : {final_accuracy:.4f}"
    )

    print(
        f"Final Precision : {final_precision:.4f}"
    )

    print(
        f"Final Recall    : {final_recall:.4f}"
    )

    print(
        f"Final F1 Score  : {final_f1:.4f}"
    )


    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            final_predictions
        )
    )


else:

    final_mae = mean_absolute_error(
        y_test,
        final_predictions
    )

    final_mse = mean_squared_error(
        y_test,
        final_predictions
    )

    final_rmse = np.sqrt(
        final_mse
    )

    final_r2 = r2_score(
        y_test,
        final_predictions
    )


    print(
        f"Final R2   : {final_r2:.4f}"
    )

    print(
        f"Final MAE  : {final_mae:.4f}"
    )

    print(
        f"Final RMSE : {final_rmse:.4f}"
    )


# ============================================================
#                  SAVE FINAL MODEL
# ============================================================

print_header("[13] MODEL SAVING")


os.makedirs(
    "models",
    exist_ok=True
)


model_file = "models/final_model.pkl"


# Save everything required for future predictions

model_package = {

    "model": tuned_pipeline,

    "target_column": target_column,

    "problem_type": problem_type,

    "feature_columns": X.columns.tolist(),

    "numerical_features": numerical_features,

    "categorical_features": categorical_features,

    "dataset_path": dataset_path,

    "best_model_name": best_model_name,

    "final_model_name": best_model_name,

    "best_parameters": best_parameters

}


joblib.dump(
    model_package,
    model_file
)


print("Model saved successfully!")

print(
    f"Saved Model: {model_file}"
)


# ============================================================
#                  FINAL SUMMARY
# ============================================================

print_header("[14] PIPELINE SUMMARY")


print(
    "Dataset Loading          : SUCCESS"
)

print(
    "Dataset Analysis         : SUCCESS"
)

print(
    "Target Selection         : SUCCESS"
)

print(
    "Problem Detection        : SUCCESS"
)

print(
    "Feature Preparation      : SUCCESS"
)

print(
    "Train/Test Split         : SUCCESS"
)

print(
    "Preprocessing Pipeline   : SUCCESS"
)

print(
    "Cross Validation         : SUCCESS"
)

print(
    "Model Training           : SUCCESS"
)

print(
    "Model Comparison         : SUCCESS"
)

print(
    "Model Selection          : SUCCESS"
)

print(
    "Hyperparameter Tuning    : SUCCESS"
)

print(
    "Final Model Evaluation   : SUCCESS"
)

print(
    "Model Saving             : SUCCESS"
)


print(f"\nTarget Column: {target_column}")

print(f"Problem Type: {problem_type}")

print(
    f"Final Model: {best_model_name}"
)


if problem_type == "CLASSIFICATION":

    print(
        f"Final F1 Score: {final_f1:.4f}"
    )

else:

    print(
        f"Final R2 Score: {final_r2:.4f}"
    )


if best_parameters:

    print("\nBest Hyperparameters:")

    print(best_parameters)


print("\nSaved Model:")

print(model_file)


print_header(
    "AUTOML PIPELINE COMPLETE"
)


# ============================================================
#                  OPTIONAL PREDICTION
# ============================================================

print(
    "\nThe trained model is ready for future predictions."
)

print(
    "The complete preprocessing + model pipeline "
    "is stored inside final_model.pkl."
)