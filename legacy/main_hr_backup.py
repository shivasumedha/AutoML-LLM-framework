import os
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.model_selection import train_test_split

from app.data_loader import load_dataset
from app.analyzer import analyze_dataset
from app.problem_detector import detect_problem_type

from app.preprocessor import (
    prepare_features_and_target,
    create_preprocessor
)

from app.model_trainer import (
    train_classification_models
)

from app.model_selector import (
    select_best_model
)

from app.model_improver import (
    train_improved_models
)

from app.hyperparameter_tuner import (
    tune_svm
)


# ==========================================================
# CONFIGURATION
# ==========================================================

DATASET_PATH = (
    "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

TARGET_COLUMN = "Attrition"

MODEL_PATH = "models/final_model.pkl"


# ==========================================================
# PROJECT HEADER
# ==========================================================

print("\n========================================")
print("       AUTOML-LLM FRAMEWORK")
print("========================================")


# ==========================================================
# STEP 1: DATASET LOADING
# ==========================================================

print("\n[1] DATASET LOADING")

df = load_dataset(DATASET_PATH)

if df is None:
    raise SystemExit(
        "Dataset loading failed."
    )

print(
    "Dataset shape:",
    df.shape
)


# ==========================================================
# STEP 2: DATASET ANALYSIS
# ==========================================================

print("\n========================================")
print("       DATASET ANALYSIS")
print("========================================")

analysis = analyze_dataset(df)

print(
    f"Rows: {analysis['rows']}"
)

print(
    f"Columns: {analysis['columns']}"
)

print("\nColumn Names:")
print(
    analysis["column_names"]
)

print("\nMissing Values:")

missing_values = analysis[
    "missing_values"
]

missing_total = sum(
    missing_values.values()
)

print(
    f"Total Missing Values: {missing_total}"
)

print(
    missing_values
)

print("\nDuplicate Rows:")

print(
    analysis["duplicate_rows"]
)

print("\nNumerical Columns:")

print(
    analysis["numerical_columns"]
)

print("\nCategorical Columns:")

print(
    analysis["categorical_columns"]
)


# ==========================================================
# STEP 3: PROBLEM DETECTION
# ==========================================================

print("\n========================================")
print("       PROBLEM DETECTION")
print("========================================")

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' "
        "does not exist."
    )

problem_type = detect_problem_type(
    df[TARGET_COLUMN]
)

print(
    "Target Column:",
    TARGET_COLUMN
)

print(
    "Problem Type:",
    problem_type
)


# ==========================================================
# STEP 4: FEATURE PREPARATION
# ==========================================================

print("\n========================================")
print("       FEATURE PREPARATION")
print("========================================")

X, y = prepare_features_and_target(
    df,
    TARGET_COLUMN
)

print(
    "Original Dataset Shape:",
    df.shape
)

print(
    "Feature Shape:",
    X.shape
)

print(
    "Target Shape:",
    y.shape
)

print("\nTarget Distribution:")

print(
    y.value_counts()
)


# ==========================================================
# STEP 5: TRAIN / TEST SPLIT
# ==========================================================

print("\n========================================")
print("       TRAIN / TEST SPLIT")
print("========================================")

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)

print(
    "Training Samples:",
    len(X_train)
)

print(
    "Testing Samples:",
    len(X_test)
)


# ==========================================================
# STEP 6: PREPROCESSING PIPELINE
# ==========================================================

print("\n========================================")
print("       PREPROCESSING PIPELINE")
print("========================================")

preprocessor = create_preprocessor(
    X_train
)

print(
    "Automatic preprocessing pipeline created."
)

print(
    "Numerical features → Imputation + Scaling"
)

print(
    "Categorical features → Imputation + One-Hot Encoding"
)


# ==========================================================
# STEP 7: NORMAL MODEL TRAINING
# ==========================================================

print("\n========================================")
print("       MODEL TRAINING")
print("========================================")

if problem_type != "Classification":
    raise ValueError(
        "This framework currently supports "
        "classification problems."
    )

results = train_classification_models(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
)


# ==========================================================
# STEP 8: NORMAL MODEL COMPARISON
# ==========================================================

print("\n========================================")
print("       MODEL COMPARISON")
print("========================================")

for model_name, result in results.items():

    print(
        f"\n{model_name}"
    )

    print(
        "  CV Accuracy : "
        f"{result['cv_accuracy']:.4f}"
    )

    print(
        "  CV Precision: "
        f"{result['cv_precision']:.4f}"
    )

    print(
        "  CV Recall   : "
        f"{result['cv_recall']:.4f}"
    )

    print(
        "  CV F1       : "
        f"{result['cv_f1']:.4f}"
    )

    print(
        "  Test Accuracy : "
        f"{result['test_accuracy']:.4f}"
    )

    print(
        "  Test Precision: "
        f"{result['test_precision']:.4f}"
    )

    print(
        "  Test Recall   : "
        f"{result['test_recall']:.4f}"
    )

    print(
        "  Test F1       : "
        f"{result['test_f1']:.4f}"
    )


# ==========================================================
# STEP 9: NORMAL MODEL SELECTION
# ==========================================================

print("\n========================================")
print("       NORMAL MODEL SELECTION")
print("========================================")

(
    best_model_name,
    best_model,
    best_score
) = select_best_model(
    results,
    metric="test_f1"
)

print(
    "Selection Metric: Test F1 Score"
)

print(
    "Selected Model:",
    best_model_name
)

print(
    "Best F1 Score:",
    f"{best_score:.4f}"
)


# ==========================================================
# STEP 10: NORMAL CONFUSION MATRIX
# ==========================================================

best_result = results[
    best_model_name
]

print("\n========================================")
print("       NORMAL CONFUSION MATRIX")
print("========================================")

print(
    best_result["confusion_matrix"]
)


# ==========================================================
# STEP 11: MODEL IMPROVEMENT
# ==========================================================

print("\n========================================")
print("       MODEL IMPROVEMENT")
print("========================================")

print(
    "Training class-balanced models..."
)

improved_results = train_improved_models(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
)


# ==========================================================
# STEP 12: IMPROVED MODEL COMPARISON
# ==========================================================

print("\n========================================")
print("       IMPROVED MODEL COMPARISON")
print("========================================")

best_improved_model = None
best_improved_name = None
best_improved_f1 = -1

for model_name, result in improved_results.items():

    y_pred = result["y_pred"]

    test_accuracy = accuracy_score(
        y_test,
        y_pred
    )

    test_precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    test_recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print(
        f"\n{model_name}"
    )

    print(
        "  CV Accuracy : "
        f"{result['cv_accuracy']:.4f}"
    )

    print(
        "  CV Precision: "
        f"{result['cv_precision']:.4f}"
    )

    print(
        "  CV Recall   : "
        f"{result['cv_recall']:.4f}"
    )

    print(
        "  CV F1       : "
        f"{result['cv_f1']:.4f}"
    )

    print(
        "  Test Accuracy : "
        f"{test_accuracy:.4f}"
    )

    print(
        "  Test Precision: "
        f"{test_precision:.4f}"
    )

    print(
        "  Test Recall   : "
        f"{test_recall:.4f}"
    )

    print(
        "  Test F1       : "
        f"{test_f1:.4f}"
    )

    if test_f1 > best_improved_f1:

        best_improved_f1 = test_f1

        best_improved_model = (
            result["model"]
        )

        best_improved_name = (
            model_name
        )


# ==========================================================
# STEP 13: IMPROVED MODEL SELECTION
# ==========================================================

print("\n========================================")
print("       IMPROVED MODEL SELECTION")
print("========================================")

print(
    "Selection Metric: Test F1 Score"
)

print(
    "Selected Improved Model:",
    best_improved_name
)

print(
    "Improved F1 Score:",
    f"{best_improved_f1:.4f}"
)


# ==========================================================
# STEP 14: IMPROVED CONFUSION MATRIX
# ==========================================================

improved_best_prediction = (
    improved_results[
        best_improved_name
    ]["y_pred"]
)

improved_confusion_matrix = (
    confusion_matrix(
        y_test,
        improved_best_prediction
    )
)

print("\n========================================")
print("       IMPROVED CONFUSION MATRIX")
print("========================================")

print(
    improved_confusion_matrix
)


# ==========================================================
# STEP 15: OLD VS IMPROVED MODEL
# ==========================================================

print("\n========================================")
print("       OLD VS IMPROVED MODEL")
print("========================================")

print(
    f"Original Model : "
    f"{best_model_name}"
)

print(
    f"Original F1    : "
    f"{best_score:.4f}"
)

print(
    f"Improved Model : "
    f"{best_improved_name}"
)

print(
    f"Improved F1    : "
    f"{best_improved_f1:.4f}"
)

f1_difference = (
    best_improved_f1 - best_score
)

print(
    f"F1 Difference  : "
    f"{f1_difference:+.4f}"
)


# ==========================================================
# STEP 16: HYPERPARAMETER TUNING
# ==========================================================

print("\n========================================")
print("       HYPERPARAMETER TUNING")
print("========================================")

print(
    "Tuning Balanced SVM..."
)

(
    tuned_model,
    best_parameters,
    best_cv_f1
) = tune_svm(
    X_train,
    y_train,
    preprocessor
)


# ==========================================================
# STEP 17: TUNED MODEL EVALUATION
# ==========================================================

print("\n========================================")
print("       TUNED MODEL EVALUATION")
print("========================================")

tuned_predictions = (
    tuned_model.predict(
        X_test
    )
)

tuned_accuracy = accuracy_score(
    y_test,
    tuned_predictions
)

tuned_precision = precision_score(
    y_test,
    tuned_predictions,
    zero_division=0
)

tuned_recall = recall_score(
    y_test,
    tuned_predictions,
    zero_division=0
)

tuned_f1 = f1_score(
    y_test,
    tuned_predictions,
    zero_division=0
)

print(
    "\nBest Parameters:"
)

print(
    best_parameters
)

print(
    "\nBest Cross-Validation F1:"
)

print(
    f"{best_cv_f1:.4f}"
)

print(
    "\nTest Accuracy:"
)

print(
    f"{tuned_accuracy:.4f}"
)

print(
    "Test Precision:"
)

print(
    f"{tuned_precision:.4f}"
)

print(
    "Test Recall:"
)

print(
    f"{tuned_recall:.4f}"
)

print(
    "Test F1:"
)

print(
    f"{tuned_f1:.4f}"
)


# ==========================================================
# STEP 18: TUNING COMPARISON
# ==========================================================

print("\n========================================")
print("       TUNING COMPARISON")
print("========================================")

print(
    f"Before Tuning F1 : "
    f"{best_improved_f1:.4f}"
)

print(
    f"After Tuning F1  : "
    f"{tuned_f1:.4f}"
)

tuning_difference = (
    tuned_f1 - best_improved_f1
)

print(
    f"F1 Difference    : "
    f"{tuning_difference:+.4f}"
)


# ==========================================================
# STEP 19: FINAL MODEL SELECTION
# ==========================================================

print("\n========================================")
print("       FINAL MODEL SELECTION")
print("========================================")

if tuned_f1 >= best_improved_f1:

    final_model = tuned_model

    final_model_name = (
        "Tuned Balanced SVM"
    )

    final_f1 = tuned_f1

    final_predictions = (
        tuned_predictions
    )

    final_source = (
        "Hyperparameter Tuning"
    )

else:

    final_model = best_improved_model

    final_model_name = (
        best_improved_name
    )

    final_f1 = best_improved_f1

    final_predictions = (
        improved_best_prediction
    )

    final_source = (
        "Model Improvement"
    )

print(
    "Final Model Source:",
    final_source
)

print(
    "Final Model:",
    final_model_name
)

print(
    "Final F1 Score:",
    f"{final_f1:.4f}"
)


# ==========================================================
# STEP 20: FINAL CONFUSION MATRIX
# ==========================================================

final_confusion_matrix = (
    confusion_matrix(
        y_test,
        final_predictions
    )
)

print("\n========================================")
print("       FINAL CONFUSION MATRIX")
print("========================================")

print(
    final_confusion_matrix
)


# ==========================================================
# STEP 21: SAVE FINAL MODEL
# ==========================================================

print("\n========================================")
print("       SAVING FINAL MODEL")
print("========================================")

# Make sure models folder exists
os.makedirs(
    "models",
    exist_ok=True
)

# Save complete trained pipeline
joblib.dump(
    final_model,
    MODEL_PATH
)

print(
    "Final model saved successfully!"
)

print(
    "Model Path:",
    MODEL_PATH
)


# ==========================================================
# STEP 22: VERIFY SAVED MODEL
# ==========================================================

print("\n========================================")
print("       VERIFYING SAVED MODEL")
print("========================================")

if os.path.exists(MODEL_PATH):

    model_size = os.path.getsize(
        MODEL_PATH
    )

    print(
        "Model file exists: YES"
    )

    print(
        "Model file size:",
        f"{model_size / 1024:.2f} KB"
    )

else:

    print(
        "Model file exists: NO"
    )


# ==========================================================
# STEP 23: PIPELINE SUMMARY
# ==========================================================

print("\n========================================")
print("       PIPELINE SUMMARY")
print("========================================")

print(
    "Dataset Loading          : SUCCESS"
)

print(
    "Dataset Analysis         : SUCCESS"
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
    "Model Evaluation         : SUCCESS"
)

print(
    "Model Improvement        : SUCCESS"
)

print(
    "Hyperparameter Tuning    : SUCCESS"
)

print(
    "Final Model Selection    : SUCCESS"
)

print(
    "Model Saving             : SUCCESS"
)

print(
    "\nTarget Column:",
    TARGET_COLUMN
)

print(
    "Problem Type:",
    problem_type
)

print(
    "Final Model:",
    final_model_name
)

print(
    "Final F1 Score:",
    f"{final_f1:.4f}"
)

print(
    "\nBest Hyperparameters:"
)

print(
    best_parameters
)

print(
    "\nSaved Model:"
)

print(
    MODEL_PATH
)


# ==========================================================
# PROJECT COMPLETE
# ==========================================================

print("\n========================================")
print("       AUTOML PIPELINE COMPLETE")
print("========================================")