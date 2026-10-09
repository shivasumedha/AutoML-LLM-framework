from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def train_classification_models(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):
    """
    Train and evaluate multiple classification models.

    Cross-validation is performed only on the
    training data to avoid data leakage.
    """

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                random_state=42
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            ),

        "SVM":
            SVC(
                probability=True,
                random_state=42
            )
    }

    results = {}

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1"
    }

    for model_name, model in models.items():

        print(
            f"\nTraining {model_name}..."
        )

        # Preprocessing + model
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

        # -------------------------------
        # CROSS VALIDATION
        # -------------------------------

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        cv_accuracy = cv_results[
            "test_accuracy"
        ].mean()

        cv_precision = cv_results[
            "test_precision"
        ].mean()

        cv_recall = cv_results[
            "test_recall"
        ].mean()

        cv_f1 = cv_results[
            "test_f1"
        ].mean()

        # -------------------------------
        # FINAL TRAINING
        # -------------------------------

        pipeline.fit(
            X_train,
            y_train
        )

        # -------------------------------
        # TEST PREDICTION
        # -------------------------------

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

        matrix = confusion_matrix(
            y_test,
            y_pred
        )

        results[model_name] = {

            "model": pipeline,

            "cv_accuracy": cv_accuracy,
            "cv_precision": cv_precision,
            "cv_recall": cv_recall,
            "cv_f1": cv_f1,

            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,

            "confusion_matrix": matrix
        }

    return results