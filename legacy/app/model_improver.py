from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold, cross_validate


def train_improved_models(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):
    """
    Train models with class-weight balancing.

    This is useful when the target classes are imbalanced.
    """

    models = {

        "Balanced Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ),

        "Balanced Decision Tree":
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=42
            ),

        "Balanced Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),

        "Balanced SVM":
            SVC(
                class_weight="balanced",
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

        # Cross-validation
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        # Train final model
        pipeline.fit(
            X_train,
            y_train
        )

        # Test prediction
        y_pred = pipeline.predict(
            X_test
        )

        results[model_name] = {
            "model": pipeline,
            "y_pred": y_pred,

            "cv_accuracy":
                cv_results[
                    "test_accuracy"
                ].mean(),

            "cv_precision":
                cv_results[
                    "test_precision"
                ].mean(),

            "cv_recall":
                cv_results[
                    "test_recall"
                ].mean(),

            "cv_f1":
                cv_results[
                    "test_f1"
                ].mean()
        }

    return results