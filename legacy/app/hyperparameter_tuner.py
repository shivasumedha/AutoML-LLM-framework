from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


def tune_svm(
    X_train,
    y_train,
    preprocessor
):
    """
    Hyperparameter tuning for a class-balanced SVM.

    GridSearchCV tests multiple parameter combinations
    and selects the best combination using F1 score.
    """

    print("\n========================================")
    print("       HYPERPARAMETER TUNING")
    print("========================================")

    # SVM model
    svm = SVC(
        class_weight="balanced",
        random_state=42
    )

    # Complete preprocessing + model pipeline
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                svm
            )
        ]
    )

    # Parameters to test
    param_grid = {
        "model__C": [
            0.1,
            1,
            10,
            100
        ],

        "model__gamma": [
            "scale",
            "auto",
            0.01,
            0.1
        ],

        "model__kernel": [
            "rbf",
            "linear"
        ]
    }

    print("\nParameters being tested:")

    print(
        "C:",
        param_grid["model__C"]
    )

    print(
        "Gamma:",
        param_grid["model__gamma"]
    )

    print(
        "Kernel:",
        param_grid["model__kernel"]
    )

    print(
        "\nTotal parameter combinations:",
        4 * 4 * 2
    )

    print(
        "\nRunning 5-fold cross-validation..."
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    # Start tuning
    grid_search.fit(
        X_train,
        y_train
    )

    print("\n========================================")
    print("       TUNING COMPLETE")
    print("========================================")

    print(
        "\nBest Parameters:"
    )

    print(
        grid_search.best_params_
    )

    print(
        "\nBest Cross-Validation F1:"
    )

    print(
        f"{grid_search.best_score_:.4f}"
    )

    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        grid_search.best_score_
    )