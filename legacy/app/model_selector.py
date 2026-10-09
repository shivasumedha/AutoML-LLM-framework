def select_best_model(
    results,
    metric="test_f1"
):
    """
    Automatically select the model with
    the highest selected evaluation metric.
    """

    best_model_name = None
    best_model = None
    best_score = -1

    for model_name, result in results.items():

        score = result[metric]

        if score > best_score:

            best_score = score
            best_model_name = model_name
            best_model = result["model"]

    return (
        best_model_name,
        best_model,
        best_score
    )