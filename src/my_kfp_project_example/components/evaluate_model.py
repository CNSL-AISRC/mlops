from kfp import dsl

@dsl.component
def evaluate_model(model: str) -> str:
    accuracy = 0.95  # Fake accuracy
    print(f"Evaluating {model}, got accuracy={accuracy}")
    return model
