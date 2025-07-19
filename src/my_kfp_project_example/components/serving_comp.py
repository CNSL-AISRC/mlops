from kfp import dsl

@dsl.component
def serving_component(model: str) -> str:
    print(f"Serving {model}")
    return model


