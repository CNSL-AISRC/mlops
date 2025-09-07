from kfp import local
from kfp import dsl

local.init(runner=local.DockerRunner())

@dsl.component
def add(a: int, b: int) -> int:
    return a + b

@dsl.component
def subtract(a: int, b: int) -> int:
    return a - b

@dsl.component
def multiply(a: int, b: int) -> int:
    return a * b

# run a single component
task = add(a=1, b=2)
assert task.output == 3

# or run it in a pipeline
@dsl.pipeline
def math_pipeline(x: int, y: int, z: int) -> int:
    t1 = add(a=x, b=y)
    t2 = subtract(a=t1.output, b=z)
    t3 = multiply(a=t2.output, b=4)
    return t3.output

pipeline_task = math_pipeline(x=1, y=2, z=3)
assert pipeline_task.output == 0
#print(pipeline_task.output)