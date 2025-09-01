import dspy
from dspy.primitives.prediction import Prediction
from mkspy.metrics import ProgramGenerationMetric


def test_metric_simple() -> None:
    metric = ProgramGenerationMetric(test_cases=[(1, 2)])
    code = "\n".join([
        "import dspy",
        "predictor = dspy.Predict('x->y')",
        "def solve(x: int) -> int:",
        "    return x + 1",
    ])
    pred = Prediction(code=code)
    gold = dspy.Example().with_inputs()
    result = metric(gold, pred)
    assert result["score"] >= 0.5
