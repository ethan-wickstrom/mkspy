# mkspy

Meta-optimization framework for generating DSPy programs. This project provides:

- `DSPyProgramGenerator`: a meta-module that produces complete DSPy programs from task descriptions.
- `ProgramGenerationMetric`: rich metric with feedback for GEPA.
- `GEPAEvolver`: optimizer using GEPA as the sole evolution strategy.
- Smart mutation strategies responding to feedback.
- A small task library for experiments.
- An orthogonal task DSL enabling local compositional specifications.

## DSL Usage Example

```python
from mkspy.task_library import Classify, Sequential, Literal, Text

first: Classify = Classify(labels=("spam", "ham"))
label_type: Literal = Literal(("spam", "ham"))
second: Classify = Classify(
    labels=("important", "unimportant"), input_type_override=label_type
)
pipeline: Sequential = Sequential(first=first, second=second)

assert pipeline.input_type is Text
assert pipeline.output_type == Literal(("important", "unimportant"))
```
