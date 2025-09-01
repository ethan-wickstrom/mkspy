import dspy
from dspy import Signature, InputField, OutputField


class ProgramSpec(Signature):
    """Generate specifications for a DSPy program."""
    task_description: str = InputField(desc="What the program should accomplish")

    signature_spec: str = OutputField(desc="Signature definition in DSPy format")
    module_architecture: str = OutputField(desc="Module composition and flow")
    implementation_code: str = OutputField(desc="Complete DSPy module implementation")


class DSPyProgramGenerator(dspy.Module):
    """Meta-module that generates complete DSPy programs."""

    def __init__(self) -> None:
        self.spec_generator: dspy.Module = dspy.ChainOfThought(ProgramSpec)
        self.code_refiner: dspy.Module = dspy.Predict("code, requirements -> refined_code")
        self.validator: dspy.Module = dspy.Predict("code -> validation_result, errors")

    def forward(self, task_description: str) -> dspy.Prediction:
        spec: dspy.Prediction = self.spec_generator(task_description=task_description)

        refined: dspy.Prediction = self.code_refiner(
            code=spec.implementation_code,
            requirements=task_description,
        )

        validation: dspy.Prediction = self.validator(code=refined.refined_code)

        return dspy.Prediction(
            code=refined.refined_code,
            signature=spec.signature_spec,
            architecture=spec.module_architecture,
            is_valid=validation.validation_result,
            errors=validation.errors,
        )
