from __future__ import annotations
from typing import Literal

import dspy
from dspy import Signature, InputField, OutputField, Code

from .author import generate_structured_code
from .cst_utils import validate_program, inspect_dspy_program


class ProgramSpec(Signature):
    """Generate specifications and code for a DSPy program."""
    task_description: str = InputField(desc="What the program should accomplish")

    # We favor typed outputs; code is a dspy.Code subtype (python).
    signature: str = OutputField(desc="Signature definition in DSPy style, if known")
    architecture: str = OutputField(desc="Module composition/flow summary")
    code: Code[Literal["python"]] = OutputField(desc="Complete DSPy module implementation")
    is_valid: bool = OutputField(desc="Validation result from static analysis")
    errors: list[str] = OutputField(desc="Validation errors, if any")


class DSPyProgramGenerator(dspy.Module):
    """Meta-module that synthesizes minimal DSPy programs and validates them with LibCST.

    - No persistence or GEPA logic here (separation of concerns).
    - Deterministic baseline synthesis (no LM required for generation).
    - Static validation uses shared LibCST analyzer.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, task_description: str) -> dspy.Prediction:
        # 1) Synthesize a minimal but runnable DSPy program.
        src: str = generate_structured_code(task_description)

        # 2) Validate with LibCST and compose summary fields.
        ok, errs = validate_program(src)
        report = inspect_dspy_program(src)

        # Signature summary: best-effort from the synthesized baseline.
        # The baseline always creates `class UserTask(dspy.Signature)`, with text->result.
        sig_summary = "UserTask: text: str -> result: str"

        # Architecture summary: minimal counts of predictor constructors and class names.
        ctor_counts = ", ".join(f"{k}={v}" for k, v in report.ctor_counts.items())
        arch_summary = (
            f"signatures={sorted(report.signature_classes)}, "
            f"modules={sorted(report.module_classes)}, "
            f"predictors=({ctor_counts})"
        )

        return dspy.Prediction(
            code=src,
            signature=sig_summary,
            architecture=arch_summary,
            is_valid=ok,
            errors=errs,
        )
