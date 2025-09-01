from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import dspy
from dspy import GEPA

from .meta_module import DSPyProgramGenerator
from .metrics import ProgramGenerationMetric
from .task_library import TaskSpec


class GEPAEvolver:
    """Evolve DSPy program generators using GEPA optimization.

    Responsibilities:
      - Build training/validation datasets from a task library (TaskSpec list).
      - Wire the evaluation metric and run GEPA.
      - Save artifacts (optional).
    """

    def __init__(self, task_library: list[TaskSpec], output_dir: Path) -> None:
        self.task_library: list[TaskSpec] = task_library
        self.output_dir: Path = output_dir
        self.output_dir.mkdir(exist_ok=True)

        # Program generator is a pure module; orchestration lives here.
        self.generator: DSPyProgramGenerator = DSPyProgramGenerator()

        # Basic split: first 80 to train, rest to val, matching previous semantics.
        self.trainset: list[dspy.Example] = self._create_dataset(task_library[:80])
        self.valset: list[dspy.Example] = self._create_dataset(task_library[80:])

        # Aggregate all test cases from the task library for the metric.
        self.metric: Any = ProgramGenerationMetric(test_cases=self._extract_test_cases())

    def evolve(self) -> dspy.Module:
        optimizer: GEPA = GEPA(
            metric=self.metric,
            track_stats=True,
            track_best_outputs=True,
            log_dir=str(self.output_dir / "gepa_logs"),
        )
        optimized_generator = optimizer.compile(
            self.generator, trainset=self.trainset, valset=self.valset
        )
        self._save_results(optimized_generator)
        return optimized_generator

    def _create_dataset(self, tasks: list[TaskSpec]) -> list[dspy.Example]:
        dataset: list[dspy.Example] = []
        for task in tasks:
            ex = dspy.Example(
                task_description=task.description,
                expected_signature=task.expected_signature,
            ).with_inputs("task_description")
            dataset.append(ex)
        return dataset

    def _extract_test_cases(self) -> list[tuple[Any, Any]]:
        cases: list[tuple[Any, Any]] = []
        for task in self.task_library:
            cases.extend(task.test_cases)
        return cases

    def _save_results(self, optimized_generator: dspy.Module) -> None:
        # Rely on dspy.Module.save for persistence.
        optimized_generator.save(self.output_dir / "optimized_generator", save_program=True)

        examples_dir: Path = self.output_dir / "generated_programs"
        examples_dir.mkdir(exist_ok=True)

        # Emit a few examples for manual inspection.
        for i, task in enumerate(self.task_library[:10]):
            result: dspy.Prediction = optimized_generator(task_description=task.description)
            code_file: Path = examples_dir / f"program_{i}.py"
            code_file.write_text(result.code, encoding="utf-8")

            meta = {
                "task": task.description,
                "signature": result.signature,
                "architecture": result.architecture,
                "is_valid": result.is_valid,
                "errors": result.errors,
            }
            meta_file: Path = examples_dir / f"program_{i}_meta.json"
            meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
