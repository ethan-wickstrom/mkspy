from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
import dspy
from dspy import GEPA

from meta_module import DSPyProgramGenerator
from metrics import ProgramGenerationMetric


class GEPAEvolver:
    """Evolve DSPy program generators using GEPA optimization."""

    def __init__(self, task_library: List[Dict[str, Any]], output_dir: Path) -> None:
        self.task_library: List[Dict[str, Any]] = task_library
        self.output_dir: Path = output_dir
        self.output_dir.mkdir(exist_ok=True)

        self.generator: DSPyProgramGenerator = DSPyProgramGenerator()
        self.trainset: List[dspy.Example] = self._create_dataset(task_library[:80])
        self.valset: List[dspy.Example] = self._create_dataset(task_library[80:])
        self.metric: ProgramGenerationMetric = ProgramGenerationMetric(test_cases=self._extract_test_cases())

    def evolve(self, num_iterations: int = 50, rollout_size: int = 4) -> DSPyProgramGenerator:
        optimizer: GEPA = GEPA(
            metric=self.metric,
            track_stats=True,
            track_best_outputs=True,
            log_dir=str(self.output_dir / "gepa_logs"),
        )

        optimized_generator: DSPyProgramGenerator = optimizer.compile(
            self.generator, trainset=self.trainset, valset=self.valset
        )
        self._save_results(optimized_generator)
        return optimized_generator

    def _create_dataset(self, tasks: List[Dict[str, Any]]) -> List[dspy.Example]:
        dataset: List[dspy.Example] = []
        for task in tasks:
            example: dspy.Example = dspy.Example(
                task_description=task["description"],
                input_types=task.get("input_types", "str"),
                output_requirements=task.get("output_requirements", "str"),
                expected_signature=task.get("expected_signature"),
            ).with_inputs("task_description", "input_types", "output_requirements")
            dataset.append(example)
        return dataset

    def _extract_test_cases(self) -> List[Dict[str, Any]]:
        cases: List[Dict[str, Any]] = []
        for task in self.task_library:
            for case in task.get("test_cases", []):
                cases.append(case)
        return cases

    def _save_results(self, optimized_generator: DSPyProgramGenerator) -> None:
        optimized_generator.save(self.output_dir / "optimized_generator", save_program=True)
        examples_dir: Path = self.output_dir / "generated_programs"
        examples_dir.mkdir(exist_ok=True)
        for i, task in enumerate(self.task_library[:10]):
            result: dspy.Prediction = optimized_generator(task_description=task["description"])
            code_file: Path = examples_dir / f"program_{i}.py"
            code_file.write_text(result.code, encoding="utf-8")
            meta: Dict[str, Any] = {
                "task": task["description"],
                "signature": result.signature,
                "architecture": result.architecture,
                "is_valid": result.is_valid,
                "errors": result.errors,
            }
            meta_file: Path = examples_dir / f"program_{i}_meta.json"
            meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
