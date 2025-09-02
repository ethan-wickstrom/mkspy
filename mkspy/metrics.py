from __future__ import annotations

from typing import Any
from collections.abc import Callable
import ast
import dspy
from dspy import Prediction, Example
from dspy.teleprompt.gepa.gepa_utils import DSPyTrace, ScoreWithFeedback


class ProgramGenerationMetric:
    """Comprehensive metric for evaluating generated DSPy programs."""

    def __init__(self, test_cases: list[tuple[Any, Any]], syntax_weight: float = 0.3, execution_weight: float = 0.4, quality_weight: float = 0.3) -> None:
        self.test_cases: list[tuple[Any, Any]] = test_cases
        self.weights: dict[str, float] = {"syntax": syntax_weight, "execution": execution_weight, "quality": quality_weight}

    def __call__(
        self,
        gold: Example,
        pred: Prediction,
        trace: DSPyTrace | None = None,
        pred_name: str | None = None,
        pred_trace: DSPyTrace | None = None,
    ) -> float | ScoreWithFeedback:
        feedback_parts: list[str] = []
        scores: dict[str, float] = {"syntax": 0.0, "execution": 0.0, "quality": 0.0}

        try:
            ast.parse(pred.code)
            scores["syntax"] = 1.0
            feedback_parts.append("\u2713 Code is syntactically valid Python")
        except SyntaxError as e:
            scores["syntax"] = 0.0
            feedback_parts.append(f"\u2717 Syntax error at line {e.lineno}: {e.msg}")

        if scores["syntax"] > 0.0:
            structure_feedback: dict[str, Any] = self._validate_dspy_structure(pred.code)
            if structure_feedback["valid"]:
                scores["quality"] += 0.5
                feedback_parts.append("\u2713 Valid DSPy module structure detected")
                feedback_parts.append(f"  - Found {structure_feedback['num_predictors']} predictors")
                feedback_parts.append(f"  - Found {structure_feedback['num_methods']} methods")
            else:
                feedback_parts.append(f"\u2717 DSPy structure issues: {structure_feedback['errors']}")

        if scores["syntax"] > 0.0:
            exec_results: dict[str, Any] = self._test_execution(pred.code, self.test_cases)
            scores["execution"] = exec_results["pass_rate"]
            feedback_parts.append(f"\u2713 Passed {exec_results['passed']}/{exec_results['total']} test cases")
            for failure in exec_results["failures"]:
                feedback_parts.append(f"  \u2717 Test '{failure['test']}' failed: {failure['error']}")

        quality_feedback: dict[str, Any] = self._assess_code_quality(pred.code)
        scores["quality"] = (scores["quality"] + quality_feedback["score"]) / 2.0
        feedback_parts.extend(quality_feedback["feedback"])

        total_score: float = sum(scores[k] * self.weights[k] for k in scores)

        feedback: str = "\n".join([
            "=== Program Generation Evaluation ===",
            f"Overall Score: {total_score:.2f}/1.0",
            "",
            "Component Scores:",
            f"  Syntax: {scores['syntax']:.2f} (weight: {self.weights['syntax']})",
            f"  Execution: {scores['execution']:.2f} (weight: {self.weights['execution']})",
            f"  Quality: {scores['quality']:.2f} (weight: {self.weights['quality']})",
            "",
            "Detailed Feedback:",
            *feedback_parts,
            "",
            "Improvement Suggestions:",
            *self._generate_suggestions(scores, pred),
        ])

        return ScoreWithFeedback(score=total_score, feedback=feedback)

    def _validate_dspy_structure(self, code: str) -> dict[str, Any]:
        result: dict[str, Any] = {"valid": False, "num_predictors": 0, "num_methods": 0, "errors": ""}
        try:
            tree: ast.AST = ast.parse(code)
        except SyntaxError as e:
            result["errors"] = f"syntax error: {e.msg}"
            return result
        num_predictors: int = 0
        num_methods: int = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "Predict":
                num_predictors += 1
            if isinstance(node, ast.FunctionDef):
                num_methods += 1
        result["num_predictors"] = num_predictors
        result["num_methods"] = num_methods
        result["valid"] = num_predictors > 0
        if not result["valid"]:
            result["errors"] = "no dspy.Predict usage found"
        return result

    def _test_execution(self, code: str, test_cases: list[tuple[Any, Any]]) -> dict[str, Any]:
        namespace: dict[str, Any] = {"dspy": dspy}
        failures: list[dict[str, str]] = []
        passed: int = 0
        total: int = len(test_cases)
        try:
            tree: ast.AST = ast.parse(code)
            for node in list(tree.body):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    module_names: list[str] = [alias.name for alias in node.names]
                    if not all(name == "dspy" for name in module_names):
                        raise RuntimeError(f"disallowed import: {', '.join(module_names)}")
                    tree.body.remove(node)
            compiled = compile(tree, "<generated>", "exec")
            exec(compiled, {"__builtins__": {}}, namespace)
        except Exception as e:
            error_info: dict[str, str] = {"test": "__compile__", "error": str(e)}
            failures.append(error_info)
            return {"pass_rate": 0.0, "passed": 0, "total": total, "failures": failures}
        func: Callable[[Any], Any] | None = namespace.get("solve")
        if func is None:
            return {"pass_rate": 0.0, "passed": 0, "total": total, "failures": [{"test": "__entry__", "error": "solve function not found"}]}
        for given, expected in test_cases:
            try:
                output: Any = func(given)
                if output == expected:
                    passed += 1
                else:
                    failures.append({"test": str(given), "error": f"expected {expected!r}, got {output!r}"})
            except Exception as e:
                failures.append({"test": str(given), "error": str(e)})
        pass_rate: float = passed / total if total > 0 else 0.0
        return {"pass_rate": pass_rate, "passed": passed, "total": total, "failures": failures}

    def _assess_code_quality(self, code: str) -> dict[str, Any]:
        lines: list[str] = code.splitlines()
        line_count: int = len(lines)
        if line_count <= 50:
            score: float = 1.0
            feedback: list[str] = ["\u2713 Code length is concise"]
        else:
            score = 0.5
            feedback = ["\u2717 Code is lengthy"]
        return {"score": score, "feedback": feedback}

    def _generate_suggestions(self, scores: dict[str, float], pred: Prediction) -> list[str]:
        suggestions: list[str] = []
        if scores["syntax"] < 1.0:
            suggestions.append("Focus on generating syntactically correct Python code")
        if scores["execution"] < 0.5:
            suggestions.append("Ensure the solve() function handles the expected inputs")
            suggestions.append("Add error handling for edge cases")
        if scores["quality"] < 0.5:
            suggestions.append("Use more DSPy built-in modules (Predict, ChainOfThought, ReAct)")
            suggestions.append("Improve modularity by breaking complex logic into sub-modules")
        return suggestions
