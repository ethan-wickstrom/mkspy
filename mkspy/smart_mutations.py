from __future__ import annotations

from abc import ABC, abstractmethod


class SmartMutation(ABC):
    """Base class for intelligent mutations guided by GEPA feedback."""

    @abstractmethod
    def should_apply(self, program: str, feedback: str) -> bool:
        """Determine if this mutation is relevant based on feedback."""
        raise NotImplementedError

    @abstractmethod
    def apply(self, program: str, feedback: str) -> str:
        """Apply the mutation to improve the program."""
        raise NotImplementedError


class AddErrorHandlingMutation(SmartMutation):
    """Add try-except blocks when execution errors are detected."""

    def should_apply(self, program: str, feedback: str) -> bool:
        return "failed:" in feedback.lower() or "error" in feedback.lower()

    def apply(self, program: str, feedback: str) -> str:
        wrapped: str = (
            f"try:\n{program}\nexcept Exception as e:\n"
            "    print(f'Exception occurred: {e}')\n    raise"
        )
        return wrapped


class RefactorToModularMutation(SmartMutation):
    """Break monolithic forward() into sub-methods."""

    def should_apply(self, program: str, feedback: str) -> bool:
        return "modularity" in feedback.lower() or "complex" in feedback.lower()

    def apply(self, program: str, feedback: str) -> str:
        comment: str = "# TODO: refactor into smaller functions"
        return f"{comment}\n{program}"


class UpgradePredictorMutation(SmartMutation):
    """Upgrade Predict to ChainOfThought or ReAct based on task complexity."""

    def should_apply(self, program: str, feedback: str) -> bool:
        return "reasoning" in feedback.lower() or "complex task" in feedback.lower()

    def apply(self, program: str, feedback: str) -> str:
        upgraded: str = program.replace("Predict", "ChainOfThought")
        return upgraded
