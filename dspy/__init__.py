from __future__ import annotations

from typing import Any
from dataclasses import dataclass

from .primitives.prediction import Prediction


class Module:
    """Minimal runtime stub for DSPy modules."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - simple
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError


class _Predict(Module):
    def __init__(self, signature: str) -> None:
        self.signature: str = signature

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return None


def Predict(signature: str) -> Module:
    """Return a stub predictive module identified by ``signature``."""

    return _Predict(signature)


@dataclass
class Example:
    """Stub example container."""

    fields: dict[str, Any]

    def __init__(self, **fields: Any) -> None:
        object.__setattr__(self, "fields", fields)

    def with_inputs(self, *fields: str) -> Example:
        return self


class GEPA(Module):
    """Stub optimizer with ``compile`` method."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def compile(self, program: Module, *, trainset: list[Example], valset: list[Example]) -> Module:
        return program


class Signature:
    """Stub signature base class."""


def InputField(desc: str = "") -> Any:
    return None


def OutputField(desc: str = "") -> Any:
    return None


class ChainOfThought(Module):
    """Stub chain-of-thought module."""

    def __init__(self, signature: type[Signature]) -> None:
        self.signature: type[Signature] = signature

    def forward(self, *args: Any, **kwargs: Any) -> Prediction:
        return Prediction()
