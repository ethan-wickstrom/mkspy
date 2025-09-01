from __future__ import annotations

from typing import Any


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
