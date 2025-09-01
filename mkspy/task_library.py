from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import singledispatch
import ast
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Tuple,
    TypeVar,
    Protocol,
)


T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


class Expr(Protocol[T]):
    """Syntax node protocol."""
    pass


@dataclass(frozen=True, slots=True)
class Value(Generic[T]):
    data: T


@dataclass(frozen=True, slots=True)
class Reference(Generic[T]):
    name: str


@dataclass(frozen=True, slots=True)
class Transform(Generic[T, U]):
    func: Callable[[T], U]


@dataclass(frozen=True, slots=True)
class Aggregation(Generic[T, U]):
    func: Callable[[Iterable[T]], U]


@dataclass(frozen=True, slots=True)
class Constraint(Generic[T]):
    predicate: Callable[[T], bool]


@dataclass(frozen=True, slots=True)
class Quality(Generic[T]):
    measure: Callable[[T], float]


@dataclass(slots=True)
class Scope:
    bindings: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class Sequential(Generic[T, U]):
    source: Expr[T]
    transforms: List[Transform[Any, Any]]


@dataclass(frozen=True, slots=True)
class Parallel(Generic[T]):
    branches: List[Expr[T]]


@dataclass(frozen=True, slots=True)
class Conditional(Generic[T]):
    subject: Expr[Any]
    constraint: Constraint[Any]
    on_true: Expr[T]
    on_false: Expr[T]


@dataclass(frozen=True, slots=True)
class Iterate(Generic[T]):
    initial: Expr[T]
    transform: Transform[T, T]
    condition: Constraint[T]


@dataclass(frozen=True, slots=True)
class Aggregate(Generic[T, U]):
    aggregator: Aggregation[T, U]
    inputs: List[Expr[T]]


@dataclass(frozen=True, slots=True)
class Measure(Generic[T]):
    quality: Quality[T]
    source: Expr[T]


@singledispatch
def evaluate(expr: object, scope: Scope) -> Any:
    raise TypeError(f"Unknown expression type: {type(expr)!r}")


@evaluate.register
def _value(expr: Value[T], scope: Scope) -> T:
    return expr.data


@evaluate.register
def _reference(expr: Reference[T], scope: Scope) -> T:
    return scope.bindings[expr.name]


@evaluate.register
def _sequential(expr: Sequential[Any, Any], scope: Scope) -> Any:
    value: Any = evaluate(expr.source, scope)
    for transform in expr.transforms:
        value = transform.func(value)
    return value


@evaluate.register
def _parallel(expr: Parallel[Any], scope: Scope) -> Tuple[Any, ...]:
    results: List[Any] = [evaluate(branch, scope) for branch in expr.branches]
    return tuple(results)


@evaluate.register
def _conditional(expr: Conditional[T], scope: Scope) -> T:
    candidate: Any = evaluate(expr.subject, scope)
    if expr.constraint.predicate(candidate):
        return evaluate(expr.on_true, scope)
    return evaluate(expr.on_false, scope)


@evaluate.register
def _iterate(expr: Iterate[T], scope: Scope) -> T:
    value: T = evaluate(expr.initial, scope)
    while not expr.condition.predicate(value):
        value = expr.transform.func(value)
    return value


@evaluate.register
def _aggregate(expr: Aggregate[Any, Any], scope: Scope) -> Any:
    items: List[Any] = [evaluate(item, scope) for item in expr.inputs]
    return expr.aggregator.func(items)


@evaluate.register
def _measure(expr: Measure[T], scope: Scope) -> float:
    subject: T = evaluate(expr.source, scope)
    return expr.quality.measure(subject)


class Polarity(Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


Example = Tuple[Any, Any]


def _parse_value(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def _case(raw: str) -> Example:
    left, right = raw.split("->", 1)
    return _parse_value(left.strip()), _parse_value(right.strip())


def task(description: str, cases: List[str]) -> Dict[str, Any]:
    parsed: List[Example] = [_case(c) for c in cases]
    return {"description": description, "test_cases": parsed}


TASK_LIBRARY: List[Dict[str, Any]] = [task("Increment number", ["1 -> 2"])]


__all__: List[str] = [
    "Value",
    "Reference",
    "Transform",
    "Aggregation",
    "Constraint",
    "Quality",
    "Scope",
    "Sequential",
    "Parallel",
    "Conditional",
    "Iterate",
    "Aggregate",
    "Measure",
    "Polarity",
    "evaluate",
    "task",
    "TASK_LIBRARY",
]

