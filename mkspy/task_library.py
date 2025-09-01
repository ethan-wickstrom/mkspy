from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Generic, List, Mapping, Optional, Tuple, Type, TypeVar


T = TypeVar("T")
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True)
class Scope:
    """Immutable variable bindings."""

    bindings: Mapping[str, object]

    def resolve(self, name: str) -> object:
        return self.bindings[name]

    def extend(self, name: str, value: object) -> "Scope":
        new_bindings: Dict[str, object] = dict(self.bindings)
        if name in new_bindings:
            raise KeyError(f"{name} already bound")
        new_bindings[name] = value
        return Scope(bindings=new_bindings)


EMPTY_SCOPE: Scope = Scope(bindings={})


class Op(Generic[InputT, OutputT]):
    """Abstract operation with explicit input and output types."""

    input_type: Type[InputT]
    output_type: Type[OutputT]

    def evaluate(self, data: InputT, scope: Scope) -> OutputT:
        raise NotImplementedError


@dataclass(frozen=True)
class Value(Op[None, T], Generic[T]):
    """Literal value primitive."""

    data: T
    input_type: Type[None] = type(None)
    output_type: Type[T]

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_type", type(self.data))

    def evaluate(self, data: None, scope: Scope) -> T:
        return self.data


@dataclass(frozen=True)
class Reference(Op[None, T], Generic[T]):
    """Reference to a scoped value."""

    name: str
    output_type: Type[T]
    input_type: Type[None] = type(None)

    def evaluate(self, data: None, scope: Scope) -> T:
        value: object = scope.resolve(self.name)
        if not isinstance(value, self.output_type):
            raise TypeError(f"reference {self.name} expected {self.output_type.__name__}")
        typed_value: T = value
        return typed_value


@dataclass(frozen=True)
class Transform(Op[InputT, OutputT], Generic[InputT, OutputT]):
    """Deterministic transformation."""

    func: Callable[[InputT], OutputT]
    input_type: Type[InputT]
    output_type: Type[OutputT]

    def evaluate(self, data: InputT, scope: Scope) -> OutputT:
        return self.func(data)


@dataclass(frozen=True)
class Constraint(Op[T, T], Generic[T]):
    """Value predicate."""

    predicate: Callable[[T], bool]
    input_type: Type[T]
    output_type: Type[T]

    def evaluate(self, data: T, scope: Scope) -> T:
        if not self.predicate(data):
            raise ValueError("constraint violated")
        return data


@dataclass(frozen=True)
class QualityConstraint(Op[T, T], Generic[T]):
    """Continuous quality function constraint."""

    quality: Callable[[T], float]
    threshold: float
    input_type: Type[T]
    output_type: Type[T]

    def evaluate(self, data: T, scope: Scope) -> T:
        score: float = self.quality(data)
        if score < self.threshold:
            raise ValueError("quality below threshold")
        return data


ReturnT = TypeVar("ReturnT")


@dataclass(frozen=True)
class Aggregate(Op[List[T], ReturnT], Generic[T, ReturnT]):
    """Aggregate a list of values."""

    func: Callable[[List[T]], ReturnT]
    output_type: Type[ReturnT]
    input_type: Type[List[T]] = list

    def evaluate(self, data: List[T], scope: Scope) -> ReturnT:
        return self.func(data)


@dataclass(frozen=True)
class Sequential(Op[InputT, OutputT], Generic[InputT, OutputT]):
    """Sequential composition."""

    ops: Tuple[Op[object, object], ...]
    input_type: Type[InputT]
    output_type: Type[OutputT]

    def __post_init__(self) -> None:
        prev: Type[object] = self.input_type
        for op in self.ops:
            op_typed: Op[object, object] = op
            if prev is not op_typed.input_type:
                raise TypeError("incompatible sequence")
            prev = op_typed.output_type
        if prev is not self.output_type:
            raise TypeError("sequence output mismatch")

    def evaluate(self, data: InputT, scope: Scope) -> OutputT:
        result: object = data
        for op in self.ops:
            op_typed: Op[object, object] = op
            result = op_typed.evaluate(result, scope)
        if not isinstance(result, self.output_type):
            raise TypeError("result type mismatch")
        final: OutputT = result
        return final


@dataclass(frozen=True)
class Parallel(Op[InputT, Tuple[object, ...]], Generic[InputT]):
    """Parallel composition without interaction."""

    ops: Tuple[Op[InputT, object], ...]
    input_type: Type[InputT]
    output_type: Type[Tuple[object, ...]] = tuple

    def evaluate(self, data: InputT, scope: Scope) -> Tuple[object, ...]:
        results: List[object] = []
        for op in self.ops:
            op_typed: Op[InputT, object] = op
            results.append(op_typed.evaluate(data, scope))
        return tuple(results)


@dataclass(frozen=True)
class Conditional(Op[InputT, OutputT], Generic[InputT, OutputT]):
    """Conditional composition."""

    predicate: Op[InputT, bool]
    if_true: Op[InputT, OutputT]
    if_false: Op[InputT, OutputT]
    input_type: Type[InputT]
    output_type: Type[OutputT]

    def __post_init__(self) -> None:
        if self.predicate.output_type is not bool:
            raise TypeError("predicate must return bool")
        if self.if_true.input_type is not self.input_type:
            raise TypeError("true branch input mismatch")
        if self.if_false.input_type is not self.input_type:
            raise TypeError("false branch input mismatch")
        if self.if_true.output_type is not self.output_type:
            raise TypeError("true branch output mismatch")
        if self.if_false.output_type is not self.output_type:
            raise TypeError("false branch output mismatch")

    def evaluate(self, data: InputT, scope: Scope) -> OutputT:
        condition: bool = self.predicate.evaluate(data, scope)
        branch: Op[InputT, OutputT]
        if condition:
            branch = self.if_true
        else:
            branch = self.if_false
        return branch.evaluate(data, scope)


@dataclass(frozen=True)
class Iterate(Op[T, T], Generic[T]):
    """Iterative composition until condition fails."""

    body: Op[T, T]
    condition: Op[T, bool]
    input_type: Type[T]
    output_type: Type[T]

    def __post_init__(self) -> None:
        if self.body.input_type is not self.input_type or self.body.output_type is not self.output_type:
            raise TypeError("body type mismatch")
        if self.condition.input_type is not self.input_type or self.condition.output_type is not bool:
            raise TypeError("condition type mismatch")

    def evaluate(self, data: T, scope: Scope) -> T:
        current: T = data
        while self.condition.evaluate(current, scope):
            current = self.body.evaluate(current, scope)
        return current


class Polarity(Enum):
    """Discrete polarity values."""

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class TaskSpec:
    """Task definition tying description to an operation."""

    description: str
    operation: Op[None, object]
    test_cases: List[Tuple[object, object]]
    expected_signature: Optional[str] = None


Example = Tuple[object, object]


def _parse_value(text: str) -> object:
    try:
        parsed: object = ast.literal_eval(text)
        return parsed
    except Exception:
        return text


def _case(raw: str) -> Example:
    left_str, right_str = raw.split("->", 1)
    left_val: object = _parse_value(left_str.strip())
    right_val: object = _parse_value(right_str.strip())
    return left_val, right_val


def task(description: str, cases: List[str], expected_signature: Optional[str] = None) -> Dict[str, object]:
    parsed: List[Example] = [_case(c) for c in cases]
    return {"description": description, "expected_signature": expected_signature, "test_cases": parsed}


TASK_LIBRARY: List[Dict[str, object]] = []

__all__: List[str] = [
    "Scope",
    "EMPTY_SCOPE",
    "Op",
    "Value",
    "Reference",
    "Transform",
    "Constraint",
    "QualityConstraint",
    "Aggregate",
    "Sequential",
    "Parallel",
    "Conditional",
    "Iterate",
    "Polarity",
    "TaskSpec",
    "task",
    "TASK_LIBRARY",
]
