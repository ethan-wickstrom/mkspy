from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypePrimitive:
    """Atomic type descriptor contributing only data shape."""

    name: str


Text: TypePrimitive = TypePrimitive("Text")
Number: TypePrimitive = TypePrimitive("Number")
DictType: TypePrimitive = TypePrimitive("Dict")


@dataclass(frozen=True)
class ListType(TypePrimitive):
    element: TypePrimitive
    name: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", f"List[{self.element.name}]")


@dataclass(frozen=True)
class Literal(TypePrimitive):
    values: Tuple[str, ...]
    name: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", f"Literal{self.values}")


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


class Composable(Protocol):
    @property
    def input_type(self) -> TypePrimitive: ...

    @property
    def output_type(self) -> TypePrimitive: ...


@dataclass(frozen=True)
class Operation:
    input_type: TypePrimitive
    output_type: TypePrimitive


@dataclass(frozen=True)
class Map(Operation):
    fn: Operation
    input_type: TypePrimitive = field(init=False)
    output_type: TypePrimitive = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_type", ListType(self.fn.input_type))
        object.__setattr__(self, "output_type", ListType(self.fn.output_type))


@dataclass(frozen=True)
class Filter(Operation):
    predicate: Operation
    input_type: TypePrimitive = field(init=False)
    output_type: TypePrimitive = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_type", ListType(self.predicate.input_type))
        object.__setattr__(self, "output_type", ListType(self.predicate.input_type))


@dataclass(frozen=True)
class Reduce(Operation):
    fn: Operation
    input_type: TypePrimitive = field(init=False)
    output_type: TypePrimitive = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_type", ListType(self.fn.input_type))
        object.__setattr__(self, "output_type", self.fn.output_type)


# ---------------------------------------------------------------------------
# Composition operators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sequential(Operation):
    first: Composable
    second: Composable
    input_type: TypePrimitive = field(init=False)
    output_type: TypePrimitive = field(init=False)

    def __post_init__(self) -> None:
        if self.first.output_type != self.second.input_type:
            raise TypeError("Type mismatch in Sequential")
        object.__setattr__(self, "input_type", self.first.input_type)
        object.__setattr__(self, "output_type", self.second.output_type)


@dataclass(frozen=True)
class Parallel(Operation):
    left: Composable
    right: Composable
    input_type: TypePrimitive = field(init=False)
    output_type: TypePrimitive = field(init=False)

    def __post_init__(self) -> None:
        if self.left.input_type != self.right.input_type:
            raise TypeError("Type mismatch in Parallel")
        tuple_name: str = (
            f"Tuple[{self.left.output_type.name}, {self.right.output_type.name}]"
        )
        object.__setattr__(self, "input_type", self.left.input_type)
        object.__setattr__(self, "output_type", TypePrimitive(tuple_name))


@dataclass(frozen=True)
class Conditional(Operation):
    condition: str
    if_true: Composable
    if_false: Composable
    input_type: TypePrimitive = field(init=False)
    output_type: TypePrimitive = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.if_true.input_type != self.if_false.input_type
            or self.if_true.output_type != self.if_false.output_type
        ):
            raise TypeError("Branch type mismatch")
        object.__setattr__(self, "input_type", self.if_true.input_type)
        object.__setattr__(self, "output_type", self.if_true.output_type)


@dataclass(frozen=True)
class Iterative(Operation):
    body: Composable
    input_type: TypePrimitive = field(init=False)
    output_type: TypePrimitive = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_type", self.body.input_type)
        object.__setattr__(self, "output_type", self.body.output_type)


# ---------------------------------------------------------------------------
# Non-discrete specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreeForm:
    shape: str
    must_include: Tuple[str, ...] = ()


@dataclass(frozen=True)
class NaturalSpec:
    domain: str
    goal: str
    constraints: Tuple[str, ...] = ()
    output_format: Optional[FreeForm] = None


Process = Union[Composable, NaturalSpec]


# ---------------------------------------------------------------------------
# Task specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec:
    description: str
    input: Optional[TypePrimitive]
    process: Optional[Process]
    output: Optional[TypePrimitive]
    test_cases: Sequence[Tuple[Any, Any]] = ()
    expected_signature: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers for simple task construction
# ---------------------------------------------------------------------------


Example = Tuple[Any, Any]


def _parse_value(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def _case(raw: str) -> Example:
    left, right = raw.split("->", 1)
    return _parse_value(left.strip()), _parse_value(right.strip())


def task(
    description: str,
    cases: Sequence[str],
    *,
    input_type: Optional[TypePrimitive] = None,
    output_type: Optional[TypePrimitive] = None,
    process: Optional[Process] = None,
) -> TaskSpec:
    parsed: List[Example] = [_case(c) for c in cases]
    return TaskSpec(
        description=description,
        input=input_type,
        process=process,
        output=output_type,
        test_cases=parsed,
    )


# ---------------------------------------------------------------------------
# Example task library
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Classify(Operation):
    labels: Tuple[str, ...]
    input_type: TypePrimitive = field(init=False)
    output_type: TypePrimitive = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_type", Text)
        object.__setattr__(self, "output_type", Literal(self.labels))


TASK_LIBRARY: List[TaskSpec] = [
    task(
        "Classify sentiment in product reviews",
        [
            "['The phone is fantastic and works great.', 'Battery died after two days.'] -> ['positive', 'negative']",
        ],
        input_type=ListType(element=Text),
        output_type=ListType(element=Literal(("positive", "negative"))),
        process=Map(fn=Classify(labels=("positive", "negative"))),
    ),
    task(
        "Determine relationships between entities",
        [],
        input_type=Text,
        process=NaturalSpec(
            domain="knowledge_graph",
            goal="determine relationships",
            constraints=("use commonsense reasoning",),
            output_format=FreeForm(
                shape="triples",
                must_include=("entity1", "relationship", "entity2"),
            ),
        ),
        output_type=ListType(element=TypePrimitive("Relationship")),
    ),
]


__all__: List[str] = [
    "TASK_LIBRARY",
    "task",
    "TaskSpec",
    "TypePrimitive",
    "ListType",
    "Literal",
    "Map",
    "Filter",
    "Reduce",
    "Sequential",
    "Parallel",
    "Conditional",
    "Iterative",
    "FreeForm",
    "NaturalSpec",
]

