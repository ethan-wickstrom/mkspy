from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass, field
from threading import Lock
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    cast,
    runtime_checkable,
)

import dspy


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


_TYPE_REGISTRY: dict[str, "TypePrimitive"] = {}
_TYPE_LOCK: Lock = Lock()


def _register_type(tp: "TypePrimitive") -> None:
    """Store ``tp`` in the registry, detecting name collisions."""

    with _TYPE_LOCK:
        existing: Optional[TypePrimitive] = _TYPE_REGISTRY.get(tp.name)
        if existing is not None:
            if existing is not tp:
                raise ValueError(
                    f"Type '{tp.name}' already registered with different instance"
                )
        else:
            _TYPE_REGISTRY[tp.name] = tp


def get_type(name: str) -> "TypePrimitive":
    """Retrieve a previously registered type primitive by name."""

    return _TYPE_REGISTRY[name]


@dataclass(frozen=True)
class TypePrimitive:
    """Atomic type descriptor contributing only data shape."""

    name: str

    def __post_init__(self) -> None:
        _register_type(self)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TypePrimitive) and self.name == other.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


Text: TypePrimitive = TypePrimitive("Text")
Number: TypePrimitive = TypePrimitive("Number")
DictType: TypePrimitive = TypePrimitive("Dict")


@dataclass(frozen=True)
class ListType(TypePrimitive):
    """Type representing a list of elements of a given type."""

    element: TypePrimitive
    name: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", f"List[{self.element.name}]")
        super().__post_init__()


@dataclass(frozen=True)
class Literal(TypePrimitive):
    """Type constrained to one of a finite set of string values."""

    values: Tuple[str, ...]
    name: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", f"Literal{self.values}")
        super().__post_init__()


@dataclass(frozen=True)
class TupleType(TypePrimitive):
    """Type representing a fixed-length tuple of heterogeneous elements."""

    elements: Tuple[TypePrimitive, ...]
    name: str = field(init=False)

    def __post_init__(self) -> None:
        names: str = ", ".join(element.name for element in self.elements)
        object.__setattr__(self, "name", f"Tuple[{names}]")
        super().__post_init__()


@dataclass(frozen=True)
class UnionType(TypePrimitive):
    """Type representing a value that may be one of several types."""

    options: Tuple[TypePrimitive, ...]
    name: str = field(init=False)

    def __post_init__(self) -> None:
        names: str = " | ".join(option.name for option in self.options)
        object.__setattr__(self, "name", f"Union[{names}]")
        super().__post_init__()


def literal_type(values: Tuple[str, ...]) -> TypePrimitive:
    """Return canonical :class:`Literal` for ``values``."""

    name: str = f"Literal{values}"
    try:
        return get_type(name)
    except KeyError:
        return Literal(values)


def list_type(element: TypePrimitive) -> ListType:
    """Return canonical :class:`ListType` for ``element``."""

    name: str = f"List[{element.name}]"
    try:
        return cast(ListType, get_type(name))
    except KeyError:
        return ListType(element)


def tuple_type(elements: Tuple[TypePrimitive, ...]) -> TupleType:
    """Return canonical :class:`TupleType` for ``elements``."""

    name: str = "Tuple[" + ", ".join(e.name for e in elements) + "]"
    try:
        return cast(TupleType, get_type(name))
    except KeyError:
        return TupleType(elements)


def union_type(options: Tuple[TypePrimitive, ...]) -> UnionType:
    """Return canonical :class:`UnionType` for ``options``."""

    name: str = "Union[" + " | ".join(o.name for o in options) + "]"
    try:
        return cast(UnionType, get_type(name))
    except KeyError:
        return UnionType(options)


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


@runtime_checkable
class ToDSPyModule(Protocol):
    """Protocol for objects that can produce a DSPy module."""

    def to_module(self) -> dspy.Module: ...


@runtime_checkable
class Composable(ToDSPyModule, Protocol):
    """Protocol for operations with typed inputs and outputs."""

    @property
    def input_type(self) -> TypePrimitive: ...

    @property
    def output_type(self) -> TypePrimitive: ...


def _ensure_composable(value: Any) -> Composable:
    """Validate that ``value`` conforms to the :class:`Composable` protocol."""

    if not isinstance(value, Composable):
        raise TypeError(
            f"Expected Composable with to_module(), got {type(value).__name__}. "
            "Ensure the object implements input_type and output_type attributes."
        )
    return value


@dataclass(frozen=True)
class Operation(ToDSPyModule):
    """Base class for typed operations."""

    input_type: TypePrimitive = field(init=False)
    output_type: TypePrimitive = field(init=False)
    _module: Optional[dspy.Module] = field(default=None, init=False, repr=False)

    def _build_module(self) -> dspy.Module:  # pragma: no cover - abstract
        """Construct a DSPy module implementing the operation."""
        raise NotImplementedError

    def to_module(self) -> dspy.Module:
        """Return a cached DSPy module, constructing lazily."""
        module: Optional[dspy.Module] = self._module
        if module is None:
            module = self._build_module()
            object.__setattr__(self, "_module", module)
        return module

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.to_module()(*args, **kwargs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(input_type={self.input_type.name}, "
            f"output_type={self.output_type.name})"
        )


@dataclass(frozen=True)
class Map(Operation):
    """Apply a function to each element of a list.

    Examples
    --------
    >>> classifier = Classify(labels=("spam", "ham"))
    >>> mapper = Map(fn=classifier)
    >>> mapper.to_module()
    """

    fn: Composable

    def __post_init__(self) -> None:
        fn_checked: Composable = _ensure_composable(self.fn)
        object.__setattr__(self, "fn", fn_checked)
        object.__setattr__(self, "input_type", list_type(fn_checked.input_type))
        object.__setattr__(self, "output_type", list_type(fn_checked.output_type))

    def _build_module(self) -> dspy.Module:
        """Return a DSPy module that maps over list items."""
        fn_module: dspy.Module = self.fn.to_module()

        class _Map(dspy.Module):
            def forward(self, items: List[Any]) -> List[Any]:
                result: List[Any] = []
                for item in items:
                    mapped: Any = fn_module(item)
                    result.append(mapped)
                return result

        return _Map()


@dataclass(frozen=True)
class Filter(Operation):
    """Keep elements that satisfy a predicate."""

    predicate: Composable

    def __post_init__(self) -> None:
        pred_checked: Composable = _ensure_composable(self.predicate)
        object.__setattr__(self, "predicate", pred_checked)
        object.__setattr__(self, "input_type", list_type(pred_checked.input_type))
        object.__setattr__(self, "output_type", list_type(pred_checked.input_type))

    def _build_module(self) -> dspy.Module:
        """Return a DSPy module that filters list items."""
        pred_module: dspy.Module = self.predicate.to_module()

        class _Filter(dspy.Module):
            def forward(self, items: List[Any]) -> List[Any]:
                result: List[Any] = []
                for item in items:
                    condition: Any = pred_module(item)
                    if condition:
                        result.append(item)
                return result

        return _Filter()


@dataclass(frozen=True)
class Reduce(Operation):
    """Aggregate list elements using a binary function.

    Parameters
    ----------
    fn
        Binary operation lifted as a composable.
    initial
        Optional initial value supplied when reducing an empty list.

    Raises
    ------
    ValueError
        If called with an empty input list and no ``initial`` value.
    """

    fn: Composable
    initial: Optional[Any] = None

    def __post_init__(self) -> None:
        fn_checked: Composable = _ensure_composable(self.fn)
        object.__setattr__(self, "fn", fn_checked)
        # If the function advertises a binary input as a TupleType, derive
        # reduce's list/item and accumulator types from it. Otherwise, fall back
        # to treating the function's input_type as the element type.
        if isinstance(fn_checked.input_type, TupleType) and len(fn_checked.input_type.elements) == 2:
            acc_tp, item_tp = fn_checked.input_type.elements
            if fn_checked.output_type is not acc_tp:
                raise TypeError(
                    "Reduce: fn output type must match accumulator type; "
                    f"got input={fn_checked.input_type.name} output={fn_checked.output_type.name}"
                )
            object.__setattr__(self, "input_type", list_type(item_tp))
            object.__setattr__(self, "output_type", acc_tp)
        else:
            object.__setattr__(self, "input_type", list_type(fn_checked.input_type))
            object.__setattr__(self, "output_type", fn_checked.output_type)

    def _build_module(self) -> dspy.Module:
        """Return a DSPy module that reduces a list."""
        fn_module: dspy.Module = self.fn.to_module()
        initial: Optional[Any] = self.initial

        class _Reduce(dspy.Module):
            def forward(self, items: List[Any]) -> Any:
                iterator: List[Any] = list(items)
                if not iterator:
                    if initial is None:
                        raise ValueError("Reduce: empty input")
                    return initial
                acc: Any = iterator[0] if initial is None else initial
                rest: List[Any] = iterator[1:] if initial is None else iterator
                # Enforce that the reducer is binary and accepts two positional args.
                sig = inspect.signature(fn_module.forward)
                params = list(sig.parameters.values())
                has_var_positional: bool = any(
                    p.kind is inspect.Parameter.VAR_POSITIONAL for p in params
                )
                positional_params = [
                    p
                    for p in params
                    if p.kind
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                ]
                if not has_var_positional and len(positional_params) < 2:
                    raise TypeError(
                        "Reduce: fn must be a binary operation that accepts two positional arguments; "
                        f"got forward{sig}"
                    )
                for item in rest:
                    acc = fn_module(acc, item)
                return acc

        return _Reduce()


# ---------------------------------------------------------------------------
# Composition operators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sequential(Operation):
    """Feed the output of one operation into another.

    Examples
    --------
    >>> first = Classify(labels=("spam", "ham"))
    >>> second = Classify(labels=("important", "unimportant"), input_type_override=Literal(("spam", "ham")))
    >>> Sequential(first=first, second=second)
    """

    first: Composable
    second: Composable

    def __post_init__(self) -> None:
        first_checked: Composable = _ensure_composable(self.first)
        second_checked: Composable = _ensure_composable(self.second)
        if first_checked.output_type is not second_checked.input_type:
            raise TypeError(
                "Sequential composition failed: "
                f"{type(first_checked).__name__} output ({first_checked.output_type.name}) "
                f"incompatible with {type(second_checked).__name__} input ({second_checked.input_type.name})"
            )
        object.__setattr__(self, "first", first_checked)
        object.__setattr__(self, "second", second_checked)
        object.__setattr__(self, "input_type", first_checked.input_type)
        object.__setattr__(self, "output_type", second_checked.output_type)

    def _build_module(self) -> dspy.Module:
        """Return a DSPy module composing stages sequentially with flattening."""
        ops: List[Composable] = []

        def collect(op: Composable) -> None:
            if isinstance(op, Sequential):
                collect(op.first)
                collect(op.second)
            else:
                ops.append(op)

        collect(self.first)
        collect(self.second)
        modules: List[dspy.Module] = [op.to_module() for op in ops]

        class _Sequential(dspy.Module):
            def forward(self, value: Any) -> Any:
                result: Any = value
                for mod in modules:
                    result = mod(result)
                return result

        return _Sequential()

    def __repr__(self) -> str:
        return (
            f"Sequential(first={type(self.first).__name__}, "
            f"second={type(self.second).__name__}, "
            f"flow={self.input_type.name}->{self.output_type.name})"
        )


@dataclass(frozen=True)
class Parallel(Operation):
    """Run two operations on the same input in parallel."""

    left: Composable
    right: Composable

    def __post_init__(self) -> None:
        left_checked: Composable = _ensure_composable(self.left)
        right_checked: Composable = _ensure_composable(self.right)
        if left_checked.input_type is not right_checked.input_type:
            raise TypeError(
                "Parallel composition failed: "
                f"left {type(left_checked).__name__} input ({left_checked.input_type.name}) "
                f"does not match right {type(right_checked).__name__} input ({right_checked.input_type.name})"
            )
        output_tp: TupleType = tuple_type(
            (left_checked.output_type, right_checked.output_type)
        )
        object.__setattr__(self, "left", left_checked)
        object.__setattr__(self, "right", right_checked)
        object.__setattr__(self, "input_type", left_checked.input_type)
        object.__setattr__(self, "output_type", output_tp)

    def _build_module(self) -> dspy.Module:
        """Return a DSPy module executing branches in parallel."""
        left_mod: dspy.Module = self.left.to_module()
        right_mod: dspy.Module = self.right.to_module()

        class _Parallel(dspy.Module):
            def forward(self, value: Any) -> Tuple[Any, Any]:
                left_result: Any = left_mod(value)
                right_result: Any = right_mod(value)
                return left_result, right_result

        return _Parallel()

    def __repr__(self) -> str:
        return (
            f"Parallel(left={type(self.left).__name__}, "
            f"right={type(self.right).__name__}, "
            f"flow={self.input_type.name}->{self.output_type.name})"
        )


@dataclass(frozen=True)
class Conditional(Operation):
    """Select between two branches using a predicate."""

    condition: Callable[[Any], bool]
    if_true: Composable
    if_false: Composable

    def __post_init__(self) -> None:
        if not callable(self.condition):
            raise TypeError("Conditional: condition must be callable")
        true_checked: Composable = _ensure_composable(self.if_true)
        false_checked: Composable = _ensure_composable(self.if_false)
        if (
            true_checked.input_type is not false_checked.input_type
            or true_checked.output_type is not false_checked.output_type
        ):
            raise TypeError(
                "Conditional branch mismatch: "
                f"if_true {true_checked.input_type.name}->{true_checked.output_type.name} vs "
                f"if_false {false_checked.input_type.name}->{false_checked.output_type.name}"
            )
        object.__setattr__(self, "if_true", true_checked)
        object.__setattr__(self, "if_false", false_checked)
        object.__setattr__(self, "input_type", true_checked.input_type)
        object.__setattr__(self, "output_type", true_checked.output_type)

    def _build_module(self) -> dspy.Module:
        """Return a DSPy module selecting a branch by predicate."""
        true_mod: dspy.Module = self.if_true.to_module()
        false_mod: dspy.Module = self.if_false.to_module()
        predicate: Callable[[Any], bool] = self.condition

        class _Conditional(dspy.Module):
            def forward(self, value: Any) -> Any:
                if predicate(value):
                    return true_mod(value)
                return false_mod(value)

        return _Conditional()

    def __repr__(self) -> str:
        return (
            f"Conditional(if_true={type(self.if_true).__name__}, "
            f"if_false={type(self.if_false).__name__}, "
            f"flow={self.input_type.name}->{self.output_type.name})"
        )


@dataclass(frozen=True)
class Iterative(Operation):
    """Repeat an operation until results stabilize."""

    body: Composable

    def __post_init__(self) -> None:
        body_checked: Composable = _ensure_composable(self.body)
        object.__setattr__(self, "body", body_checked)
        object.__setattr__(self, "input_type", body_checked.input_type)
        object.__setattr__(self, "output_type", body_checked.output_type)

    def _build_module(self) -> dspy.Module:
        """Return a DSPy module that iterates until convergence."""
        body_mod: dspy.Module = self.body.to_module()

        class _Iterative(dspy.Module):
            def forward(self, value: Any) -> Any:
                prev: Any = value
                while True:
                    nxt: Any = body_mod(prev)
                    if nxt == prev:
                        return nxt
                    prev = nxt

        return _Iterative()

    def __repr__(self) -> str:
        return (
            f"Iterative(body={type(self.body).__name__}, "
            f"type={self.input_type.name})"
        )


# ---------------------------------------------------------------------------
# Non-discrete specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreeForm:
    """Non-discrete output template."""

    shape: str
    must_include: Tuple[str, ...] = ()


@dataclass(frozen=True)
class NaturalSpec:
    """Natural language task description within typed bounds."""

    domain: str
    goal: str
    constraints: Tuple[str, ...] = ()
    output_format: Optional[FreeForm] = None

    def to_module(self) -> dspy.Module:
        """Generate a DSPy module from the natural specification."""
        prompt: str = self.goal
        if self.constraints:
            prompt = f"{prompt}. Constraints: {'; '.join(self.constraints)}"
        signature: str = f"{self.domain}_input->output"
        return dspy.Predict(signature)

@runtime_checkable
class Process(ToDSPyModule, Protocol):
    """Protocol for executable processes."""

    pass


# ---------------------------------------------------------------------------
# Task specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec:
    """Complete task specification built from primitives."""

    description: str
    input: Optional[TypePrimitive]
    process: Optional[Process]
    output: Optional[TypePrimitive]
    test_cases: Sequence[Tuple[Any, Any]] = ()
    expected_signature: Optional[str] = None

    def to_module(self) -> Optional[dspy.Module]:
        """Compile the task's process into a DSPy module if possible."""
        if isinstance(self.process, Composable):
            return self.process.to_module()
        if isinstance(self.process, NaturalSpec):
            return self.process.to_module()
        return None


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
    """Build a TaskSpec from simple string cases."""

    parsed: List[Example] = [_case(c) for c in cases]
    proc_checked: Optional[Process] = None
    if process is not None:
        if isinstance(process, NaturalSpec):
            proc_checked = process
        else:
            proc_checked = _ensure_composable(process)
    return TaskSpec(
        description=description,
        input=input_type,
        process=proc_checked,
        output=output_type,
        test_cases=parsed,
    )


# ---------------------------------------------------------------------------
# Example task library
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Classify(Operation):
    """Classify input into one of several labels."""

    labels: Tuple[str, ...]
    input_type_override: TypePrimitive = Text

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_type", self.input_type_override)
        object.__setattr__(self, "output_type", literal_type(self.labels))

    def _build_module(self) -> dspy.Module:
        """Return a DSPy classifier module for the given labels."""
        label_names: str = ",".join(self._sanitize(label) for label in self.labels)
        signature: str = f"{self.input_type.name.lower()}->{label_names}"
        return dspy.Predict(signature)

    @staticmethod
    def _sanitize(label: str) -> str:
        """Sanitize ``label`` for safe use in signatures."""
        if not re.fullmatch(r"[A-Za-z0-9_]+", label):
            raise ValueError(f"Invalid label: {label!r}")
        return label


TASK_LIBRARY: List[TaskSpec] = [
    task(
        "Classify sentiment in product reviews",
        [
            "['The phone is fantastic and works great.', 'Battery died after two days.'] -> ['positive', 'negative']",
        ],
        input_type=list_type(Text),
        output_type=list_type(literal_type(("positive", "negative"))),
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
        output_type=list_type(TypePrimitive("Relationship")),
    ),
]


def describe(op: Composable, indent: int = 0) -> str:
    """Return a multi-line description of ``op`` and its children."""

    pad: str = "  " * indent
    lines: List[str] = [f"{pad}{repr(op)}"]
    if isinstance(op, Sequential):
        lines.append(describe(op.first, indent + 1))
        lines.append(describe(op.second, indent + 1))
    elif isinstance(op, Parallel):
        lines.append(describe(op.left, indent + 1))
        lines.append(describe(op.right, indent + 1))
    elif isinstance(op, Conditional):
        lines.append(describe(op.if_true, indent + 1))
        lines.append(describe(op.if_false, indent + 1))
    elif isinstance(op, Iterative):
        lines.append(describe(op.body, indent + 1))
    return "\n".join(lines)

__all__: List[str] = [
    "TASK_LIBRARY",
    "task",
    "TaskSpec",
    "TypePrimitive",
    "get_type",
    "ListType",
    "Literal",
    "TupleType",
    "UnionType",
    "list_type",
    "literal_type",
    "tuple_type",
    "union_type",
    "Map",
    "Filter",
    "Reduce",
    "Sequential",
    "Parallel",
    "Conditional",
    "Iterative",
    "describe",
    "FreeForm",
    "NaturalSpec",
]
