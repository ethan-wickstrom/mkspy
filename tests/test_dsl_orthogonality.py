from hypothesis import given, strategies as st
import pytest
from typing import Any, List

from mkspy.task_library import (
    Classify,
    Conditional,
    Parallel,
    Sequential,
    TaskSpec,
    Text,
    Number,
    TupleType,
    UnionType,
    TypePrimitive,
    Operation,
    _ensure_composable,
    describe,
    get_type,
    list_type,
    literal_type,
    union_type,
    Map,
    Filter,
    Reduce,
)
from dataclasses import dataclass
import dspy
from mkspy.migrate import dict_to_spec


@given(st.lists(st.text(min_size=1), min_size=1, max_size=3))
def test_map_output_type(labels: list[str]) -> None:
    cls: Classify = Classify(labels=tuple(labels))
    m: Map = Map(fn=cls)
    assert m.input_type is list_type(Text)
    assert m.output_type is list_type(literal_type(tuple(labels)))


def test_parallel_substitution() -> None:
    a: Classify = Classify(labels=("x",))
    b: Classify = Classify(labels=("y",))
    c: Classify = Classify(labels=("z",))
    p1: Parallel = Parallel(left=a, right=b)
    p2: Parallel = Parallel(left=a, right=c)
    assert p1.input_type == p2.input_type
    assert p1.left.output_type == p2.left.output_type


def test_parallel_output_tuple_type() -> None:
    a: Classify = Classify(labels=("x",))
    b: Classify = Classify(labels=("y",))
    p: Parallel = Parallel(left=a, right=b)
    assert isinstance(p.output_type, TupleType)
    assert p.output_type.elements == (a.output_type, b.output_type)


def test_sequential_type_mismatch() -> None:
    a: Classify = Classify(labels=("x",))
    b: Classify = Classify(labels=("y",))
    with pytest.raises(TypeError):
        Sequential(first=a, second=b)


def test_migration() -> None:
    legacy: dict[str, object] = {
        "description": "Echo text",
        "cases": ["\"a\" -> \"a\""]
    }
    spec: TaskSpec = dict_to_spec(legacy)
    assert isinstance(spec, TaskSpec)
    assert spec.description == "Echo text"


def test_migration_validation_error() -> None:
    with pytest.raises(ValueError):
        dict_to_spec({"cases": []})


def test_type_primitive_singleton() -> None:
    assert get_type("Text") is Text


def test_type_registry_collision() -> None:
    TypePrimitive("Foo")
    with pytest.raises(ValueError):
        TypePrimitive("Foo")


def test_conditional_requires_callable() -> None:
    cls: Classify = Classify(labels=("a",))
    bad_condition: Any = "x>0"
    with pytest.raises(TypeError, match="callable"):
        Conditional(condition=bad_condition, if_true=cls, if_false=cls)


def test_conditional_branch_mismatch_message() -> None:
    a: Classify = Classify(labels=("x",))
    b: Classify = Classify(labels=("y", "z"))
    with pytest.raises(TypeError, match="branch mismatch"):
        Conditional(condition=lambda x: True, if_true=a, if_false=b)


def test_classify_custom_input() -> None:
    cls: Classify = Classify(labels=("a",), input_type_override=Number)
    assert cls.input_type is Number


def test_ensure_composable_error() -> None:
    with pytest.raises(TypeError, match="Composable"):
        _ensure_composable(object())


def test_map_to_module_executes() -> None:
    cls: Classify = Classify(labels=("a",))
    m: Map = Map(fn=cls)
    result: List[Any] = m(["x"])
    assert result == [None]


def test_map_empty_input() -> None:
    cls: Classify = Classify(labels=("a",))
    m: Map = Map(fn=cls)
    assert m([]) == []


def test_filter_empty_input() -> None:
    cls: Classify = Classify(labels=("a",))
    f: Filter = Filter(predicate=cls)
    assert f([]) == []


def test_reduce_empty_raises() -> None:
    cls: Classify = Classify(labels=("a",))
    r: Reduce = Reduce(fn=cls)
    with pytest.raises(ValueError, match="empty input"):
        r([])


@dataclass(frozen=True)
class Fail(Operation):
    def __post_init__(self) -> None:
        object.__setattr__(self, "input_type", Text)
        object.__setattr__(self, "output_type", Text)

    def _build_module(self) -> dspy.Module:
        class _Fail(dspy.Module):
            def forward(self, value: Any) -> Any:
                raise ValueError("boom")

        return _Fail()


def test_error_propagation_sequential() -> None:
    fail: Fail = Fail()
    cls: Classify = Classify(labels=("a",))
    seq: Sequential = Sequential(first=fail, second=cls)
    with pytest.raises(ValueError, match="boom"):
        seq("x")


def test_module_reusability() -> None:
    cls: Classify = Classify(labels=("a",))
    m1 = cls.to_module()
    m2 = cls.to_module()
    assert m1 is m2
    assert cls("x") is None


def test_union_type_singleton() -> None:
    u1: UnionType = union_type((Text, Number))
    u2: UnionType = union_type((Text, Number))
    assert u1 is u2


def test_describe_outputs_tree() -> None:
    first: Classify = Classify(labels=("a",))
    second: Classify = Classify(labels=("b",), input_type_override=literal_type(("a",)))
    seq: Sequential = Sequential(first=first, second=second)
    text: str = describe(seq)
    assert "Sequential" in text and text.count("Classify") >= 2


def test_classify_to_module_instance() -> None:
    cls: Classify = Classify(labels=("a",))
    module = cls.to_module()
    assert hasattr(module, "forward")
