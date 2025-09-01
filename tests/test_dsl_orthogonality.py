from hypothesis import given, strategies as st
import pytest
from typing import Any, List

from mkspy.task_library import (
    Classify,
    Conditional,
    get_type,
    ListType,
    Literal,
    Map,
    Number,
    Parallel,
    Sequential,
    TaskSpec,
    Text,
    TupleType,
    _ensure_composable,
)
from mkspy.migrate import dict_to_spec


@given(st.lists(st.text(min_size=1), min_size=1, max_size=3))
def test_map_output_type(labels: list[str]) -> None:
    cls: Classify = Classify(labels=tuple(labels))
    m: Map = Map(fn=cls)
    assert m.input_type == ListType(Text)
    assert m.output_type == ListType(Literal(tuple(labels)))


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


def test_type_primitive_singleton() -> None:
    assert get_type("Text") is Text


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
    module = m.to_module()
    result: List[Any] = module(["x"])
    assert result == [None]


def test_classify_to_module_instance() -> None:
    cls: Classify = Classify(labels=("a",))
    module = cls.to_module()
    assert hasattr(module, "forward")
