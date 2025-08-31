from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from .model import DSPyField


class FieldSpec(TypedDict):
    name: str
    field_type: NotRequired[str]
    description: NotRequired[str]
    is_input: NotRequired[bool]


class ImportSpec(TypedDict, total=False):
    module: str
    alias: str
    from_module: str
    imported_names: list[str]


class SignatureSpec(TypedDict):
    name: str
    docstring: NotRequired[str]
    inputs: NotRequired[list[FieldSpec]]
    outputs: NotRequired[list[FieldSpec]]


class ModuleSpec(TypedDict):
    name: str
    module_type: NotRequired[str]
    signature_ref: NotRequired[str]
    parameters: NotRequired[dict[str, Any]]


def mk_field(fd: FieldSpec) -> DSPyField:
    return DSPyField(
        name=fd["name"],
        field_type=fd.get("field_type", "str"),
        description=fd.get("description", ""),
        is_input=bool(fd.get("is_input", True)),
    )
