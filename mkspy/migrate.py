from typing import Any, Dict, List, Optional

from .task_library import (
    NaturalSpec,
    Process,
    TaskSpec,
    TypePrimitive,
    _ensure_composable,
    task,
)


def dict_to_spec(data: Dict[str, Any]) -> TaskSpec:
    description: Any = data.get("description")
    if not isinstance(description, str):
        raise ValueError("Task description missing or not a string")
    raw_cases: Any = data.get("cases", [])
    if not isinstance(raw_cases, list) or not all(isinstance(c, str) for c in raw_cases):
        raise ValueError("cases must be a list of strings")
    input_type_raw: Any = data.get("input")
    if input_type_raw is not None and not isinstance(input_type_raw, TypePrimitive):
        raise ValueError("input must be TypePrimitive")
    output_type_raw: Any = data.get("output")
    if output_type_raw is not None and not isinstance(output_type_raw, TypePrimitive):
        raise ValueError("output must be TypePrimitive")
    process_raw: Any = data.get("process")
    process: Optional[Process] = None
    if process_raw is not None:
        if isinstance(process_raw, dict):
            raise ValueError("nested dict processes unsupported")
        if isinstance(process_raw, NaturalSpec):
            process = process_raw
        else:
            process = _ensure_composable(process_raw)
    return task(
        description=description,
        cases=raw_cases,
        input_type=input_type_raw,
        output_type=output_type_raw,
        process=process,
    )


def migrate_tasks(tasks: List[Dict[str, Any]]) -> List[TaskSpec]:
    return [dict_to_spec(t) for t in tasks]
