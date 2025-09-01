from typing import Any, Dict, List, Optional

from .task_library import Process, TaskSpec, TypePrimitive, task


def dict_to_spec(data: Dict[str, Any]) -> TaskSpec:
    description: str = data["description"]
    cases: List[str] = data.get("cases", [])
    input_type: Optional[TypePrimitive] = data.get("input")
    output_type: Optional[TypePrimitive] = data.get("output")
    process: Optional[Process] = data.get("process")
    return task(
        description=description,
        cases=cases,
        input_type=input_type,
        output_type=output_type,
        process=process,
    )


def migrate_tasks(tasks: List[Dict[str, Any]]) -> List[TaskSpec]:
    return [dict_to_spec(t) for t in tasks]
