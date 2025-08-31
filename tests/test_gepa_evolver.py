from pathlib import Path
from mkspy.gepa_evolver import GEPAEvolver


def test_dataset_creation(tmp_path: Path) -> None:
    tasks = [
        {
            "description": "Increment number",
            "input_types": "int",
            "output_requirements": "int",
            "test_cases": [{"input": 1, "expected": 2}],
        }
    ]
    evolver = GEPAEvolver(task_library=tasks, output_dir=tmp_path)
    assert len(evolver.trainset) == 1
    assert len(evolver.valset) == 0
