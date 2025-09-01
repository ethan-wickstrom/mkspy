from pathlib import Path
from mkspy.gepa_evolver import GEPAEvolver
from mkspy.task_library import task


def test_dataset_creation(tmp_path: Path) -> None:
    tasks = [task("Increment number", ["1 -> 2"])]
    evolver = GEPAEvolver(task_library=tasks, output_dir=tmp_path)
    assert len(evolver.trainset) == 1
    assert len(evolver.valset) == 0
