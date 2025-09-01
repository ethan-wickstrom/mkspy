"""Simple profiling of DSL constructions and registry access."""

from __future__ import annotations

import timeit
import tracemalloc

from mkspy.task_library import Classify, Map, Sequential, get_type, Operation


def _build_chain(depth: int) -> Operation:
    current: Operation = Classify(labels=("a",))
    for _ in range(depth - 1):
        next_op: Classify = Classify(labels=("a",))
        current = Sequential(first=current, second=next_op)
    return current


def main() -> None:
    def _map_construct() -> None:
        Map(fn=Classify(labels=("a",)))

    duration: float = timeit.timeit(_map_construct, number=1000)
    print(f"Map construction: {duration:.4f}s/1000")

    chain: Operation = _build_chain(20)
    to_mod: float = timeit.timeit(lambda: chain.to_module(), number=100)
    print(f"Sequential.to_module(): {to_mod:.4f}s/100")

    def _lookup() -> None:
        get_type("Text")

    lookup: float = timeit.timeit(_lookup, number=10000)
    print(f"Registry lookup: {lookup:.4f}s/10000")

    tracemalloc.start()
    _build_chain(50)
    current: int
    peak: int
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Nested chain memory: current={current} bytes peak={peak} bytes")


if __name__ == "__main__":
    main()
