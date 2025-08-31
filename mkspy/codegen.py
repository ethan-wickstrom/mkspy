from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CodeBuilder:
    """
    Minimal, explicit indentation-aware code builder.
    Avoids brittle string operations when emitting Python.
    """
    indent: str = "  "
    lines: list[str] = field(default_factory=list)
    _level: int = 0

    def write(self, line: str = "") -> None:
        self.lines.append(f"{self.indent * self._level}{line}")

    def writelines(self, raw: str) -> None:
        for ln in raw.splitlines():
            self.write(ln)

    def block(self) -> "_Block":
        return _Block(self)

    def render(self) -> str:
        code = "\n".join(self.lines)
        if not code.endswith("\n"):
            code += "\n"
        return code


class _Block:
    def __init__(self, cb: CodeBuilder) -> None:
        self.cb = cb

    def __enter__(self) -> None:
        self.cb._level += 1

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cb._level -= 1
