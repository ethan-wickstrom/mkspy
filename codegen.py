class CodeBuilder:
    def __init__(self) -> None:
        self._parts: list[str] = []

    def add_line(self, line: str) -> None:
        self._parts.append(line)

    def build(self) -> str:
        return "\n".join(self._parts)
