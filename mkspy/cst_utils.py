from __future__ import annotations

from dataclasses import dataclass, field

import libcst as cst


@dataclass(frozen=True)
class DSPyProgramReport:
    """Structural report for a DSPy Python module, extracted with LibCST."""

    parsed: bool
    parse_error: str | None = None

    # Imports and symbols
    has_dspy_import: bool = False
    imported_symbols: set[str] = field(default_factory=set)

    # Class inventory
    signature_classes: set[str] = field(default_factory=set)
    module_classes: set[str] = field(default_factory=set)

    # Methods found
    forward_in_classes: set[str] = field(default_factory=set)  # class names that define forward()/aforward()

    # Exports
    program_assignments: set[str] = field(default_factory=set)  # names assigned to an instance of a Module subclass

    # Calls to DSPy constructors
    ctor_counts: dict[str, int] = field(
        default_factory=lambda: {"Predict": 0, "ChainOfThought": 0, "ReAct": 0}
    )

    def total_predictors(self) -> int:
        return sum(self.ctor_counts.values())

    @property
    def has_forward(self) -> bool:
        return bool(self.forward_in_classes)

    @property
    def has_program_assignment(self) -> bool:
        return bool(self.program_assignments)


def _attr_name(node: cst.BaseExpression) -> str | None:
    """Return dotted name for an Attribute/Name, e.g., 'dspy.Module'."""
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        left = _attr_name(node.value)
        right = node.attr.value if isinstance(node.attr, cst.Name) else None
        if left and right:
            return f"{left}.{right}"
    return None


class _DSPyVisitor(cst.CSTVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.has_dspy_import: bool = False
        self.imported_symbols: set[str] = set()

        self.signature_classes: set[str] = set()
        self.module_classes: set[str] = set()
        self.forward_in_classes: set[str] = set()

        self.program_assignments: set[str] = set()
        self.ctor_counts: dict[str, int] = {"Predict": 0, "ChainOfThought": 0, "ReAct": 0}

        # Nesting trackers
        self._class_stack: list[str] = []
        self._function_depth: int = 0

    # ---------- Imports ----------

    def visit_Import(self, node: cst.Import) -> None:
        for alias in node.names:
            nm = alias.name
            if isinstance(nm, cst.Name) and nm.value == "dspy":
                self.has_dspy_import = True

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if node.module is None:
            return
        mod_name = _attr_name(node.module)
        if mod_name == "dspy":
            self.has_dspy_import = True
            if isinstance(node.names, cst.ImportStar):
                # Star import; we can't enumerate names, but it's still a dspy import.
                return
            for alias in node.names or []:
                if isinstance(alias, cst.ImportAlias):
                    # Support "from dspy import X as Y"
                    if alias.asname and isinstance(alias.asname.name, cst.Name):
                        self.imported_symbols.add(alias.asname.name.value)
                    else:
                        name_node = alias.name
                        if isinstance(name_node, cst.Name):
                            self.imported_symbols.add(name_node.value)
                        elif isinstance(name_node, cst.Attribute):
                            # Rare, but record rightmost part.
                            right = name_node.attr.value if isinstance(name_node.attr, cst.Name) else None
                            if right:
                                self.imported_symbols.add(right)

    # ---------- Classes and methods ----------

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        bases = [b.value for b in node.bases]
        base_names = {_attr_name(v) for v in bases if v is not None}
        is_signature = ("dspy.Signature" in base_names) or ("Signature" in base_names)
        is_module = ("dspy.Module" in base_names) or ("Module" in base_names)
        if is_signature:
            self.signature_classes.add(node.name.value)
        if is_module:
            self.module_classes.add(node.name.value)
        # Push class name to track forward() ownership later
        self._class_stack.append(node.name.value)

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        if self._class_stack:
            self._class_stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self._function_depth += 1
        # Track forward/aforward defined inside a Module subclass
        if self._class_stack:
            current_cls = self._class_stack[-1]
            if node.name.value in {"forward", "aforward"} and current_cls in self.module_classes:
                self.forward_in_classes.add(current_cls)

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        self._function_depth -= 1

    # ---------- Top-level assignments ----------

    def visit_Assign(self, node: cst.Assign) -> None:
        # Only consider module-level assignments (outside classes/functions)
        if self._class_stack or self._function_depth > 0:
            return
        if len(node.targets) != 1:
            return
        tgt = node.targets[0].target
        if not isinstance(tgt, cst.Name):
            return
        # program = MainClass()
        val = node.value
        if isinstance(val, cst.Call):
            fn = val.func
            # Call to a Name referencing a known Module subclass
            if isinstance(fn, cst.Name) and fn.value in self.module_classes:
                self.program_assignments.add(tgt.value)

    # ---------- Calls ----------

    def visit_Call(self, node: cst.Call) -> None:
        fn = node.func
        if isinstance(fn, cst.Attribute):
            base = _attr_name(fn.value)
            attr = fn.attr.value if isinstance(fn.attr, cst.Name) else None
            if base == "dspy" and attr in self.ctor_counts:
                self.ctor_counts[attr] += 1
        elif isinstance(fn, cst.Name):
            # from dspy import Predict
            if fn.value in self.ctor_counts:
                self.ctor_counts[fn.value] += 1


def inspect_dspy_program(code: str) -> DSPyProgramReport:
    """Parse and analyze a Python module containing a DSPy program."""
    try:
        mod = cst.parse_module(code)
    except cst.ParserSyntaxError as e:
        return DSPyProgramReport(parsed=False, parse_error=str(e))

    v = _DSPyVisitor()
    mod.visit(v)
    return DSPyProgramReport(
        parsed=True,
        has_dspy_import=v.has_dspy_import,
        imported_symbols=v.imported_symbols,
        signature_classes=v.signature_classes,
        module_classes=v.module_classes,
        forward_in_classes=v.forward_in_classes,
        program_assignments=v.program_assignments,
        ctor_counts=v.ctor_counts,
    )


def score_program(code: str) -> tuple[float, str]:
    """Compute a structural score (0..1) and feedback for a DSPy program.

    Weights (sum to 1.0):
      - Parsable by LibCST (0.2)
      - Has a dspy import (0.1)
      - Has a Signature subclass (0.2)
      - Has a Module subclass (0.2)
      - Has forward/aforward (0.1)
      - Exports a program var assigned to a Module subclass (0.1)
      - Uses Predict/ChainOfThought/ReAct (0.1)
    """
    report = inspect_dspy_program(code)
    score = 0.0
    notes: list[str] = []

    if report.parsed:
        score += 0.2
    else:
        notes.append(f"Parse error: {report.parse_error}")
        # Parsing failed; short-circuit remaining checks
        return 0.0, "\n".join(notes)

    if report.has_dspy_import:
        score += 0.1
    else:
        notes.append("Missing `import dspy` or `from dspy import ...`.")

    if report.signature_classes:
        score += 0.2
        notes.append(f"✓ Found {len(report.signature_classes)} Signature class(es): {sorted(report.signature_classes)}")
    else:
        notes.append("Missing `class <Name>(dspy.Signature)`.")

    if report.module_classes:
        score += 0.2
        notes.append(f"✓ Found {len(report.module_classes)} Module class(es): {sorted(report.module_classes)}")
    else:
        notes.append("Missing `class <Name>(dspy.Module)`.")

    if report.has_forward:
        score += 0.1
    else:
        notes.append("Missing `forward(...)` or `aforward(...)` in a Module subclass.")

    if report.has_program_assignment:
        score += 0.1
    else:
        notes.append("Missing `program = <ModuleClass>()` export.")

    if report.total_predictors() > 0:
        score += 0.1
    else:
        notes.append("No `dspy.Predict`, `dspy.ChainOfThought`, or `dspy.ReAct` calls found.")

    return min(score, 1.0), "\n".join(notes)


def validate_program(code: str) -> tuple[bool, list[str]]:
    """Return (ok, errors) for core DSPy structural expectations."""
    report = inspect_dspy_program(code)
    if not report.parsed:
        return False, [f"Parse error: {report.parse_error}"]

    errors: list[str] = []
    if not report.has_dspy_import:
        errors.append("Missing `import dspy` or `from dspy import ...`.")
    if not report.signature_classes:
        errors.append("At least one `dspy.Signature` subclass is required.")
    if not report.module_classes:
        errors.append("At least one `dspy.Module` subclass is required.")
    if not report.has_forward:
        errors.append("A `forward(...)` or `aforward(...)` method is required in a Module subclass.")
    if not report.has_program_assignment:
        errors.append("Export a program instance: `program = <YourModuleClass>()`.")
    if report.total_predictors() == 0:
        errors.append("Construct at least one of `dspy.Predict`, `dspy.ChainOfThought`, or `dspy.ReAct`.")

    return (len(errors) == 0), errors