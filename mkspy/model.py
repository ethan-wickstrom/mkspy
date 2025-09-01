from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

"""Declarative DSPy program model and minimal codegen.

This module defines a small, typed AST for describing DSPy programs and a
deterministic renderer to Python. It keeps behavior (Signatures) separate from
strategy (Modules), aligning with DSPy principles and enabling optimizers to
operate later without changing the frontend program.
"""

from .codegen import CodeBuilder

logger = logging.getLogger(__name__)

# Allowed dspy.Module constructors we support when wiring in __init__.
SAFE_MODULE_TYPES: tuple[str, ...] = ("Predict", "ChainOfThought", "ReAct")


# -----------------------------
# Core nodes & simple codegen
# -----------------------------

@dataclass
class DSPyImport:
    module: str | None = None
    alias: str | None = None
    from_module: str | None = None
    imported_names: list[str] = field(default_factory=list)  # simple: names only

    def to_code(self) -> str:
        if self.from_module:
            if self.imported_names:
                return f"from {self.from_module} import {', '.join(self.imported_names)}"
            return f"from {self.from_module} import *"
        if not self.module:
            return ""
        if self.alias:
            return f"import {self.module} as {self.alias}"
        return f"import {self.module}"


@dataclass
class DSPyField:
    name: str
    field_type: str = "str"
    description: str = ""
    is_input: bool = True

    def to_code(self, cb: CodeBuilder) -> None:
        prefix = "dspy.InputField" if self.is_input else "dspy.OutputField"
        desc = f'desc="{self.description}"' if self.description else ""
        cb.write(f"{self.name}: {self.field_type} = {prefix}({desc})".rstrip())


@dataclass
class DSPySignature:
    name: str
    docstring: str = ""
    inputs: list[DSPyField] = field(default_factory=list)
    outputs: list[DSPyField] = field(default_factory=list)

    def to_code(self, cb: CodeBuilder) -> None:
        cb.write(f"class {self.name}(dspy.Signature):")
        with cb.block():
            if self.docstring:
                cb.write('"""')
                for ln in self.docstring.strip().splitlines():
                    cb.write(ln)
                cb.write('"""')
            for fld in self.inputs:
                fld.to_code(cb)
            for fld in self.outputs:
                fld.to_code(cb)


@dataclass
class DSPyParameter:
    name: str
    param_type: str | None = None
    default_value: str | None = None

    def as_sig(self) -> str:
        s = self.name
        if self.param_type:
            s += f": {self.param_type}"
        if self.default_value is not None:
            s += f" = {self.default_value}"
        return s


@dataclass
class DSPyStatement:
    """Base class for statements."""
    def to_code(self, cb: CodeBuilder) -> None:
        raise NotImplementedError


@dataclass
class DSPyAssignment(DSPyStatement):
    target: str
    value: str
    def to_code(self, cb: CodeBuilder) -> None:
        cb.write(f"{self.target} = {self.value}")


@dataclass
class DSPyReturn(DSPyStatement):
    value: str
    def to_code(self, cb: CodeBuilder) -> None:
        cb.write(f"return {self.value}")


@dataclass
class DSPyMethodCall(DSPyStatement):
    target: str
    method: str = ""
    args: list[str] = field(default_factory=list)
    kwargs: dict[str, str] = field(default_factory=dict)

    def to_code(self, cb: CodeBuilder) -> None:
        accessor = f"{self.target}.{self.method}" if self.method else self.target
        parts: list[str] = []
        parts.extend(self.args)
        parts.extend([f"{k}={v}" for k, v in self.kwargs.items()])
        cb.write(f"{accessor}({', '.join(parts)})")


@dataclass
class DSPyIf(DSPyStatement):
    condition: str
    body: list[DSPyStatement] = field(default_factory=list)
    else_body: list[DSPyStatement] = field(default_factory=list)

    def to_code(self, cb: CodeBuilder) -> None:
        cb.write(f"if {self.condition}:")
        with cb.block():
            if not self.body:
                cb.write("pass")
            for st in self.body:
                st.to_code(cb)
        if self.else_body:
            cb.write("else:")
            with cb.block():
                for st in self.else_body:
                    st.to_code(cb)


@dataclass
class DSPyFor(DSPyStatement):
    target: str
    iterable: str
    body: list[DSPyStatement] = field(default_factory=list)

    def to_code(self, cb: CodeBuilder) -> None:
        cb.write(f"for {self.target} in {self.iterable}:")
        with cb.block():
            if not self.body:
                cb.write("pass")
            for st in self.body:
                st.to_code(cb)


@dataclass
class DSPyMethod:
    name: str
    parameters: list[DSPyParameter] = field(default_factory=list)
    return_type: str | None = None
    docstring: str = ""
    body: list[DSPyStatement] = field(default_factory=list)

    def to_code(self, cb: CodeBuilder) -> None:
        sig = ", ".join(p.as_sig() for p in self.parameters)
        head = f"def {self.name}({sig})"
        if self.return_type:
            head += f" -> {self.return_type}"
        head += ":"
        cb.write(head)
        with cb.block():
            if self.docstring:
                cb.write('"""')
                for ln in self.docstring.strip().splitlines():
                    cb.write(ln)
                cb.write('"""')
            if not self.body:
                cb.write("pass")
            else:
                for st in self.body:
                    st.to_code(cb)


@dataclass
class DSPyClass:
    name: str
    base_classes: list[str] = field(default_factory=lambda: ["dspy.Module"])
    docstring: str = ""
    attrs: dict[str, str] = field(default_factory=dict)
    methods: list[DSPyMethod] = field(default_factory=list)

    def to_code(self, cb: CodeBuilder) -> None:
        bases = f"({', '.join(self.base_classes)})" if self.base_classes else ""
        cb.write(f"class {self.name}{bases}:")
        with cb.block():
            if self.docstring:
                cb.write('"""')
                for ln in self.docstring.strip().splitlines():
                    cb.write(ln)
                cb.write('"""')
            for k, v in self.attrs.items():
                cb.write(f"{k} = {v}")
            for i, m in enumerate(self.methods):
                if i > 0 or self.attrs or self.docstring:
                    cb.write()
                m.to_code(cb)
            if not (self.methods or self.attrs or self.docstring):
                cb.write("pass")


@dataclass
class DSPySignatureBinding:
    """A registry entry tying a submodule to a signature by name."""
    module_name: str
    signature_name: str
    module_type: str = "Predict"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class DSPyProgram:
    imports: list[DSPyImport] = field(default_factory=list)
    signatures: list[DSPySignature] = field(default_factory=list)
    main_class: DSPyClass | None = None
    bound_modules: list[DSPySignatureBinding] = field(default_factory=list)
    program_var: str = "program"

    # ----------- codegen ------------

    def to_code(self) -> str:
        cb = CodeBuilder()

        # imports (dedup by text)
        seen = set()
        for imp in self.imports:
            line = imp.to_code()
            if line and line not in seen:
                cb.write(line)
                seen.add(line)

        # signatures
        for sig in self.signatures:
            cb.write()
            sig.to_code(cb)

        # main class + init wires for bound modules
        if self.main_class:
            cb.write()
            # ensure __init__ exists if we have bindings
            if self.bound_modules and not any(m.name == "__init__" for m in self.main_class.methods):
                self.main_class.methods.insert(0, DSPyMethod(name="__init__", parameters=[DSPyParameter("self")]))

            # wire modules in __init__
            if self.bound_modules:
                init_m = next(m for m in self.main_class.methods if m.name == "__init__")
                # validate bindings upfront
                sig_names = {s.name for s in self.signatures}
                for bind in self.bound_modules:
                    if bind.module_type not in SAFE_MODULE_TYPES:
                        raise ValueError(
                            f"Unsupported module_type '{bind.module_type}'. Allowed: {SAFE_MODULE_TYPES}"
                        )
                    if bind.signature_name not in sig_names:
                        raise ValueError(
                            f"Unknown signature '{bind.signature_name}' for binding '{bind.module_name}'"
                        )
                    ctor = f"dspy.{bind.module_type}({bind.signature_name}"
                    for k, v in bind.parameters.items():
                        ctor += f", {k}={repr(v)}"
                    ctor += ")"
                    init_m.body.append(DSPyAssignment(f"self.{bind.module_name}", ctor))

            self.main_class.to_code(cb)

        # export
        if self.main_class:
            cb.write()
            cb.write(f"{self.program_var} = {self.main_class.name}()")

        return cb.render()
