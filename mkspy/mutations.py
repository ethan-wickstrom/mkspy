from __future__ import annotations

import logging
from typing import Any, Protocol, Sequence, TypeVar, runtime_checkable
from abc import ABC, abstractmethod
import random

from .model import (
    DSPyProgram, DSPyImport, DSPySignature, DSPyField, DSPyMethod, DSPyAssignment,
    DSPyReturn, DSPyIf, DSPyFor, DSPyParameter, DSPySignatureBinding, SAFE_MODULE_TYPES,
)
from .types import ImportSpec, SignatureSpec, ModuleSpec, FieldSpec, mk_field

logger = logging.getLogger(__name__)


_T = TypeVar("_T")

@runtime_checkable
class RNG(Protocol):
    def random(self) -> float: ...
    def choice(self, seq: Sequence[_T]) -> _T: ...
    def randint(self, a: int, b: int) -> int: ...


def ensure_forward(program: DSPyProgram) -> DSPyMethod:
    if program.main_class is None:
        raise ValueError("Program has no main class.")
    for m in program.main_class.methods:
        if m.name == "forward":
            return m
    f = DSPyMethod(
        name="forward",
        parameters=[DSPyParameter("self"), DSPyParameter("question", "str")],
        return_type="dspy.Prediction",
        body=[DSPyReturn("None")],
    )
    program.main_class.methods.append(f)
    return f


class Mutation(ABC):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def applicable(self, p: DSPyProgram) -> bool: ...

    @abstractmethod
    def apply(self, p: DSPyProgram, rng: RNG) -> None: ...


class AddImport(Mutation):
    def __init__(self, candidates: list[ImportSpec]):
        super().__init__("AddImport")
        self.candidates = candidates

    def applicable(self, p: DSPyProgram) -> bool:
        return bool(self.candidates)

    def apply(self, p: DSPyProgram, rng: RNG) -> None:
        cand = rng.choice(self.candidates)
        imp = DSPyImport(
            module=cand.get("module"),
            alias=cand.get("alias"),
            from_module=cand.get("from_module"),
            imported_names=list(cand.get("imported_names", [])),
        )
        text = imp.to_code()
        if text and text not in {i.to_code() for i in p.imports}:
            p.imports.append(imp)


class AddSignature(Mutation):
    def __init__(self, candidates: list[SignatureSpec]):
        super().__init__("AddSignature")
        self.candidates = candidates

    def applicable(self, p: DSPyProgram) -> bool:
        return bool(self.candidates)

    def apply(self, p: DSPyProgram, rng: RNG) -> None:
        sd = rng.choice(self.candidates)
        if sd["name"] not in {s.name for s in p.signatures}:
            p.signatures.append(
                DSPySignature(
                    name=sd["name"],
                    docstring=sd.get("docstring", ""),
                    inputs=[mk_field(fd) for fd in sd.get("inputs", [])],
                    outputs=[mk_field(fd) for fd in sd.get("outputs", [])],
                )
            )


class BindModule(Mutation):
    def __init__(self, candidates: list[ModuleSpec]):
        super().__init__("BindModule")
        self.candidates = candidates

    def applicable(self, p: DSPyProgram) -> bool:
        return p.main_class is not None and bool(self.candidates)

    def apply(self, p: DSPyProgram, rng: RNG) -> None:
        cd = rng.choice(self.candidates)
        mod_name = cd["name"]
        mod_type = cd["module_type"] if cd["module_type"] in SAFE_MODULE_TYPES else "Predict"
        # existing signature or any
        if p.signatures:
            sig_name = cd.get("signature_ref") or rng.choice(p.signatures).name
        else:
            # minimal bootstrap signature
            sig = DSPySignature(
                name="BootstrapSignature",
                inputs=[DSPyField("question", "str", "Question")],
                outputs=[DSPyField("answer", "str", "Answer", is_input=False)],
            )
            p.signatures.append(sig)
            sig_name = sig.name
        p.bound_modules.append(DSPySignatureBinding(module_name=mod_name, signature_name=sig_name, module_type=mod_type))


class AddCallInForward(Mutation):
    def __init__(self):
        super().__init__("AddCallInForward")

    def applicable(self, p: DSPyProgram) -> bool:
        return p.main_class is not None and bool(p.bound_modules)

    def apply(self, p: DSPyProgram, rng: RNG) -> None:
        fwd = ensure_forward(p)
        mod_name = rng.choice(p.bound_modules).module_name
        # guess an arg name
        params = [pr.name for pr in fwd.parameters if pr.name != "self"]
        arg = params[0] if params else "question"
        tmp = f"out_{mod_name}"
        call = DSPyAssignment(tmp, f"self.{mod_name}({arg}={arg})")
        # place before final return if exists
        if fwd.body and isinstance(fwd.body[-1], DSPyReturn):
            fwd.body.insert(-1, call)
        else:
            fwd.body.append(call)
        if not fwd.body or not isinstance(fwd.body[-1], DSPyReturn):
            fwd.body.append(DSPyReturn(tmp))


class AddControlFlow(Mutation):
    def __init__(self):
        super().__init__("AddControlFlow")

    def applicable(self, p: DSPyProgram) -> bool:
        return p.main_class is not None

    def apply(self, p: DSPyProgram, rng: RNG) -> None:
        fwd = ensure_forward(p)
        params = [pr.name for pr in fwd.parameters if pr.name != "self"] or ["question"]
        src = rng.choice(params)
        if rng.random() < 0.5:
            stmt = DSPyIf(condition=f"len({src}) > 10", body=[DSPyAssignment("flag", '"long"')], else_body=[DSPyAssignment("flag", '"short"')])
        else:
            stmt = DSPyFor(target="tok", iterable=f"{src}.split()", body=[DSPyAssignment("last_tok", "tok")])
        if fwd.body and isinstance(fwd.body[-1], DSPyReturn):
            fwd.body.insert(-1, stmt)
        else:
            fwd.body.append(stmt)


def default_mutations(import_lib: list[ImportSpec],
                      signature_lib: list[SignatureSpec],
                      module_lib: list[ModuleSpec]) -> list[Mutation]:
    return [
        AddImport(import_lib),
        AddSignature(signature_lib),
        BindModule(module_lib),
        AddCallInForward(),
        AddControlFlow(),
    ]


class RandomRNG:
    """Default RNG wrapper (allows swapping with a deterministic one)."""
    def __init__(self, seed: int | None = None) -> None:
        self._r = random.Random(seed)
    def random(self) -> float: return self._r.random()
    def choice(self, seq: Sequence[_T]) -> _T: return self._r.choice(seq)  # type: ignore[no-any-return]
    def randint(self, a: int, b: int) -> int: return self._r.randint(a, b)

    
