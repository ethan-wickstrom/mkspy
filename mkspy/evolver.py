from __future__ import annotations

import logging
from typing import Any, Callable

from .model import DSPyProgram, DSPyClass, DSPyMethod, DSPyParameter, DSPySignature, DSPyImport
from .mutations import default_mutations, RandomRNG, RNG, Mutation
from .validation import validate_program, prune_unused_imports, ValidationResult
from .types import ImportSpec, SignatureSpec, ModuleSpec, make_field

logger = logging.getLogger(__name__)


# Minimal defaults: declarative, strong baseline.
DEFAULT_IMPORTS: list[ImportSpec] = [
    {"module": "dspy"},
    {"from_module": "typing", "imported_names": ["Optional", "List", "Dict", "Any"]},
]

DEFAULT_SIGNATURES: list[SignatureSpec] = [
    {
        "name": "BasicSignature",
        "docstring": "Simple QA signature.",
        "inputs": [{"name": "question", "field_type": "str", "description": "Question"}],
        "outputs": [
            {"name": "answer", "field_type": "str", "description": "Answer", "is_input": False}
        ],
    }
]

DEFAULT_MODULES: list[ModuleSpec] = [
    {"name": "predictor", "module_type": "Predict", "signature_ref": "BasicSignature", "parameters": {}},
]


class DSPyProgramEvolver:
    """
    Evolver = (structure + compile) ⊕ (instruction optimization via GEPA).

    - Structure: typed specs → DSPyProgram (signatures, class, bindings elsewhere)
    - Compile: render to Python
    - Optimize (optional): call DSPy.GEPA; propagate improved instructions back
    """

    def __init__(
        self,
        import_library: list[ImportSpec] | None = None,
        signature_library: list[SignatureSpec] | None = None,
        module_library: list[ModuleSpec] | None = None,
        seed: int | None = None,
    ) -> None:
        self.import_library = import_library or DEFAULT_IMPORTS
        self.signature_library = signature_library or DEFAULT_SIGNATURES
        self.module_library = module_library or DEFAULT_MODULES
        self.rng: RNG = RandomRNG(seed)
        self._mutations: list[Mutation] = default_mutations(
            self.import_library, self.signature_library, self.module_library
        )

    # ---------- Construction ----------
    def seed_program(self) -> DSPyProgram:
        return DSPyProgram(
            imports=[DSPyImport(**d) for d in self.import_library],
            signatures=[
                DSPySignature(
                    name=s["name"],
                    docstring=s.get("docstring", s.get("description", "")),
                    inputs=[make_field(fd) for fd in s.get("inputs", [])],
                    outputs=[make_field(fd) for fd in s.get("outputs", [])],
                )
                for s in self.signature_library
            ],
            main_class=DSPyClass(
                name="BasicProgram",
                docstring="A basic DSPy program that answers questions.",
                methods=[
                    DSPyMethod(
                        name="forward",
                        parameters=[DSPyParameter("self"), DSPyParameter("question", "str")],
                        return_type="dspy.Prediction",
                        body=[],
                    )
                ],
            ),
        )

    # ---------- Evolution (structural) ----------
    def evolve(self, program: DSPyProgram, steps: int = 1) -> DSPyProgram:
        p = program
        for _ in range(max(0, steps)):
            applicable = [m for m in self._mutations if m.applicable(p)]
            if not applicable:
                logger.info("No applicable mutations.")
                break
            self.rng.choice(applicable).apply(p, self.rng)
        return p

    # ---------- Validation ----------
    def validate(self, program: DSPyProgram, prune_imports: bool = True) -> ValidationResult:
        res = validate_program(program)
        if not res.ok:
            return res
        if prune_imports:
            code = program.to_code()
            fixed = prune_unused_imports(code)
            if fixed != code:
                logger.info("Imports pruned in rendered code.")
        return res

    # ---------- Optimization (DSPy GEPA) ----------
    def optimize_with_gepa(
        self,
        program: DSPyProgram,
        trainset: list[Any],
        *,
        valset: list[Any] | None = None,
        metric: Callable[..., Any] | None = None,
        gepa_kwargs: dict[str, Any] | None = None,
    ) -> DSPyProgram:
        if not program.bound_modules:
            raise ValueError("optimize_with_gepa requires at least one bound module.")
        if metric is None:
            raise ValueError("optimize_with_gepa requires a metric callable.")

        try:
            import dspy
        except Exception as e:
            raise RuntimeError("DSPy is required to run GEPA optimization.") from e

        code = program.to_code()
        ns: dict[str, Any] = {"dspy": dspy}
        exec(code, ns, ns)
        student = ns.get(program.program_var)
        if student is None:
            raise RuntimeError(f"Program variable `{program.program_var}` not found after exec().")

        kw = dict(gepa_kwargs or {})
        if not kw.get("reflection_lm"):
            raise ValueError("gepa_kwargs must include a configured `reflection_lm` (dspy.LM).")
        gepa = dspy.GEPA(metric=metric, **kw)

        optimized = gepa.compile(student, trainset=trainset, valset=(valset or trainset))

        updated: dict[str, str] = {}
        for name, pred in optimized.named_predictors():
            try:
                instr = pred.signature.instructions
            except Exception:
                continue
            if isinstance(instr, str):
                updated[name] = instr

        if not updated:
            return program

        sig_by_name = {s.name: s for s in program.signatures}
        for bind in program.bound_modules:
            if bind.module_name in updated and bind.signature_name in sig_by_name:
                sig_by_name[bind.signature_name].docstring = updated[bind.module_name]
        return program
