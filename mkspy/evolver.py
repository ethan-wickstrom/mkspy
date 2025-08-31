from __future__ import annotations

import logging
from .model import DSPyProgram, DSPyClass, DSPyMethod, DSPyParameter, DSPySignature, DSPyImport
from .mutations import default_mutations, RandomRNG, RNG, Mutation
from .validation import validate_program, prune_unused_imports, ValidationResult
from .types import ImportSpec, SignatureSpec, ModuleSpec, make_field

logger = logging.getLogger(__name__)


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
    High-level orchestrator:
    - Create a seed program
    - Apply safe mutations
    - Validate with LibCST
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

    # --------- construction ---------

    def seed_program(self) -> DSPyProgram:
        p = DSPyProgram(
            imports=[DSPyImport(**d) for d in self.import_library],
            signatures=[
                DSPySignature(
                    name=s["name"],
                    docstring=s.get("docstring", ""),
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
        return p

    # --------- evolution ---------

    def evolve(self, program: DSPyProgram, steps: int = 1) -> DSPyProgram:
        p = program
        for _ in range(max(0, steps)):
            applicable = [m for m in self._mutations if m.applicable(p)]
            if not applicable:
                logger.info("No applicable mutations.")
                break
            mut = self.rng.choice(applicable)
            mut.apply(p, self.rng)
        return p

    # --------- validation ---------

    def validate(self, program: DSPyProgram, prune_imports: bool = True) -> ValidationResult:
        res = validate_program(program)
        if not res.ok:
            return res
        if prune_imports:
            # pass the program through a best-effort import pruner
            code = program.to_code()
            fixed = prune_unused_imports(code)
            if fixed != code:
                # Note: We do not re-hydrate the structured Program from code here;
                # callers should persist the fixed code.
                logger.info("Imports pruned in rendered code.")
        return res
