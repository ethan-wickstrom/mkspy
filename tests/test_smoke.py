from mkspy.evolver import DSPyProgramEvolver
from mkspy.validation import validate_program
from mkspy.codemod import scan_code_for_ast_usage, codemod_ast_to_libcst


def test_seed_and_validate():
    evo = DSPyProgramEvolver(seed=1)
    p = evo.seed_program()
    res = validate_program(p)
    assert res.ok, res.errors
    code = p.to_code()
    # program line present
    assert "program = BasicProgram()" in code


def test_evolve_roundtrip():
    evo = DSPyProgramEvolver(seed=42)
    p = evo.seed_program()
    p = evo.evolve(p, steps=3)
    res = evo.validate(p)
    assert res.ok, res.errors


def test_codemod_simple():
    src = """
import ast as pyast
from ast import NodeVisitor as NV

class V(pyast.NodeVisitor, NV):
    pass

def parse(s):
    return pyast.parse(s)
"""
    rec = scan_code_for_ast_usage(src, "<mem>")
    assert rec.import_ast
    dst = codemod_ast_to_libcst(src)
    assert "import libcst as cst" in dst
    assert ".parse_module(" in dst
    assert "CSTVisitor" in dst
