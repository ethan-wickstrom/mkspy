from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Dict, Set

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider, ScopeProvider

from .model import DSPyProgram

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str]


def validate_program(program: DSPyProgram) -> ValidationResult:
    errors: List[str] = []
    if program.main_class is None:
        return ValidationResult(False, ["Program is missing a main class."])

    code = program.to_code()
    try:
        module = cst.parse_module(code)
    except cst.ParserSyntaxError as e:
        return ValidationResult(False, [f"LibCST parse error: {e}"])

    class StructureVisitor(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider,)

        def __init__(self, cls: str, var: str) -> None:
            self.cls = cls
            self.var = var
            self.has_dspy_import = False
            self.cls_ok = False
            self.assignment_ok = False

        def visit_Import(self, node: cst.Import) -> None:
            for alias in node.names:
                if isinstance(alias.name, cst.Name) and alias.name.value == "dspy":
                    self.has_dspy_import = True

        def visit_ClassDef(self, node: cst.ClassDef) -> None:
            if node.name.value != self.cls:
                return
            for base in node.bases:
                v = base.value
                if isinstance(v, cst.Attribute) and isinstance(v.value, cst.Name):
                    if v.value.value == "dspy" and v.attr.value == "Module":
                        self.cls_ok = True
                elif isinstance(v, cst.Name) and v.value == "dspy.Module":
                    self.cls_ok = True

        def visit_Assign(self, node: cst.Assign) -> None:
            if len(node.targets) != 1:
                return
            tgt = node.targets[0].target
            if isinstance(tgt, cst.Name) and tgt.value == program.program_var:
                if isinstance(node.value, cst.Call) and isinstance(node.value.func, cst.Name):
                    if node.value.func.value == self.cls:
                        self.assignment_ok = True

    w = MetadataWrapper(module)
    sv = StructureVisitor(program.main_class.name, program.program_var)
    w.visit(sv)

    if not sv.has_dspy_import:
        errors.append("Missing `import dspy`.")
    if not sv.cls_ok:
        errors.append(f"Main class `{program.main_class.name}` must inherit from `dspy.Module`.")
    if not sv.assignment_ok:
        errors.append(f"Missing `{program.program_var} = {program.main_class.name}()` assignment.")

    return ValidationResult(len(errors) == 0, errors)


# -------------------------
# Optional import pruning
# -------------------------

def prune_unused_imports(code: str) -> str:
    """
    Remove unused imports with a LibCST-based pass (best effort).
    """
    try:
        module = cst.parse_module(code)
    except cst.ParserSyntaxError:
        return code

    w = MetadataWrapper(module)
    ranges = w.resolve(PositionProvider)
    scopes = w.resolve(ScopeProvider)

    # Collect unused import names grouped by import node (Import/ImportFrom)
    from collections import defaultdict
    unused: Dict[cst.CSTNode, Set[str]] = defaultdict(set)

    # Convert maps to a per-node view
    all_scopes = set(scopes.values())
    for scope in all_scopes:
        for assignment in scope.assignments:
            node = assignment.node
            if isinstance(node, (cst.Import, cst.ImportFrom)):
                # If no references, mark as unused
                if len(assignment.references) == 0:
                    unused[node].add(assignment.name)

    class RemoveUnusedImportTransformer(cst.CSTTransformer):
        def leave_Import(
            self, original_node: cst.Import, updated_node: cst.Import
        ) -> cst.BaseSmallStatement:
            if original_node not in unused:
                return updated_node
            names_to_keep = []
            for alias in updated_node.names:
                name_value = alias.asname.name.value if alias.asname else alias.name.value
                if name_value not in unused[original_node]:
                    names_to_keep.append(alias.with_changes(comma=cst.MaybeSentinel.DEFAULT))
            if not names_to_keep:
                return cst.RemoveFromParent()
            return updated_node.with_changes(names=tuple(names_to_keep))

        def leave_ImportFrom(
            self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
        ) -> cst.BaseSmallStatement:
            if original_node not in unused:
                return updated_node
            # star imports cannot be partially pruned
            if isinstance(updated_node.names, cst.ImportStar):
                return updated_node
            kept = []
            for alias in updated_node.names or []:
                if not isinstance(alias, cst.ImportAlias):
                    kept.append(alias)
                    continue
                name_value = alias.asname.name.value if alias.asname else alias.name.value
                if name_value not in unused[original_node]:
                    kept.append(alias.with_changes(comma=cst.MaybeSentinel.DEFAULT))
            if not kept:
                return cst.RemoveFromParent()
            return updated_node.with_changes(names=tuple(kept))

    cleaned = w.module.visit(RemoveUnusedImportTransformer())
    return cleaned.code
