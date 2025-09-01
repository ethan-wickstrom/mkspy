from __future__ import annotations

import logging
from dataclasses import dataclass, field

import libcst as cst
import libcst.matchers as m
from libcst.metadata import MetadataWrapper, PositionProvider

logger = logging.getLogger(__name__)

# A lightweight guide; your code may need bespoke translations.
AST_TO_LIBCST_MAP: dict[str, str] = {
    "ast.parse": "cst.parse_module",
    "ast.NodeVisitor": "cst.CSTVisitor",
    "ast.NodeTransformer": "cst.CSTTransformer",
    "ast.Module": "cst.Module",
    "ast.FunctionDef": "cst.FunctionDef",
    "ast.ClassDef": "cst.ClassDef",
    "ast.Assign": "cst.Assign",
    "ast.Return": "cst.Return",
    "ast.If": "cst.If",
    "ast.For": "cst.For",
    "ast.Name": "cst.Name",
    "ast.Call": "cst.Call",
    "ast.Attribute": "cst.Attribute",
    "ast.arguments": "cst.Parameters",  # note: shape differs
}


# ---------- scanning ----------

@dataclass
class AstUsageRecord:
    path: str
    import_ast: bool = False
    from_ast_imports: list[str] = field(default_factory=list)
    attr_uses: list[str] = field(default_factory=list)
    lines: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, str | bool | list[str] | list[int]]:
        return {
            "path": self.path,
            "import_ast": self.import_ast,
            "from_ast_imports": sorted(set(self.from_ast_imports)),
            "attr_uses": sorted(set(self.attr_uses)),
            "lines": sorted(set(self.lines)),
        }


class _PreScan(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self) -> None:
        self.ast_aliases: set[str] = {"ast"}  # will include alias names, e.g., {"ast", "pyast"}

    def visit_Import(self, node: cst.Import) -> None:
        for alias in node.names:
            if isinstance(alias.name, cst.Name) and alias.name.value == "ast":
                if alias.asname and isinstance(alias.asname.name, cst.Name):
                    self.ast_aliases.add(alias.asname.name.value)

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if isinstance(node.module, cst.Name) and node.module.value == "ast":
            # no alias tracking for from-imports here; handled in transformer
            pass


class _Scan(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, path: str) -> None:
        self.record = AstUsageRecord(path)

    def visit_Import(self, node: cst.Import) -> None:
        for alias in node.names:
            if isinstance(alias.name, cst.Name) and alias.name.value == "ast":
                self.record.import_ast = True
                pos = self.get_metadata(PositionProvider, node, default=None)
                if pos is not None:
                    self.record.lines.append(pos.start.line)

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if isinstance(node.module, cst.Name) and node.module.value == "ast":
            names_list: list[cst.ImportAlias] = []
            if node.names is None:
                names_list = []
            elif isinstance(node.names, cst.ImportStar):
                names_list = []
            else:
                names_list = list(node.names)
            for alias in names_list:
                if isinstance(alias.name, cst.Name):
                    self.record.from_ast_imports.append(alias.name.value)
            pos = self.get_metadata(PositionProvider, node, default=None)
            if pos is not None:
                self.record.lines.append(pos.start.line)

    def visit_Attribute(self, node: cst.Attribute) -> None:
        if isinstance(node.value, cst.Name) and node.value.value == "ast":
            self.record.attr_uses.append(node.attr.value)
            pos = self.get_metadata(PositionProvider, node, default=None)
            if pos is not None:
                self.record.lines.append(pos.start.line)


def scan_code_for_ast_usage(code: str, path: str = "<memory>") -> AstUsageRecord:
    """Return a minimal, structured summary of builtin `ast` usage."""
    try:
        mod = cst.parse_module(code)
    except cst.ParserSyntaxError:
        return AstUsageRecord(path)
    w = MetadataWrapper(mod)
    s = _Scan(path)
    w.visit(s)
    return s.record


# ---------- transform ----------

class AstToLibCSTCodemod(m.MatcherDecoratableTransformer):
    """
    Best-effort transform:
      * import ast[/as X] → import libcst as cst
      * from ast import NodeVisitor/NodeTransformer (with alias) → from libcst import CSTVisitor/CSTTransformer
      * ast.parse → cst.parse_module
      * class Foo(ast.NodeVisitor) → class Foo(cst.CSTVisitor)  (also for alias or from-imports)
    """

    def __init__(self, ast_aliases: set[str] | None = None) -> None:
        super().__init__()
        self.ast_aliases: set[str] = (ast_aliases or {"ast"})

    # -- imports --

    def leave_Import(
        self, original_node: cst.Import, updated_node: cst.Import
    ) -> cst.Import | cst.RemovalSentinel:
        new_names = []
        changed = False
        for alias in updated_node.names:
            if isinstance(alias.name, cst.Name) and alias.name.value in self.ast_aliases:
                # Replace whole alias with libcst as cst
                changed = True
                new_names.append(cst.ImportAlias(name=cst.Name("libcst"), asname=cst.AsName(cst.Name("cst"))))
            else:
                new_names.append(alias)
        return updated_node.with_changes(names=tuple(new_names)) if changed else updated_node

    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.ImportFrom | cst.FlattenSentinel[cst.BaseSmallStatement] | cst.RemovalSentinel:
        if not isinstance(updated_node.module, cst.Name) or updated_node.module.value != "ast":
            return updated_node

        # map specific names to libcst equivalents; preserve aliases
        to_map = {"NodeVisitor": "CSTVisitor", "NodeTransformer": "CSTTransformer"}
        keep: list[cst.ImportAlias] = []
        add_for_libcst: list[cst.ImportAlias] = []
        names_list: list[cst.ImportAlias] = []
        if updated_node.names is None:
            names_list = []
        elif isinstance(updated_node.names, cst.ImportStar):
            names_list = []
        else:
            names_list = list(updated_node.names)

        for alias in names_list:
            if isinstance(alias.name, cst.Name) and alias.name.value in to_map:
                target = to_map[alias.name.value]
                if alias.asname:
                    add_for_libcst.append(cst.ImportAlias(name=cst.Name(target), asname=alias.asname))
                else:
                    add_for_libcst.append(cst.ImportAlias(name=cst.Name(target)))
            else:
                keep.append(alias)

        small_stmts: list[cst.BaseSmallStatement] = []
        if keep:
            small_stmts.append(
                updated_node.with_changes(module=cst.Name("libcst"), names=tuple(keep))
            )
        if add_for_libcst:
            small_stmts.append(
                cst.ImportFrom(module=cst.Name("libcst"), names=tuple(add_for_libcst))
            )
        if small_stmts:
            return cst.FlattenSentinel(tuple(small_stmts))
        return cst.RemoveFromParent()

    # -- attribute replacements --

    @m.leave(m.Attribute(value=m.Name(), attr=m.Name("parse")))
    def replace_parse(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> cst.BaseExpression:
        # Replace X.parse → cst.parse_module when X is an ast alias
        base = updated_node.value
        if isinstance(base, cst.Name) and base.value in self.ast_aliases:
            return cst.Attribute(value=cst.Name("cst"), attr=cst.Name("parse_module"))
        return updated_node

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        # Replace ast.NodeVisitor/Transformer in bases, including alias bases from "from ast import …"
        new_bases = []
        changed = False
        for arg in updated_node.bases:
            v = arg.value
            # ast alias base
            if isinstance(v, cst.Attribute) and isinstance(v.value, cst.Name) and v.value.value in self.ast_aliases:
                if v.attr.value == "NodeVisitor":
                    changed = True
                    new_bases.append(cst.Arg(value=cst.Attribute(cst.Name("cst"), cst.Name("CSTVisitor"))))
                    continue
                if v.attr.value == "NodeTransformer":
                    changed = True
                    new_bases.append(cst.Arg(value=cst.Attribute(cst.Name("cst"), cst.Name("CSTTransformer"))))
                    continue
            # direct names from "from ast import NodeVisitor as NV"
            if isinstance(v, cst.Name) and v.value in {"NodeVisitor", "NodeTransformer"}:
                changed = True
                mapped = "CSTVisitor" if v.value == "NodeVisitor" else "CSTTransformer"
                new_bases.append(cst.Arg(value=cst.Attribute(cst.Name("cst"), cst.Name(mapped))))
                continue
            new_bases.append(arg)
        return updated_node.with_changes(bases=tuple(new_bases)) if changed else updated_node


def codemod_ast_to_libcst(code: str) -> str:
    """Rewrite builtin `ast` usage to LibCST equivalents."""
    try:
        mod = cst.parse_module(code)
    except cst.ParserSyntaxError:
        return code
    w = MetadataWrapper(mod)
    pre = _PreScan()
    w.visit(pre)
    transformed = w.module.visit(AstToLibCSTCodemod(pre.ast_aliases))
    return transformed.code


def codemod_needed(code: str) -> bool:
    """Checks if codemod will change the code."""
    try:
        return codemod_ast_to_libcst(code) != code
    except Exception:
        # Be safe: on parse errors or unexpected issues, report no change
        return False
