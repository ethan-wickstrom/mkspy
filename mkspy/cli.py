from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .evolver import DSPyProgramEvolver
from .validation import prune_unused_imports
from .codemod import scan_code_for_ast_usage, codemod_ast_to_libcst


def _iter_py_files(root: Path):
    for p in root.rglob("*.py"):
        if any(part in {".git", ".venv", "venv", "__pycache__"} for part in p.parts):
            continue
        yield p


def cmd_seed(args: argparse.Namespace) -> None:
    evo = DSPyProgramEvolver(seed=args.seed)
    p = evo.seed_program()
    code = p.to_code()
    print(code, end="")


def cmd_evolve(args: argparse.Namespace) -> None:
    evo = DSPyProgramEvolver(seed=args.seed)
    p = evo.seed_program()
    p = evo.evolve(p, steps=args.steps)
    res = evo.validate(p)
    code = p.to_code()
    if args.prune:
        code = prune_unused_imports(code)
    print(code, end="")
    if not res.ok:
        print("\n# Validation errors:", file=sys.stderr)
        for e in res.errors:
            print(f"# - {e}", file=sys.stderr)


def cmd_scan(args: argparse.Namespace) -> None:
    for path in _iter_py_files(Path(args.root)):
        rec = scan_code_for_ast_usage(path.read_text(encoding="utf-8"), str(path))
        d = rec.to_dict()
        if d["import_ast"] or d["from_ast_imports"] or d["attr_uses"]:
            print(d)


def cmd_codemod(args: argparse.Namespace) -> None:
    for path in _iter_py_files(Path(args.root)):
        src = path.read_text(encoding="utf-8")
        dst = codemod_ast_to_libcst(src)
        if dst != src:
            path.write_text(dst, encoding="utf-8")
            print(f"Rewrote {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("dspy-evolver")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("seed", help="Emit a seed DSPy program")
    s.add_argument("--seed", type=int, default=None)
    s.set_defaults(func=cmd_seed)

    s = sub.add_parser("evolve", help="Evolve a seed program and emit code")
    s.add_argument("--steps", type=int, default=3)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--prune", action="store_true")
    s.set_defaults(func=cmd_evolve)

    s = sub.add_parser("scan", help="Scan a tree for built-in `ast` usage")
    s.add_argument("root", help="Directory to scan")
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("codemod", help="Rewrite built-in `ast` usage to LibCST")
    s.add_argument("root", help="Directory to transform in-place")
    s.set_defaults(func=cmd_codemod)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
