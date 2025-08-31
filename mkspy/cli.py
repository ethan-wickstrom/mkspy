from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .evolver import DSPyProgramEvolver
from .validation import prune_unused_imports
from .codemod import scan_code_for_ast_usage, codemod_ast_to_libcst
from .author import get_program_author, optimize_program_author, default_author_trainset


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


def cmd_author(args: argparse.Namespace) -> None:
    # Read requirements from file/stdin/arg
    if args.in_file:
        req = Path(args.in_file).read_text(encoding="utf-8")
    elif args.requirements is not None:
        req = args.requirements
    else:
        req = sys.stdin.read()

    # Configure LM if provided
    lm = None
    if args.model:
        try:
            import dspy  # type: ignore
        except Exception:
            print("dspy must be installed to use --model", file=sys.stderr)
            raise
        lm = dspy.LM(args.model, temperature=args.temperature, max_tokens=args.max_tokens)
        dspy.configure(lm=lm)

    author = get_program_author(lm=lm)
    pred = author(requirements=req)
    print(pred.source, end="")


def _load_requirements_lines(path: str) -> list[str]:
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


def cmd_optimize_author(args: argparse.Namespace) -> None:
    try:
        import dspy  # type: ignore
    except Exception:
        print("dspy must be installed to optimize the author", file=sys.stderr)
        raise

    # Gen LM
    gen_lm = None
    if args.model:
        gen_lm = dspy.LM(args.model, temperature=args.temperature, max_tokens=args.max_tokens)
        dspy.configure(lm=gen_lm)

    # Reflection LM
    if not args.reflection_model:
        print("--reflection-model is required", file=sys.stderr)
        raise SystemExit(2)
    reflection_lm = dspy.LM(args.reflection_model, temperature=1.0, max_tokens=32000)

    # Datasets
    if args.train:
        tr_reqs = _load_requirements_lines(args.train)
        trainset = [dspy.Example(requirements=r).with_inputs("requirements") for r in tr_reqs]
    else:
        trainset = default_author_trainset()
    valset = None
    if args.val:
        va_reqs = _load_requirements_lines(args.val)
        valset = [dspy.Example(requirements=r).with_inputs("requirements") for r in va_reqs]

    # Budget
    gepa_kwargs = dict(max_metric_calls=args.max_metric_calls, num_threads=args.num_threads)

    optimized = optimize_program_author(
        trainset=trainset,
        valset=valset,
        gen_model=gen_lm,
        reflection_lm=reflection_lm,
        gepa_kwargs=gepa_kwargs,
    )

    # Emit once if requested
    if args.emit:
        req_text = Path(args.emit).read_text(encoding="utf-8") if Path(args.emit).exists() else args.emit
        out = optimized(requirements=req_text)
        print(out.source, end="")
    else:
        # Print the current optimized instruction for visibility
        try:
            name, pred = next(iter(optimized.named_predictors()))
            print(pred.signature.instructions)
        except Exception:
            print("Optimization finished.")


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

    s = sub.add_parser("author", help="Use an LLM agent to write a DSPy program from requirements")
    s.add_argument("requirements", nargs="?", help="Inline requirements text. If omitted, read stdin.")
    s.add_argument("--in", dest="in_file", help="Read requirements from file path")
    s.add_argument("--model", help="Model name for dspy.LM (e.g., openai/gpt-4o-mini)")
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--max-tokens", dest="max_tokens", type=int, default=2048)
    s.set_defaults(func=cmd_author)

    s = sub.add_parser("optimize-author", help="Run GEPA to optimize the authoring agent")
    s.add_argument("--train", help="File with training requirements (one per line)")
    s.add_argument("--val", help="File with validation requirements (one per line)")
    s.add_argument("--model", help="Generation model for the author (e.g., openai/gpt-4o-mini)")
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--max-tokens", dest="max_tokens", type=int, default=2048)
    s.add_argument("--reflection-model", required=True, help="Reflection model for GEPA (e.g., gpt-5)")
    s.add_argument("--max-metric-calls", type=int, default=100)
    s.add_argument("--num-threads", type=int, default=4)
    s.add_argument("--emit", help="After optimization, emit code for this requirements string or file path")
    s.set_defaults(func=cmd_optimize_author)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
