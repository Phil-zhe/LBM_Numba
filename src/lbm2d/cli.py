from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import benchmark_cavity
from .cases.cavity import run_cavity
from .config import load_cavity_config
from .environment import inspect_environment


def build_parser():
    parser = argparse.ArgumentParser(prog="lbm2d")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect")
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--backend", required=True, choices=["numpy-cpu", "numba-cuda-mlir"])
    run.add_argument("--mode", default="full", choices=["smoke", "validation", "benchmark", "full"])
    run.add_argument("--results-root", default="results")
    validate = sub.add_parser("validate-cavity")
    validate.add_argument("--result-dir", required=True)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--sizes", nargs="+", type=int, default=[129, 257, 513])
    bench.add_argument("--timed-steps", type=int, default=2000)
    bench.add_argument("--repeats", type=int, default=5)
    bench.add_argument("--results-root", default="results")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.command == "inspect":
        print(json.dumps(inspect_environment(), indent=2, ensure_ascii=False))
    elif args.command == "run":
        result, summary = run_cavity(load_cavity_config(args.config), args.backend, args.mode,
                                     args.results_root, root / "reference")
        print(json.dumps({"result_dir": str(result), "summary": summary}, indent=2))
    elif args.command == "validate-cavity":
        metadata = json.loads((Path(args.result_dir) / "run_metadata.json").read_text(encoding="utf-8"))
        print(json.dumps(metadata, indent=2))
    elif args.command == "benchmark":
        result, rows = benchmark_cavity(args.sizes, args.timed_steps, args.repeats,
                                        results_root=args.results_root)
        print(json.dumps({"result_dir": str(result), "summary": rows}, indent=2))


if __name__ == "__main__":
    main()

