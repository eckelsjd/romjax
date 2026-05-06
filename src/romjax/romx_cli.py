"""ROM CLI for building reduced-order models with romjax."""

import argparse
import shutil
import sys
from pathlib import Path

import romjax
from romjax.typing import RoutineError


def build_parser() -> argparse.ArgumentParser:
    """Build the rom CLI argument parser."""
    parser = argparse.ArgumentParser(description="romjax reduced-order model workflow")
    subparsers = parser.add_subparsers(dest="command", required = True)
    
    gen = subparsers.add_parser("run", help="Run a romjax routine from yaml config file.")
    gen.add_argument("config", help=f"Path to config file. Provided routines: {romjax.routines.__all__}")

    return parser


def cli(argv: list[str] | None = None) -> int:
    """Run the rom CLI.

    :param argv: CLI arguments excluding the interpreter name
    :return: process exit code
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        try:
            if not Path(args.config).exists():
                raise RoutineError(f"Config file '{args.config}' not found")
            
            routine = romjax.load(args.config)

            if not hasattr(routine, "run") or not callable(getattr(routine, "run")):
                raise RoutineError(f"Top-level object loaded from '{args.config}' must implement `run()`.")

            if hasattr(routine, "root"):
                shutil.copy(Path(args.config), routine.root / Path(args.config).name)
            
            return routine.run()
            
        except RoutineError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    
    else:
        parser.error(f"Unhandled command: {args.command}")
        return 2


def main():
    """Console-script entrypoint for the rom CLI."""
    raise SystemExit(cli())


__all__ = ["cli"]
