"""Summarize JAX tracing, compilation, and cache-miss diagnostics from a log file."""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

_TRACING_RE = re.compile(
    r"Finished (?P<stage>tracing \+ transforming) (?P<name>.+?) for \S+ in (?P<seconds>[0-9.]+) sec"
)
_LOWERING_RE = re.compile(
    r"Finished (?P<stage>jaxpr to MLIR module conversion) (?P<name>\S+) in (?P<seconds>[0-9.]+) sec"
)
_XLA_RE = re.compile(r"Finished (?P<stage>XLA compilation) of (?P<name>\S+) in (?P<seconds>[0-9.]+) sec")
_COMPILING_RE = re.compile(r"Compiling (?P<name>\S+)")
_CACHE_MISS_RE = re.compile(r"TRACING CACHE MISS .*\((?P<context>[^)]+)\):")
_FUNCTION_ID_RE = re.compile(r"(?P<name>[A-Za-z_][\w.]*) id=\d+ defined at (?P<location>[^ \n]+:\d+)")
_HLO_MODULE_RE = re.compile(r"HLO module (?P<name>jit_[A-Za-z0-9_.$-]+)")


def _read_lines(path: Path) -> list[str]:
    """Read a log file with replacement for malformed bytes."""
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def summarize(path: Path, top: int) -> str:
    """Return a human-readable summary of the JAX compile log."""
    stage_seconds: defaultdict[str, float] = defaultdict(float)
    stage_counts: Counter[str] = Counter()
    compile_counts: Counter[str] = Counter()
    cache_contexts: Counter[str] = Counter()
    function_ids: Counter[str] = Counter()
    hlo_modules: Counter[str] = Counter()

    lines = _read_lines(path)
    for line in lines:
        duration_match = _TRACING_RE.search(line) or _LOWERING_RE.search(line) or _XLA_RE.search(line)
        if duration_match:
            stage = duration_match.group("stage").strip()
            name = duration_match.group("name")
            key = f"{stage} {name}".strip()
            stage_seconds[key] += float(duration_match.group("seconds"))
            stage_counts[key] += 1
        if match := _COMPILING_RE.search(line):
            compile_counts[match.group("name")] += 1
        if match := _CACHE_MISS_RE.search(line):
            cache_contexts[match.group("context") or "<unknown>"] += 1
        if match := _FUNCTION_ID_RE.search(line):
            function_ids[f"{match.group('name')} @ {match.group('location')}"] += 1
        if match := _HLO_MODULE_RE.search(line):
            hlo_modules[match.group("name")] += 1

    total_compile_time = sum(seconds for key, seconds in stage_seconds.items() if key.startswith("XLA compilation"))
    total_trace_time = sum(seconds for key, seconds in stage_seconds.items() if key.startswith("tracing"))
    total_lower_time = sum(seconds for key, seconds in stage_seconds.items() if key.startswith("jaxpr to MLIR"))

    out = [
        f"Log: {path}",
        f"Lines: {len(lines)}",
        "",
        "Totals:",
        f"  tracing: {total_trace_time:.3f} s",
        f"  lowering: {total_lower_time:.3f} s",
        f"  XLA compilation: {total_compile_time:.3f} s",
        f"  compile invocations: {sum(compile_counts.values())}",
        f"  tracing cache misses: {sum(cache_contexts.values())}",
        "",
        f"Top {top} stage durations:",
    ]
    for key, seconds in sorted(stage_seconds.items(), key=lambda item: item[1], reverse=True)[:top]:
        out.append(f"  {seconds:8.3f} s  {stage_counts[key]:4d}x  {key}")

    out.append("")
    out.append(f"Top {top} compiled functions:")
    for name, count in compile_counts.most_common(top):
        out.append(f"  {count:4d}x  {name}")

    out.append("")
    out.append(f"Top {top} cache-miss contexts:")
    for context, count in cache_contexts.most_common(top):
        out.append(f"  {count:4d}x  {context}")

    out.append("")
    out.append(f"Top {top} repeated function ids:")
    for location, count in function_ids.most_common(top):
        out.append(f"  {count:4d}x  {location}")

    out.append("")
    out.append(f"Top {top} eager/tiny HLO modules:")
    for name, count in hlo_modules.most_common(top):
        out.append(f"  {count:4d}x  {name}")

    return "\n".join(out)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    """Print the log summary."""
    args = parse_args()
    print(summarize(args.log, args.top))


if __name__ == "__main__":
    main()
