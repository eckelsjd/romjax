"""Measure indexed HDF5 storage costs separately from model computation.

Run: uv run python -m demo.benchmark_data_storage --samples 256 --width 256
The temporary synthetic dataset contains one solution and four outputs per input.
Timings include the operating system's filesystem cache; they are not cold-disk benchmarks.
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from romjax.data_gen import LoadImplicitModel
from romjax.storage import H5StorageConfig, H5Store, Record


def measure(root: Path, layout: str, capacity: int | None, samples: int, width: int) -> dict:
    """Measure writing, metadata discovery, and loading a fixed synthetic dataset."""
    records = []
    for index in range(samples):
        value = np.full(width, index, dtype=np.float32)
        records.append(Record(0, index, "input", {"inputs": {"x": value}, "solutions": {"y": value}}))
        for output in range(4):
            records.append(
                Record(0, index, "output", {"values": {"y": value + output}, "residuals": {"r": value}}, 1, output)
            )
    store = H5Store(root, "implicit", H5StorageConfig(layout=layout, samples_per_file=capacity))
    start = perf_counter()
    with store.locked():
        store.write(records)
    write_seconds = perf_counter() - start
    loader = LoadImplicitModel(cache_policy="none")
    start = perf_counter()
    refs = loader.discover_sample_refs(root)
    discover_seconds = perf_counter() - start
    start = perf_counter()
    loader.load_batch(refs)
    full_load_seconds = perf_counter() - start
    shuffled = np.random.default_rng(42).permutation(len(refs))
    start = perf_counter()
    for offset in range(0, len(refs), 32):
        loader.load_batch([refs[index] for index in shuffled[offset : offset + 32]])
    batch_load_seconds = perf_counter() - start
    loader.cache_policy = "epoch"
    loader.load_batch(refs)
    start = perf_counter()
    loader.load_batch(refs)
    cached_seconds = perf_counter() - start
    return {
        "layout": layout,
        "samples_per_file": capacity,
        "files": len(list(root.glob("*.h5"))),
        "references": len(refs),
        "write_seconds": write_seconds,
        "discover_seconds": discover_seconds,
        "full_load_seconds": full_load_seconds,
        "shuffled_batches_seconds": batch_load_seconds,
        "cached_epoch_seconds": cached_seconds,
    }


def main() -> None:
    """Compare individual-family files, grouped shards, and packed storage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()
    if args.samples < 1 or args.width < 1:
        parser.error("samples and width must be positive")
    with TemporaryDirectory(prefix="romjax-storage-benchmark-") as directory:
        for layout, capacity in [("grouped", 1), ("grouped", 64), ("packed", 64), ("packed", None)]:
            result = measure(Path(directory) / f"{layout}-{capacity}", layout, capacity, args.samples, args.width)
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
