"""Generate, inspect, extend and load a tiny implicit dataset.

Run: uv run python -m demo.data_storage_demo --root /tmp/romjax-storage-demo
MPI: mpiexec -n 4 uv run python -m demo.data_storage_demo --executor mpi --root /tmp/romjax-mpi-demo
"""

import argparse
from pathlib import Path

import h5py

import romjax
from demo.data_storage_model import ToyImplicit
from romjax.data_gen import DataGeneration, DataLoader, GenImplicitModel
from romjax.graph import FunctionGraph
from romjax.storage import H5StorageConfig, H5Store


def main() -> None:
    """Run a reusable demonstration without training or a PDE solve."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("demo/outputs/data-storage"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--map-batch-size", type=int, default=0)
    parser.add_argument("--samples-per-file", type=int, default=2, help="0 uses one dataset file")
    parser.add_argument("--layout", choices=["packed", "grouped"], default="packed")
    parser.add_argument("--executor", choices=["serial", "thread", "process", "mpi"], default="serial")
    args = parser.parse_args()
    graph = FunctionGraph(edges={"toy": ToyImplicit(source="inputs", target="outputs")})
    generator = GenImplicitModel(
        input_samples=5,
        outputs_per_input=2,
        input_seed=1,
        output_seed=2,
        batch_size=args.batch_size,
        map_batch_size=args.map_batch_size,
    )
    execution = {"kind": args.executor, "show_progress": False}
    if args.executor in {"thread", "process"}:
        execution["max_workers"] = 2
    generation = DataGeneration(
        root=args.root,
        graph=graph,
        executor=execution,
        storage=H5StorageConfig(layout=args.layout, samples_per_file=args.samples_per_file or None),
        datasets={"toy": generator},
    )
    generation.run()
    store = H5Store(args.root / "toy")
    refs = store.discover()
    print(f"Generated {len(refs)} training references in {len(list(store.root.glob('*.h5')))} files")
    with h5py.File(refs[0].file) as file:
        print("Input index:", file["index/inputs"].fields(["seed", "input_index", "row"])[:])
        if args.layout == "packed":
            print("Input field:", file["seeds/seed_1/inputs/x"][:])
        else:
            print("Input field:", file["seeds/seed_1/inputs/row_0/x"][:])
    generation.run()  # Reuse: no numerical work or shard rewrite.
    generator.outputs_per_input = 3
    generation.run()
    print(f"After extending outputs: {len(H5Store(store.root).discover())} references")
    loader = DataLoader(
        root=args.root,
        datasets={
            "toy": {
                "kind": "implicit",
                "batch_size": 4,
                "max_samples": 6,
                "shuffle_seed": 7,
                "max_epochs": 2,
                "cache_policy": "epoch",
            }
        },
    )
    batches = list(loader)
    print("Two shuffled epochs:", [len(batch["toy"]) for batch in batches])
    print("First sample fields:", sorted(batches[0]["toy"][0]))
    print("Cached samples:", len(loader.datasets["toy"]._cache_entries))


if __name__ == "__main__":
    with romjax.mpi_context() as controller:
        if controller:
            main()
