# Generated data storage

Source and implicit generation use indexed HDF5 files. Existing nested sample-directory datasets are not supported: regenerate them into a new directory. Ordinary `save_h5`/`load_h5` artifact utilities are unchanged.

## Three independent batch sizes

```yaml
!romx:DataGeneration
root: data
executor: {kind: process, max_workers: 2, max_in_flight: 4}
storage: {layout: packed, samples_per_file: 256}
graph: graph.yml
datasets:
  Fh:
    input_samples: 1024
    outputs_per_input: 8
    input_seed: 2
    output_seed: 3
    batch_size: 64
    map_batch_size: 0
```

`batch_size` is the number of source samples or complete implicit input families in one worker task/result buffer. `map_batch_size` controls JIT-compiled `jax.lax.map` inside that task: 0 uses full vmap-like computation, 1 computes sequentially, and positive values vectorize that many entries with a partial tail. Input and per-input output axes are mapped separately. `samples_per_file` controls file capacity and never numerical execution. A generator's explicit storage configuration replaces the inherited routine configuration.

Serial execution runs one task at a time; it can still vectorize within that task. For sequential computation choose both `executor: serial` and `map_batch_size: 1`. For large samples use small task buffers. One implicit family includes *all* its requested outputs, so `batch_size: 1` still retains that family's outputs. `throw: false` uses host-side per-record error isolation with the same numerical kernels; arbitrary Python exceptions cannot be isolated to individual lanes inside a JAX transform.

## Files and fields

Finite capacities produce `seed_2_part_000000.h5`, `seed_2_part_000001.h5`, etc. `samples_per_file: null` uses `dataset.h5` for all seeds. Files contain a version, dataset kind, layout, capacity and a component-status legend in their attributes.

Packed implicit layout:

```text
/index/inputs                  # structured table: seed, input_index, row, component states
/index/outputs                 # also output_seed, output_index
/index/failures                # JSON identity/message/traceback records
/seeds/seed_2/
    inputs/forcing/beta        # [N, ...]
    solutions/phi              # [N, ...]
    solution_residuals/r       # [N, ...]
    outputs/seed_3/
        conditions/noise      # [M, ...]
        values/phi            # [M, ...]
        residuals/r           # [M, ...]
```

Indices use physical rows in their file's input/output tables. Field tables are separated by seed and may have unused rows; index component states determine whether a row contains valid data. Output rows carry parent identities rather than requiring equal outputs per input. Missing/None components do not acquire fabricated field values. Component groups record their tree schema, preserving empty dictionaries and None leaves.

Source storage uses `/index/inputs` and `/seeds/seed_i/values/<fields>`. The grouped layout uses `row_<physical-row>` subgroups beneath each component, allowing different shapes/trees per sample. Both layouts use the same flat indices for discovery. Packed storage requires matching schemas, shapes and dtypes within each component/seed table; use grouped storage for heterogeneous data.

Manual inspection:

```python
import h5py
with h5py.File('data/Fh/seed_2_part_000000.h5') as f:
    index = f['index/inputs'][:]
    entry = index[(index['seed'] == 2) & (index['input_index'] == 17)][0]
    beta = f['seeds/seed_2/inputs/forcing/beta'][entry['row']]
    print(f.attrs['status_legend'])
```

Numerical arrays are appendable and initially uncompressed. The default targets one complete sample per chunk for 10 KiB–1 MiB samples, combines tiny samples toward 64 KiB, and tiles larger samples. Chunk shape and file capacity are separate. Chunk alternatives should be benchmarked for full-epoch versus shuffled batch reads.

## Reproducibility and subsequent runs

An input is `(input_seed, input_index)`. An output additionally has `(output_seed, output_index)`. Physical rows and task completion order do not define identity.

```python
input_key = jax.random.fold_in(jax.random.key(input_seed), input_index)
# mix_output_seed=True (default)
output_key = jax.random.fold_in(jax.random.fold_in(input_key, output_seed), output_index)
# mix_output_seed=False
output_key = jax.random.fold_in(jax.random.key(output_seed), output_index)
```

Conditions and outputs use the same output key as before. Numerical grouping, shard capacity and executor placement do not alter keys.

| Subsequent request | Behavior |
| --- | --- |
| Same identities, reuse | Keep complete components; no numerical writes if nothing is missing |
| More inputs, same settings | Fill the last eligible file, then create another |
| New input seed | Create files for that seed; a single-file dataset adds another seed group |
| Different finite capacity/layout | New inputs start new files; existing records keep their recorded locations/settings |
| More outputs or new output seed | Append records inside the existing parent input's file |
| Overwrite | Replace requested identities in place, without duplicate rows |
| Change between single-file/sharded | Explicitly repack to a new destination |

`reuse` repairs missing requested components. Previously skipped solve/evaluate work can be enabled later. Sampler-returned None differs from intentionally skipped operations. `overwrite` replaces requested components and clears requested omitted values. Changed input/solution values invalidate dependent results; unrequested stale outputs remain excluded until regenerated. `error` fails when the dataset destination already exists.

Status legend: pending=0, complete=1, skipped=2, None=3, failed=4, stale=5. Data is flushed before completion is recorded. An interrupted write can leave pending work, which reuse repairs; commits are not batch-wide HDF5 transactions.

Repartition without regenerating:

```python
from romjax import RepackData
RepackData(source='data/Fh', destination='data/Fh-packed',
           storage={'layout': 'packed', 'samples_per_file': None}).run()
```

The destination must not exist or overlap the source. Repacking operates only on the new format. Stop generation before repacking/loading the same dataset.

## Executor ownership and failures

Workers compute one task batch at a time and return host arrays. Only the controller reads reuse data, allocates identities, and writes target files. Reuse context travels in task payloads. Graph/compression artifacts may still be read on workers. Process/MPI serialization retains imported model classes and declaring YAML paths; compiled functions and HDF5 handles are never serialized.

The existing executor handles pools, bounded submission, progress, cancellation and MPI ownership. Result callbacks commit in completion order and do not retain the whole run. `max_in_flight` bounds outstanding batches; memory still depends on task/family sizes. MPI results carry numerical arrays back to the controller, so transport and controller write throughput are explicit costs.

A `.romjax-generation.lock` prevents two controllers from generating the same resolved dataset concurrently. Abrupt termination can leave a lock: remove it manually only after confirming its recorded process is no longer using the dataset. No concurrent reader/writer or multiwriter mode is provided.

With `throw: false`, failures are recorded per input/output and successful neighbors remain available. With `throw: true`, valid results are committed, further submission stops, running work drains, and generation raises. Infrastructure/write failures propagate. JAX-mapped failures are replayed per record to identify the failed identity; replay assumes pure numerical sampling as required by the library.

## Discovery, batches, and cache

Discovery bulk-reads flat index tables, not individual groups or numerical datasets. References expose `.kind`, `.seed`, `.input_index`, `.output_seed`, `.output_index`, `.input_id`, and `.identity`. Skip callbacks now receive these typed references:

```yaml
# In Python: LoadImplicitModel(skip_input=lambda ref: ref.input_index % 2 == 1)
```

Selection retains the existing lexical seed/sample ordering, solution-before-output ordering, fixed selected pool, caps, deterministic epoch shuffling, and resume behavior. Solution references still count toward `max_outputs_per_input`.

Batch loading groups requests by file/table, reads shared inputs once, uses slices or sorted bounded row selections, then restores the requested order. Grouped layout retains per-record numerical access costs; packed layout reads shared field arrays. List output remains the default. `stack_batch: true` requires compatible trees and shapes (solutions and conditioned outputs may differ). Train, GraphLoss, normalization and compression continue consuming the same sample dictionaries.

`cache_policy: none` loads only the current batch. `cache_policy: epoch` retains every selected sample when encountered, with optional `cache_storage: host | device`. Later epochs read no numerical data. LRU, `cache_max_items`, and `cache_max_bytes` are removed and rejected.

Discovery checks shard names, modification times and sizes. When generation has changed a dataset between loader runs, discovery rebuilds its metadata and clears that dataset's cached samples. Batch reads perform no filesystem version checks; stop generation before starting an iterator.

## Demo and checks

```sh
uv run python -m demo.data_storage_demo --root /tmp/romjax-storage-demo
uv run python -m demo.data_storage_demo --root /tmp/romjax-process-demo --executor process --map-batch-size 1
uv run python -m demo.benchmark_data_storage --samples 256 --width 256
mpiexec -n 4 uv run python -m demo.data_storage_demo --root /tmp/romjax-mpi-demo --executor mpi
```

The importable toy model solves `y=2x`, stores sampled perturbation conditions and residuals, extends output counts, and loads two shuffled cached epochs. Process/MPI entrypoints are guarded and enter the shared MPI context.

GenDataBenchmark uses the same lax.map kernels, task sizes and map sizes, synchronizing and discarding numerical results on workers. Its compute-only timings exclude numerical-array transport and persistence. Schema 4 includes compute grouping and invalidates older timings; storage settings do not affect numerical identity. End-to-end generation/loading should be measured separately without wall-time assertions in unit tests.

The storage benchmark separately measures writes, discovery, full loading, shuffled batches and a cached epoch. It uses temporary synthetic data and includes the operating system's filesystem cache. One local run with 64 inputs, four outputs/input and 256 float32 values/field produced:

| Layout | Families/file | Files | Discovery | Full loading | Shuffled batches |
| --- | --- | --- | --- | --- | --- |
| Grouped | 1 | 64 | 43 ms | 83 ms | 143 ms |
| Grouped | 64 | 1 | 2.9 ms | 63 ms | 85 ms |
| Packed | 64 | 1 | 2.8 ms | 3.1 ms | 19 ms |
| Packed | Entire dataset | 1 | 2.7 ms | 2.9 ms | 19 ms |

These measurements illustrate both file-open and per-record dataset-access costs; they are not a prediction for PDE data or network storage. The grouped one-family case already combines fields into one file, so it is not a measurement of the removed per-field backend. The access-count regression checks the mechanism directly: discovering eight inputs/four outputs each reads only two index tables; loading all outputs reads the parent input field and output field once each, without loading unused solutions or residuals.
