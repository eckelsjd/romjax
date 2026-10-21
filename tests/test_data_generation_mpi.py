"""End-to-end MPI data generation and nested executor ownership."""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from romjax.graph import Edge
from romjax.model import SourceSampleable
from romjax.utils import load_h5

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec('mpi4py') is None or shutil.which('mpiexec') is None,
    reason='requires the mpi extra and an MPI launcher',
)


class RankSource(Edge, SourceSampleable):
    """Record worker placement along with deterministic samples."""

    def forward(self, x):
        return x

    def backward(self, x):
        return x

    def sample_source(self, key):
        rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', os.environ.get('PMI_RANK', '0')))
        return {'x': jax.random.uniform(key), 'rank': jnp.asarray(rank)}


def launch(config: Path, python_context: bool = False) -> subprocess.CompletedProcess:
    script = ('import sys, romjax;\n'
              'with romjax.mpi_context() as controller:\n'
              '    if controller: sys.exit(romjax.load(sys.argv[1]).run())') if python_context else (
                  'from romjax.romx_cli import main; main()')
    arguments = [str(config)] if python_context else ['run', str(config)]
    environment = {**os.environ, 'JAX_PLATFORMS': 'cpu'}
    return subprocess.run([shutil.which('mpiexec'), '-n', '4', sys.executable, '-c', script, *arguments],
                          capture_output=True, text=True, timeout=120, check=False, env=environment)


def generation_yaml(root: Path, executor: str) -> str:
    executor = ('{kind: mpi, show_progress: false}' if executor == 'mpi'
                else executor[:-1] + ', show_progress: false}')
    return (f'!romx:DataGeneration\nroot: {root}\nexecutor: {executor}\n'
            'graph: !romx:FunctionGraph\n  edges:\n'
            '    sample: !pd:tests.test_data_generation_mpi.RankSource\n'
            '      source: noise\n      target: sample\n'
            'datasets:\n  sample: {samples: 20, seed: 7, batch_size: 2}\n')


@pytest.mark.parametrize('python_context', [False, True])
def test_mpi_generation_distributes_batches(tmp_path, python_context):
    config = tmp_path / 'generate.yml'
    root = tmp_path / 'data'
    config.write_text(generation_yaml(root, 'mpi'), encoding='utf-8')
    result = launch(config, python_context)
    assert result.returncode == 0, result.stdout + result.stderr
    files = sorted(root.rglob('source.h5'))
    assert len(files) == 20
    ranks = {int(load_h5({}, path)['rank']) for path in files}
    assert len(ranks) > 1
    assert 0 not in ranks
    for path in files:
        index = int(path.parent.name.removeprefix('sample_'))
        expected = jax.random.uniform(jax.random.fold_in(jax.random.key(7), index), dtype=jnp.float32)
        assert float(load_h5({}, path)['x']) == pytest.approx(float(expected))


def test_serial_composite_reuses_pool_for_generation_then_grid(tmp_path):
    generation = tmp_path / 'generate.yml'
    generation.write_text(generation_yaml(tmp_path / 'data', 'mpi'), encoding='utf-8')
    base = tmp_path / 'base.yml'
    base.write_text('!pd:tests.test_grid_search_mpi.MpiEnvRoutine\nroot: original\nresolved: 0\n', encoding='utf-8')
    grid = tmp_path / 'grid.yml'
    grid.write_text(f'!romx:GridSearch\nroot: {tmp_path / "grid"}\nbase: __parent__/base.yml\n'
                    'override: [{path: [resolved], cases: [1, 2]}]\nexecutor: mpi\n', encoding='utf-8')
    composite = tmp_path / 'composite.yml'
    composite.write_text('!romx:CompositeRoutine\nexecutor: {kind: serial, show_progress: false}\n'
                         'routines: [__parent__/generate.yml, __parent__/grid.yml]\n', encoding='utf-8')
    result = launch(composite)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list((tmp_path / 'data').rglob('source.h5'))) == 20
    assert (tmp_path / 'grid' / 'grid_search_manifest.yml').exists()


def test_mpi_composite_allows_local_pools_and_rejects_nested_mpi(tmp_path):
    children = []
    for kind in ['thread', 'process']:
        child = tmp_path / f'{kind}.yml'
        child.write_text(generation_yaml(tmp_path / kind, f'{{kind: {kind}, max_workers: 2}}'), encoding='utf-8')
        children.append(str(child))
    composite = tmp_path / 'composite.yml'
    composite.write_text(
        f'!romx:CompositeRoutine\nexecutor: mpi\njax_enable_x64: true\nroutines: {children}\n',
        encoding='utf-8')
    result = launch(composite)
    assert result.returncode == 0, result.stdout + result.stderr
    assert all(len(list((tmp_path / kind).rglob('source.h5'))) == 20 for kind in ['thread', 'process'])
    for kind in ['thread', 'process']:
        sample = load_h5({}, next((tmp_path / kind).rglob('source.h5')))
        assert sample['x'].dtype == np.float64
    nested = tmp_path / 'nested.yml'
    nested.write_text(generation_yaml(tmp_path / 'nested', 'mpi'), encoding='utf-8')
    composite.write_text(f'!romx:CompositeRoutine\nexecutor: mpi\nroutines: [{nested}]\n', encoding='utf-8')
    result = launch(composite)
    assert result.returncode != 0
    assert 'Nested MPI execution' in result.stderr


def test_mpi_implicit_generation_matches_serial_and_reuses_outputs(tmp_path):
    import romjax

    config = tmp_path / 'implicit.yml'
    root = tmp_path / 'parallel'
    config.write_text(
        f'!romx:DataGeneration\nroot: {root}\nexecutor: {{kind: mpi, show_progress: false}}\njax_platforms: cpu\n'
        'graph: !romx:FunctionGraph\n  edges:\n'
        '    sample: !pd:tests.test_data_gen._ConditionedSampleableEdge\n'
        '      source: inputs\n      target: outputs\n'
        'datasets:\n  sample: {input_samples: 5, outputs_per_input: 3, input_seed: 1, output_seed: 2, '
        'batch_size: 2}\n', encoding='utf-8')
    reference = romjax.load(config)
    reference.root = tmp_path / 'serial'
    reference.executor = romjax.ExecutorConfig(kind='serial', show_progress=False)
    assert reference.run() == 0
    result = launch(config)
    assert result.returncode == 0, result.stdout + result.stderr
    paths = sorted(path.relative_to(reference.root) for path in reference.root.rglob('*.h5'))
    assert len(list(root.rglob('output.h5'))) == 15
    assert paths == sorted(path.relative_to(root) for path in root.rglob('*.h5'))
    for path in paths:
        before = load_h5({}, reference.root / path)
        after = load_h5({}, root / path)
        for left, right in zip(jax.tree.leaves(before), jax.tree.leaves(after)):
            np.testing.assert_allclose(left, right)
    timestamps = {path: (root / path).stat().st_mtime_ns for path in paths}
    result = launch(config)
    assert result.returncode == 0, result.stdout + result.stderr
    assert timestamps == {path: (root / path).stat().st_mtime_ns for path in paths}


def test_mpi_poisson_graph_generation_matches_serial(tmp_path):
    import romjax

    fixture = Path(__file__).with_name('fixtures_poisson_graph.yml')
    (tmp_path / 'graph.yml').write_text(
        fixture.read_text().replace('[64, 64]', '[8, 8]'), encoding='utf-8')
    root = tmp_path / 'parallel'
    config = tmp_path / 'poisson.yml'
    config.write_text(
        f'!romx:DataGeneration\nroot: {root}\n'
        'executor: {kind: mpi, show_progress: false}\njax_platforms: cpu\n'
        'graph: __parent__/graph.yml\ndatasets:\n'
        '  Fh: {input_samples: 2, outputs_per_input: 0, input_seed: 1, output_seed: 2}\n', encoding='utf-8')
    reference = romjax.load(config)
    reference.root = tmp_path / 'serial'
    reference.executor = romjax.ExecutorConfig(kind='serial', show_progress=False)
    with jax.default_device(jax.devices('cpu')[0]):
        assert reference.run() == 0
    result = launch(config)
    assert result.returncode == 0, result.stdout + result.stderr
    paths = sorted(path.relative_to(reference.root) for path in reference.root.rglob('*.h5'))
    assert paths
    assert paths == sorted(path.relative_to(root) for path in root.rglob('*.h5'))
    for path in paths:
        before, after = load_h5({}, reference.root / path), load_h5({}, root / path)
        assert jax.tree.structure(before) == jax.tree.structure(after)
        for left, right in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True):
            np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-6)
