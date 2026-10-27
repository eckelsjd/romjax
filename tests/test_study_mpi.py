"""Optional real MPI study launch; skipped when MPI is not installed."""
import importlib.util
import os
import shutil
import subprocess
import sys

import pytest
import yaml

pytest.importorskip('optuna')


@pytest.mark.skipif(importlib.util.find_spec('mpi4py') is None or shutil.which('mpiexec') is None,
                    reason='requires mpi4py and an MPI launcher')
def test_mpi_studies_borrow_pool_and_prune(tmp_path):
    base = tmp_path / 'base.yml'
    base.write_text('''!romx:Train
root: unused
loss: !!python/name:tests.study_helpers.loss
init_params: !pd:tests.study_helpers.Initial {}
optimizer: {name: optax.sgd, kwargs: {learning_rate: 0.1}}
termination: {max_steps: 4}
diagnostics: {show_progress: false}
''')
    for name in ('first', 'second'):
        (tmp_path / f'{name}.yml').write_text(f'''!romx:OptunaStudy
root: {tmp_path / name}
base: {base}
jax_platforms: cpu
n_trials: 2
pruner: {{name: tests.study_helpers.AlwaysPrune}}
metric: {{report_interval: 1}}
executor: {{kind: mpi, max_in_flight: 2, show_progress: false}}
''')
    composite = tmp_path / 'composite.yml'
    composite.write_text(f'''!romx:CompositeRoutine
executor: {{kind: serial, show_progress: false}}
routines: [{tmp_path / 'first.yml'}, {tmp_path / 'second.yml'}]
''')
    result = subprocess.run([shutil.which('mpiexec'), '-n', '3', sys.executable, '-c',
                             'from romjax.romx_cli import main; main()', 'run', str(composite)],
                            env={**os.environ, 'JAX_PLATFORMS': 'cpu'}, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    for name in ('first', 'second'):
        rows = yaml.safe_load((tmp_path / name / 'study_manifest.yml').read_text())['trials']
        assert len(rows) == 2 and all(row['state'] == 'PRUNED' for row in rows)
