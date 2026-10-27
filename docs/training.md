# Train diagnostics and host transfers

Use `Train.fit()` to train from Python. It returns a `TrainingResult`, whose
`params`, `step`, `metrics`, and `stop_reason` describe the final state. A routine's
`run()` continues to return an exit code.

```python
result = train.fit()
params = result.params
```

Intervals count completed optimizer updates, including after checkpoint restore.
`host_interval` controls when Train can collect current device values on the host.
A device-dependent activity runs only when its own interval and `host_interval`
are both due. Missed ticks are skipped. `host_interval: null` disables periodic
host collection; finalization still collects final metrics and saves artifacts.

```yaml
!romx:Train
# loss, init_params, optimizer, test, and termination omitted
host_interval: 10
diagnostics:
  test_interval: 20
  report_interval: 10
  report_metrics: [loss, test_score]
  checkpoint_interval: 50
  log_interval: 10
  plot_interval: 100
```

For example, with `host_interval: 8` and `report_interval: 10`, periodic reports
occur at steps 40, 80, and so on. Align the intervals deliberately, particularly for
Optuna pruning. An observer is a Python runtime callback:

```python
from romjax.train import TrainingDecision

def observe(report):
    print(report.step, report.metrics)
    return TrainingDecision.CONTINUE

result = train.fit(observer=observe)
```

`report_interval: null` disables periodic reports. An attached observer receives
one final notification with `final=True`; its decision is ignored. A STOP decision
on a periodic report stops training and saves the current state. Reporting
`test_score` requires `Train.test`. A test tick or an enabled report requesting
`test_score` requests validation; when both occur together it is computed once.
Validation and reporting both obey the host gate. Fresh values are never replaced
with an older score.

Checkpoint checks and parameter callbacks also obey the host gate. Checkpoint
policies receive the current metrics already collected at the checkpoint step; a
policy needing `test_score` requires a test or report request at that step.
Logging and plotting can run on their own intervals using cached host metrics and
histories. Logged metric steps identify when cached data was collected. Loss and
gradient tolerance checks and adaptive loss balancing use host-enabled steps.

The gate controls Train's built-in metric collection, callbacks, and checkpoint
submission. User-provided loss functions, data loaders, initial placement, and
checkpoint restoration can still transfer data. Waiting for device computation is
also distinct from transferring device values to the host.

Move a former top-level `checkpoint_interval` into `diagnostics` and replace
`train()` with `train.fit().params`. Reporting configuration is now part of the
Train YAML; `fit()` accepts only the runtime observer.
