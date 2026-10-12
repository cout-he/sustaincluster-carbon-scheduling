# E1-5C workload pressure diagnostic method

This is an offline, read-only diagnostic. It does not alter the executor, policy,
task features, CI/weather, checkpoints, E1-5B selection rule or historical output.
It is not an RL sensor or a scheduler input. The command-line interface accepts
only `--split train` and `--split validation`; Test is rejected before inputs are
opened. Existing output directories are never overwritten.

## Time and resource units

All times use UTC. The arrival cohort is `[start,end)` on the 15-minute grid.
Every interval, including zero-arrival intervals, is included in arrival-rate
quantiles. A 48-hour cohort therefore has 192 arrival buckets. A task at `end`
does not enter the cohort, and Drain contributes zero additional arrivals.

The upstream task start and nominal finish are `start_time` and
`start_time + duration`. Completed-task SLA uses this **raw nominal finish**, not
the resource-release observation time. The single executor checks resource
release only at grid boundaries; resources/power are held for
`ceil(duration / 15 minutes) * 15 minutes`. For example, duration 31 minutes has
45 resource-holding minutes, but still uses a raw 31-minute finish for SLA.
No physical formula is changed by this diagnostic.

For each resource `r`, both workloads are retained:

* `raw_work_r = sum(demand_ir * raw_duration_i)`;
* `occupied_work_r = sum(demand_ir * rounded_holding_minutes_i)`.

CPU is measured in core-minute, GPU in GPU-minute, memory in
original-trace-memory-unit-minute. This stage does not infer bytes or GiB from
`avg_mem`. Instantaneous arriving demands have units core/GPU/memory-unit.
Interval-arrival pressure is occupied work divided by `capacity_r * 15 minutes`;
the separately named `_raw` pressure uses raw work. Neither pressure measures
actual resource utilization: arrivals may execute across later intervals.

The rolling 120-minute pressure uses the current bucket plus the seven previous
arrival buckets and divides by `capacity_r * 120 minutes`. Before cohort start
arrivals contribute zero; `history_observed_minutes` explicitly records the cold
start coverage. Drain continues to expire old arrivals from this history, without
adding tasks. This diagnostic normalization is frozen here and does not replace
RiskOnly's existing risk statistic or its cold-start decision rule.

## Deadline flexibility

The synthetic deadline remains `arrival + 1.5 * raw_duration` in this study.
Raw solo slack is `deadline - arrival - raw_duration`; the theoretical latest
start is `deadline - raw_duration`. The latest feasible **grid** start is the UTC
15-minute floor of that theoretical start. Usable grid slack is the latter minus
arrival. An arrival is urgent if `arrival + 15 minutes > latest_raw_start`, which
means it cannot wait a full step even in isolation. Equality permits waiting one
step. A deadline preceding the earliest raw finish is labeled solo infeasible.
Using `deadline - rounded_holding_duration` would incorrectly change upstream SLA
semantics and is deliberately avoided.

Task-level records retain duration, rounded holding, synthetic deadline, raw and
grid slack, raw/grid latest start, resource demand and urgency. These large
task-level outputs remain local. Summary quantiles use NumPy's linear population
quantile rule; empty task sets are explicitly represented by zero-valued
summaries plus count zero, and empty arrival buckets remain in interval quantiles.

## Joining actual EDF behavior

`align_edf` reads actual sealed interval, task-ledger and per-step policy records.
It audits task IDs and features, candidate-decision partitions, pending/running
ownership, resource capacity, and running-ID demand against recorded usage. It
reports capacity waiting per actual task turn, pending/running counts, resource
fractions, energy/carbon and the exact observed completion boundary.

Two distinct time markers are retained:

* `new_completed_misses_observed`: completed tasks whose raw finish exceeds their
  deadline, counted when the executor actually observes completion and releases
  their resources;
* `new_latest_start_misses_known`: first grid boundary at which a task is still
  unstarted and cannot meet its deadline even if immediately started in isolation.

The second marker is reconstructed offline from the final ledger and known task
features, not taken from a nonexistent logged prediction, and is not a second
deadline penalty. Running tasks that already started on time are excluded.
Events at the final observation boundary are saved in `terminal_boundary` rather
than inserted into an earlier power interval. Whole-cohort DMR uses final task
outcomes, not a sum of the two marker series.

Arrival pressure and later queue/miss peaks may co-occur with resource saturation
and restrictive slack; that correspondence is descriptive and cannot establish a
single causal factor. EDF's observed violations are not a theoretical optimum.

## Stress seed interpretation

At 2× every original task receives one same-time exact-feature copy. A stress
seed changes copy IDs and ranking, hence can affect stable EDF ties, but the task
arrival/resource/duration/deadline **aggregate shape is unchanged**. Different 2×
seeds are not independent workload samples. At 1.25×/1.5× the seed selects a
hash-ranked prefix of originals to duplicate; a shared seed gives nested task
sets. Such fractional seed variation still reuses the same original trace and
does not create independent real arrival processes. Same-time replication can
amplify bursts and is synthetic pressure, not a measured larger workload trace.

## API and reproduction

`summarize(workload, capacity, history_minutes=120)` returns `PressureAnalysis`
containing plain JSON-serializable `summary`, `arrivals` and `task_details`.
`align_edf(analysis, steps, ledger, policy_steps)` returns `series`,
`terminal_boundary`, `summary`, `audit` and event definitions.

```bash
python -m experiments.e1.workload_pressure \
  --bundle research_notes/E1/E1-5B/precheck-001/load-2/input \
  --split train --output research_notes/E1/E1-5C/train-pressure-example-001
python -m pytest tests/e1/test_workload_pressure.py -q -p no:cacheprovider
```

Behavior tests were written before implementation. Logs `pressure-tests-red-001`
and `red-002` preserve missing-module and missing-consistency-check failures. The
final targeted run `pressure-tests-green-003.txt` contains **20 passed in 2.59 s**
under the existing SustainCluster WSL Python environment. Tests cover interval
endpoints and zero buckets, raw/rounded work and exact release grid, slack/urgency
boundary, past-only rolling prefix invariance, 2× seed aggregate invariance,
fractional same-trace replication, immutable originals, empty data, bad capacity,
EDF capacity events and resource-ID consistency, separate miss timestamps,
terminal boundary and Drain, wrong task/event records, and Test CLI rejection.
