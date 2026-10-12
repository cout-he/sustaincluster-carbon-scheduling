# E1-5A fixed-action numerical equivalence

The frozen-source formal validation completed successfully on 2026-10-12,
03:22:36–03:26:29 UTC. All **14/14 runs have exact numerical equality** with
their sealed E1-3 reference, and every arrived task completed. **12/14 runs are
also byte-identical across all eight result files.** The two exceptions are
fixed action 0 (EDF) at loads 1.5 and 2.0: only `intervals.csv` differs because
the historical idle extension writes integer zero resource use as `0`, whereas
the continuous stepwise inventory writes `0.0`. Every numeric CSV cell remains
exactly equal as a `Decimal`; its columns, row count and timestamps match.
An independent read of both CSV pairs identified exactly 12 changed zero cells
in each of `used_cores`, `used_gpus` and `used_mem` per load, all `0` versus
`0.0`; no other CSV cell differed.

The other seven files require byte-identical SHA256 in every run: task ledger,
task events, physical summary, policy ledger, policy events, policy intervals,
and policy summary. Consequently task order, start/finish/release times,
capacity/policy attribution, all per-interval IT/HVAC/total power and energy,
carbon, and all summary values match. No numerical tolerance was applied, and
neither engine nor historical serializers were changed to hide the formatting
difference.

The complete machine-readable evidence is in
[numerical_equivalence.csv](numerical_equivalence.csv),
`provenance.json` (private local archival evidence),
`archive_integrity.json` (private local archival evidence)
and each new run folder's `regression.json`. The formal status is `complete` in
`run_status.json` (private local archival evidence).

## All 14 formal comparisons

Both factors use the original accounting endpoint **2023-07-03 13:45 UTC**, with
227 intervals, including persistent physical-model idle accounting. Load 1.5
contains 14,914 tasks; load 2.0 contains 19,886. There are no unfinished or
censored tasks. DMR below is completed deadline misses divided by all arrived
tasks, displayed as a percentage to six decimal places. The CSV retains the
counts and full-precision carbon values. These are reproduced historical
metrics, not evidence of a trained agent's performance.

| Load | Execution | Policy / fixed action | Misses | DMR (%) | Carbon (kgCO2e) | Eight-file bytes | Numerical fields |
|---|---|---|---:|---:|---:|---|---|
| 1.5 | Direct | FIFO | 42 | 0.281615 | 16462.71909586536 | Identical | Exact |
| 1.5 | Direct | EDF | 5 | 0.033526 | 16463.054483729164 | Identical | Exact |
| 1.5 | Direct | Carbon-FIFO | 410 | 2.749095 | 16314.062348612595 | Identical | Exact |
| 1.5 | Direct | RiskOnly | 94 | 0.630280 | 16278.946405428067 | Identical | Exact |
| 1.5 | Gymnasium | EDF / 0 | 5 | 0.033526 | 16463.054483729164 | CSV zero formatting only | Exact |
| 1.5 | Gymnasium | EDF+Carbon / 1 | 212 | 1.421483 | 16319.346405308204 | Identical | Exact |
| 1.5 | Gymnasium | RiskOnly / 2 | 94 | 0.630280 | 16278.946405428067 | Identical | Exact |
| 2.0 | Direct | FIFO | 1335 | 6.713266 | 20172.550796326497 | Identical | Exact |
| 2.0 | Direct | EDF | 517 | 2.599819 | 20182.946093411243 | Identical | Exact |
| 2.0 | Direct | Carbon-FIFO | 2476 | 12.450971 | 19951.82345898601 | Identical | Exact |
| 2.0 | Direct | RiskOnly | 795 | 3.997787 | 19887.08887869618 | Identical | Exact |
| 2.0 | Gymnasium | EDF / 0 | 517 | 2.599819 | 20182.946093411243 | CSV zero formatting only | Exact |
| 2.0 | Gymnasium | EDF+Carbon / 1 | 1509 | 7.588253 | 19962.882345706614 | Identical | Exact |
| 2.0 | Gymnasium | RiskOnly / 2 | 795 | 3.997787 | 19887.08887869618 | Identical | Exact |

## Resource work and physical components

Resource-hours match the sealed common-phase values explicitly and are equal
across all seven reruns at each load. Their units are allocated cores, GPUs and
memory capacity multiplied by interval hours.

| Load | Core-hours | GPU-hours | Memory-hours |
|---|---:|---:|---:|
| 1.5 | 724255.3264160807 | 27365.757945322726 | 1122848.251079979 |
| 2.0 | 960198.6053357549 | 36052.69389654991 | 1501409.6126800892 |

Fixed modes also reproduce all physical interval values and the following
aggregate components exactly:

| Load | Fixed action | IT energy (kWh) | HVAC energy (kWh) | Total energy (kWh) |
|---|---:|---:|---:|---:|
| 1.5 | 0 | 37290.43108692245 | 50965.60175804931 | 88256.03284497176 |
| 1.5 | 1 | 37312.88822605729 | 50964.350870753195 | 88277.23909681049 |
| 1.5 | 2 | 37319.52058004543 | 50815.90191277264 | 88135.42249281806 |
| 2.0 | 0 | 40184.799016347766 | 68341.47156296081 | 108526.27057930858 |
| 2.0 | 1 | 40217.20754891454 | 68360.79713895038 | 108578.00468786492 |
| 2.0 | 2 | 40215.84891269006 | 68057.03721722883 | 108272.8861299189 |

## Integrity and random smoke episode

Before any numerical run, the independent importer verified every sealed
scientific file and source snapshot from all four E1-3 levels and both E1-4A
levels, including strategies not rerun here. Current original source bytes must
match those historical manifests except the authorized shared engine refactor;
only the eight E1-4A additions and five declared E1-5A Python files are allowed
as appended sources. Full reference declarations, input hashes, provenance
links, DC/config/selection equality, and E1-4A legacy-control results were
checked. Old strict replay verifiers and old provenance remain unchanged.

The run used the exact frozen source manifest SHA256
`2df222b9ea7bfbc1520ff70a02dbe7414940ff0d07167e64f8e77217cca7c18e`
covering 155 Python files. The E1-5A result provenance records the same hash.
It records 144 scientific output SHA256 values and saves the current source
snapshot inside the new result directory.

The load-1.5 random episode with seed **20261012** completed all **14,914 tasks**
with zero pending/running tasks and true termination, after 215 intervals at
2023-07-03 10:45 UTC. It selected actions 0/1/2 respectively 67/73/75 times.
Its per-step physical carbon and complete reward terms are retained in
`trajectory.csv` (private local archival evidence).
Its policy-summary label describes the last selected mode; accumulated counts
cover the whole episode. This episode stops at its own drain boundary and has
a different accounting horizon from the fixed-mode comparisons. It is only a
complete-cohort operational smoke check; no training or performance claim was
made.

The historical threshold **137.64999999999998** was imported without changing
its bits. The separately audited April calibration threshold **161.075** is
reserved for future Train/Validation/Test use and was not substituted into this
historical equivalence validation.
