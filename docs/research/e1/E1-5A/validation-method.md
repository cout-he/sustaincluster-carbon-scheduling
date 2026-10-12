# E1-5A independent validation method

Scope: numerical equivalence of the shared executor and fixed Gymnasium modes,
plus one seeded complete-cohort random-action smoke episode. No training,
policy optimization, or policy-performance claim is part of this validation.

## Historical import integrity

`experiments/e1/rl_validation.py` independently imports the sealed E1-3 and
E1-4A campaigns. It keeps the existing replay verifiers unchanged. It checks
all four E1-3 load levels and both E1-4A levels, including policies that are not
rerun in this validation:

- Exact required scientific inventory and every recorded scientific SHA256.
- Runtime-file integrity, complete campaign/child provenance links and frozen
  workload/environment/DC/config/parameter identities.
- Every original source snapshot's inventory and SHA256 against its own
  historical source manifest, and that manifest's canonical hash.
- Every current original source has the archived SHA256, except the explicitly
  authorized `experiments/e1/engine.py` refactor. Additional Python files are
  restricted to the eight E1-4A modules and five declared E1-5A modules/tests.
- E1-4A references the exact E1-3 provenance, full scientific manifest, frozen
  inputs and parameters. Its legacy control scientific files remain identical.

This is an independent cross-version equivalence importer. Original strict
replay still requires the archived source environment and is not weakened.

## Numerical execution and comparison

For each load factor 1.5 and 2.0, rerun FIFO, EDF, Carbon-FIFO and RiskOnly through
`LocalExecutor`; then run fixed Gymnasium actions 0/1/2 corresponding to EDF,
EDF+Carbon and RiskOnly. The episode retains one persistent executor and one
physical model. Fixed modes run through the sealed original seven-policy
accounting endpoint, including idle intervals. Their step count is derived from
the endpoint; the historical window contains 227 intervals, not a guessed 224.

For each of these 14 runs, reuse `run._write_result` and require byte-identical
SHA256 for ledger, task events, summary, policy ledger, policy events, policy
intervals and policy summary. The interval CSV requires identical column order,
row count and timestamps and exact `Decimal` equality for every numeric cell.
This pins every per-interval
IT/HVAC/total power, energy and carbon value, all task decisions and attribution,
and all summary fields. CPU/GPU/MEM resource-hours are also computed with the
same `math.fsum` integration and compared explicitly with archived common-phase
resource-hours. No floating tolerance, metadata exclusion or altered expected
file is used to make these eight result files pass.

Historical `extend_idle` supplies integer zero resources, while continuous
`advance` observes float zeros from the inventory. As a result, fixed modes can
serialize an idle resource value as `0.0` where the archived CSV records `0`.
The KnownPower miniature additionally exposes `2.0` versus `2` idle power from
the same arithmetic type propagation. These exactly equal decimal values are
reported as `serialization_only_differences`; they are not claimed to be
byte-identical. Seven non-CSV files still require byte equality. Reports keep
`all_files_byte_identical` and `all_numerical_fields_exact` separate. A difference
as small as `100` versus `100.000000000000000001` is rejected.

The load-1.5 random episode uses NumPy seed 20261012 and the same online executor
with observed RiskOnly history across mode changes. It must reach genuine
finite-cohort completion with no pending/running tasks; its actions, reward
components available in the trajectory, result files and seed are evidence of
environment operation only. The mixed episode's policy summary name describes
its last selected mode while counts cover the whole episode.

Legacy threshold bits are imported unchanged as 137.64999999999998 for
equivalence. The independent April calibration threshold 161.075 belongs to
future Train/Validation/Test use described in `data_split.json`; it is not
substituted into historical numerical validation.

## Evidence status before source freeze

- `validation-red-001.txt`: 14 expected failures before driver implementation.
- `validation-resource-red-001.txt`: expected missing resource-hours evidence
  failure before adding explicit resource-hour reporting.
- `validation-green-002.txt`: 10 miniature integrity/comparison tests passed.
- `validation-identity-red-001.txt`: a fully self-hashed false frozen-reference
  declaration reproduced the import gap before its complete identity check was
  added. The rejection passed in the subsequent miniature run.
- `validation-green-003.txt`: 15 passed, one miniature integration failure
  exposed the integer/float idle CSV representation described above.
- `validation-csv-red-001.txt`: expected failure before the exact Decimal CSV
  comparator was added.
- `validation-green-004.txt`: all 15 fast miniature tests passed, including
  three fixed modes, seeded completion, explicit resource-hours, exact decimal
  comparison and rejection of an arbitrarily small numeric CSV change. Two
  archive-generation tests are covered by the final suite; their source-copy
  fixture uses unchanged real policy/power modules rather than copying the
  entire repository repeatedly. Real history uses the full source inventory.
- `validation-mini-green-001.txt`: reused the valid miniature archives without
  regenerating them; all 14 numerical comparisons passed, with 10 fully
  byte-identical results and four CSV-only serialization differences. The seeded
  random finite cohort completed. Full miniature output is retained in
  `validation-mini-recheck-001`.
- `historical-import-readonly-001.txt`: real sealed E1-3 four-level and E1-4A
  two-level archives passed independent read-only import integrity checks.

The root agent froze 155 Python files at source manifest SHA256
`2df222b9ea7bfbc1520ff70a02dbe7414940ff0d07167e64f8e77217cca7c18e`,
recorded in `source-freeze.json`, before starting formal numerical runs.
The full formal results are saved in this E1-5A directory's `results-001`, with a
small `numerical_equivalence.csv`, complete result files, trajectories,
`archive_integrity.json`, current source snapshots and new provenance. Previous
archives and provenance are read only.

Formal validation completed at 03:26:29 UTC: 14/14 exact numerical comparisons,
12/14 eight-file byte-identical comparisons, and two action-0 idle CSV zero
format differences. The seeded load-1.5 random episode completed all 14,914
tasks in 215 intervals. Detailed DMR, carbon, physical components, resource work
and classification are recorded in `E1-5A-fixed-action-equivalence.md`.

Reproduction command, from the upstream workspace with its established
`sustaincluster` Python environment:

```text
python -m experiments.e1.rl_validation --output research_notes/E1/E1-5A/results-001
```
