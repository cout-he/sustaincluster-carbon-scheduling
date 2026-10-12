# E1-5C resource competition evidence: Train versus Validation

This read-only analysis uses the six sealed cohorts listed in
`diagnostics-001/input-manifest.json` and their actual EDF logs. Train 1×/1.5×/2×
and Validation 2× are unchanged E1-5B history. Validation 1×/1.5× are E1-5C
exploratory diagnostics, not a replacement for registered E1-5B model selection.
No Test data, checkpoint, policy or physical formula was accessed or changed.

## Main finding

The 2× multiplier does not imply similar pressure in the two windows. Validation
has 57.13% more tasks and 81.29% more rounded GPU work in the same 48-hour arrival
period; its mean GPU demand per task is also 44.66% larger. Its GPU workload
arrives at much higher local peaks and sustained 120-minute pressure. In the
actual EDF logs, Validation capacity waiting is predominantly GPU-only, whereas
Train's largest backlog and the majority of its capacity waiting are CPU-only.
Shorter Validation tasks also have less absolute deadline flexibility under the
unchanged 1.5 SLA multiplier. These factors jointly accompany the DMR difference;
the analysis does not establish one unique cause or theoretical unavoidable DMR.

| 2× arrival-cohort measure | Train | Validation |
|---|---:|---:|
| UTC arrival window | May 1 05:00–May 3 05:00 | July 8 05:00–July 10 05:00 |
| Arrived tasks | 11,084 | 17,416 |
| Mean tasks per 15 minutes | 57.729 | 90.708 |
| Maximum tasks per 15 minutes | 558 | 536 |
| Rounded GPU work, GPU-minute | 1,504,918.407 | 2,728,208.176 |
| Raw GPU work, GPU-minute | 1,375,180.442 | 2,452,770.436 |
| Rounded GPU arrival work / 48-hour capacity | 0.522541 | 0.947295 |
| Rounded CPU arrival work / 48-hour capacity | 0.494904 | 0.343085 |
| Rounded MEM arrival work / 48-hour capacity | 0.324025 | 0.351391 |
| Mean GPU work per 15 minutes, GPU-minute | 7,838.117 | 14,209.418 |
| P95 GPU work per 15 minutes, GPU-minute | 29,372.989 | 44,507.838 |
| Maximum GPU work per 15 minutes, GPU-minute | 57,443.500 | 152,339.227 |
| P95 trailing 120-minute GPU pressure | 1.210087 | 1.558624 |
| Maximum trailing 120-minute GPU pressure | 1.610018 | 3.621394 |
| Arrival buckets with 120-minute GPU pressure > 1 | 17 / 192 | 61 / 192 |
| Mean raw duration, minutes | 103.096 | 80.676 |
| Median raw solo slack, minutes | 37.433 | 28.488 |
| Median usable grid slack, minutes | 30 | 15 |
| Urgent at arrival: cannot safely wait 15 minutes alone | 19.091% | 30.064% |
| EDF completed misses / tasks | 779 / 11,084 | 5,301 / 17,416 |
| EDF DMR | 7.028% | 30.438% |

The resource-work figures above use held-resource minutes; corresponding raw
work is retained in every pressure summary. Both windows retain relative solo
slack approximately 0.5, with nanosecond timestamp rounding noise below 1.2e-9.
Thus the SLA multiplier has not become tighter in Validation. Absolute flexibility
does shrink with shorter duration: median usable grid slack falls from two steps
to one, and the urgent share rises by about 10.97 percentage points.

Higher task-count peaks are **not** supported: Train's maximum task count is
larger. GPU resource-work peaks are larger in Validation, showing why task count
alone is inadequate. CPU rounded work is lower in Validation, so the shift is not
a uniform increase in every resource. A cohort average below one also does not
prove feasible deadlines, because local intervals and nonpreemptive holding
matter. The independent interval necessary-condition analysis is required for
any claim about unavoidable misses.

## Actual capacity-wait masks

For each actual `capacity_waited` event, the diagnostic looks up that task's
immutable CPU/GPU/MEM demand and compares it with `available_before` at the
candidate's turn. The mask lists every strictly positive demand-minus-availability
dimension. The following categories are mutually exclusive; repeated events for
the same task are not independent tasks. Retesting masks with an absolute 1e-9
comparison tolerance changes **zero** events in all six cohorts.

| Cohort | EDF misses | Capacity wait events | CPU only | GPU only | MEM only | CPU+GPU | Other multi-resource |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train 1× | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Train 1.5× | 153 | 1,536 | 1,250 | 286 | 0 | 0 | 0 |
| Train 2× | 779 | 5,101 | 4,028 | 1,073 | 0 | 0 | 0 |
| Validation 1× | 403 | 2,156 | 0 | 2,156 | 0 | 0 | 0 |
| Validation 1.5× | 2,112 | 17,441 | 0 | 17,441 | 0 | 0 | 0 |
| Validation 2× | 5,301 | 80,784 | 1,289 | 79,411 | 0 | 84 | 0 |

Train 2× waiting is 78.965% CPU-only and 21.035% GPU-only. Validation 2× waiting
is 98.300% GPU-only, 1.596% CPU-only and 0.104% CPU+GPU. No wait has a memory
deficit in these runs; this is evidence about these realized EDF trajectories,
not a claim that memory can never bottleneck another window.

The number of unique tasks with capacity waiting is 1,512 in Train 2× and 6,555
in Validation 2×. All 779/5,301 EDF-missed tasks have a prior capacity-wait event,
and EDF has zero policy-induced deferrals. The observed EDF misses therefore
cannot be assigned to carbon-induced active deferral. They still reflect EDF
ordering, backfill and nonpreemptive allocation; this association does **not**
mean all those misses are capacity-unavoidable under every possible scheduler.
Mask event counts cannot be subtracted from policy violations or summed as causal
miss counts.

## Time evidence selected by one objective rule

For every cohort, select the earliest UTC maximum of each of three quantities:
trailing 120-minute GPU pressure (over the 192 arrival buckets), final pending
queue size, and first-known solo-latest-start losses (over the full common
observation window). This identical rule produces all 18 rows in
`pressure-timing-evidence.csv`, including zero-pressure/wait cases; it does not
search for a peak favorable to one method. Completion misses and first-known
latest-start losses remain distinct diagnostics, not two separate violation
counts. Terminal-boundary events remain outside earlier energy intervals.

| 2× cohort and selected peak | UTC time | GPU 120-minute pressure | Pending | CPU use | GPU use | Capacity mask context |
|---|---|---:|---:|---:|---:|---|
| Train: GPU pressure maximum | May 1 16:00 | 1.610018 | 28 | 27.380% | 99.990% | 28 GPU-only waits |
| Train: pending maximum | May 1 10:45 | 0.346367 | 434 | 99.992% | 24.562% | 434 CPU-only waits |
| Train: new known-loss maximum, 140 tasks | May 1 11:15 | 0.280282 | 304 | 99.995% | 22.941% | 304 CPU-only waits |
| Validation: GPU pressure maximum | July 8 19:30 | 3.621394 | 790 | 34.060% | 99.9999% | 790 GPU-only waits |
| Validation: pending maximum | July 9 00:45 | 1.364934 | 1,777 | 3.645% | 99.9998% | 1,777 GPU-only waits |
| Validation: new known-loss maximum, 232 tasks | July 8 19:15 | 3.358766 | 712 | 39.598% | 99.9994% | 712 GPU-only waits |

The Validation backlog maximum follows its rolling GPU pressure maximum by
5 hours 15 minutes. This is consistent with accumulated nonpreemptive GPU demand
and insufficient release/service capacity, but does not isolate a unique causal
lag. Its largest first-known-loss count actually occurs 15 minutes **before**
the GPU pressure maximum. Train's backlog maximum precedes its GPU pressure
maximum by 5 hours 15 minutes and coincides with CPU saturation instead. Using
one GPU-only scalar to characterize both windows would miss this resource shift.

Using utilization >= 0.99 as an explicit diagnostic threshold, Train 2× has
17 CPU-near-full and 17 GPU-near-full intervals; Validation 2× has 7 and 130,
respectively. Memory has zero near-full intervals. These counts include each
cohort's actual public observation window (Train 57 hours, Validation 55.75 hours,
including idle tails). They are descriptive usage, not the same quantity as
arrival work / 48-hour capacity.

## Development implication and boundary

EDF supports a 5% empirical reference at Train 1.5× (1.840%) and Validation 1×
(4.628%), both with actual capacity waiting. It does not support that reference
at Train 2× or Validation 1.5×/2×. This evidence identifies candidate development
pressure layers, not a guarantee for a new scheduler or an independent Test
claim. The frozen E1-5B 5% eligibility outcome remains unchanged.

A task-level risk budget should match each candidate's resource demand to the
actual resource competition state: CPU pressure must matter for CPU-heavy tasks
at the Train queue peak, and GPU pressure for GPU-heavy tasks in Validation.
Using the maximum pressure across all resources for every task is a concrete
future ablation, rather than an assumed correct design. The present evidence
does not implement or prove a new risk-budget formula.

At 2× every original task is copied once at the same arrival time. Different
stress seeds alter copy IDs/hash ordering and possibly EDF ties, but not the
aggregate arrival/resource/duration/deadline shape. They are not independent
real arrival samples. Fractional same-seed loads are nested hash-selected copies
of the same trace and also share that limitation.

## Files, hashes and replay

`analyze_pressure_evidence.py` reads only the sealed Train/Validation diagnostics
and baseline ledger/events, checks event counts against the EDF audit, saves six
exclusive-mask rows and 18 predetermined timing rows, and records SHA256 for
every input and output CSV in `pressure-evidence-audit.json`. The source module
freeze was unchanged by this post-processing. No scheduler or physics is run.

```powershell
& 'D:/anaconda/python.exe' -X utf8 -B research_notes/E1/E1-5C/analyze_pressure_evidence.py --root 'D:/A-myprofile/SustainCluster/upstream' --diagnostics 'D:/A-myprofile/SustainCluster/upstream/research_notes/E1/E1-5C/diagnostics-001' --output 'D:/A-myprofile/SustainCluster/upstream/research_notes/E1/E1-5C/pressure-replay-NEW'
```

The successful run is preserved in `pressure-evidence-run-001.txt`. Its first
pre-output helper invocation found an incorrect relative-path root for the new
Validation cohorts and exited without producing outputs; the helper root was
corrected and the complete six-cohort analysis then passed. Existing outputs
cause the script to stop, so replay requires a new output directory. The new
explicit root/diagnostics/output arguments also support this script when stored
in the publication docs directory. The completed pressure-replay-002 reproduces
both CSVs and the evidence JSON byte for byte; it never overwrites archived evidence.
