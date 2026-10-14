# E1-7A Implementation Plan

> For agentic workers: execute independent tasks with subagents and tests first; preserve the explicit user scope and protected history.

**Goal:** Diagnose finite scheduling headroom, real carbon opportunities and a single defer's incremental effect.

**Architecture:** Independent analysis modules and policy wrappers over the unchanged LocalExecutor. Offline solver and clairvoyant CI analysis never enter online policy state. All scientific outputs use new E1-7A paths.

**Tech Stack:** Existing Python3.10 E1 physics/runtime; isolated OR-Tools9.14.6206; pytest; matplotlib PNG/PDF with CSV.

**Spec:** User E1-7A pasted request and E1-7A-code-audit.md.

## Global constraints

- Train2023-05-01 05:00→May3 05:00 UTC only; existing factors1/1.5/2, stress seed510001, immutableTaskSpec.
- Threshold161.075, SLA1.5, resource multiplier5, capacities50000/1000/80000, 15min unchanged.
- No old source/test/physics/PPO/checkpoint edits; no Test/Validation selection; no old archive overwrite.
- Max120s/case,1 solverworker,seed710001; max40 selected real tasks/case; retain all solverstatuses.
- C target12pairs, can-fit/current solo≥15, deterministic selection before outcomes; EDF thereafter.

## Steps and review focus

- [ ] A0 capture Git/source and historical metadata protection; bind3Train manifests and9historical baseline scientific outputs; audit papers.
- [ ] Freeze config and selection algorithm before solver/pair outcomes. Save all eligible snapshot/target metrics and selection manifest.
- [ ] A1 TDD: task/arrival/actualfinish/release/grid/background/int64/scaling/all-complete/extremelysmall exhaustive equivalence; source three offline_schedule modules and threetests.
- [ ] A2 Execute selected A1/A2 bounded cases, verify raw-resource legality, replay exactschedule with LocalExecutor; keep conditionalEDF distinction explicit.
- [ ] B TDD: IT/HVAC/total kWh and carbon identities, idle bins, current/hour CI structure, execution-window proxy; full real physics replay9Train baselines and legal alternative schedules.
- [ ] C TDD: oneoverride, same-prefix includingpersistentpower, capacity-first, strict online history, task impacts/firstmiss;3counterfactual modules/tests. Onepair gate then all frozenpairs, completeDrain/commonidle.
- [ ] Joint report: descriptive results, failures/zeros, riskcounterexamples and one priority; no forecast/SLA/optimality overclaims.
- [ ] Old key/full E1 regression and publication applicable tests, evidencehashes, preservation; synchronize explicit curatedallowlist; two authorizedcommits/normalpush only publicationresearch.

Risks pinned by tests: binaryfloats/halfULP; between-grid deadline versus release; fixedbackground doublecount; candidate popped from deque at intervention; unknown power-state types fail rather than guess; no Test payload parsing; commonidle beyond ownDrain; no after-outcome selection.
