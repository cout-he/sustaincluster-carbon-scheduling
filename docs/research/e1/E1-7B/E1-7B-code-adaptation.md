# E1-7B code adaptation audit (before implementation)

Version: e1-7b-cgd-rl-v1. User specification SHA256: 7338c542005eff3d656a8554d2bf4f5b72773d264017d7b2624752cfb5b90854.

## Protected starting state

Original workspace is on research/e1-carbon-deadline, HEAD and main both 3f6ea95cb835b89ba50b0ef76d66d14b8037643e. Its modified M0 notebook and all pre-existing untracked research files stay in place. Publication is clean on the research branch at 50b6d8bc8709f4c8c310b7b04bc787dd405a2c8e; main remains 749dc6c1868065c12a71bac71b3ae08a25c45f56. Only the publication copy is committed/pushed.

## Sole execution path

engine.LocalExecutor.start prepares _execute and suspends before policy/physics. Each interval records arrivals, releases completions, enqueues arrivals, gives all unstarted tasks to policy.order, and checks capacity before calling decide. ResourceSnapshot exposes only arrived immutable TaskSpec, currently running tasks, and current capacity. Allocation immediately changes later candidates' capacity; on_task_started informs observer state. Backfilling is a single EDF traversal. power.UpstreamPowerModel.measure supplies current load/weather before the unchanged SustainDC step. metrics.interval_record integrates kW * .25 h, then kWh * gCO2e/kWh /1000. Exact task finish decides SLA; ceil-to-15min release controls occupied resources. The exclusive endpoint only flushes completion. Drain introduces no new tasks; extend_idle keeps the same physics alive to the group's maximum drain boundary.

## Existing online/offline boundaries

cars_state.ArrivalHistory implements the frozen RiskOnly 120min resource-work history, max-resource rho, buffers 0/1/2 and cold=2. policies.latest_start preserves floating duration. workload.TaskSpec and FrozenWorkload are immutable. selection.TemporalEnvironment uses the preceding hourly CI observation on the 15min grid. rl_env owns the tested 40-dimensional base observation and real carbon/wait/first-miss/truncation reward; adaptive_rl_env preserves its separate 83-dimensional interface. E1-7A carbon_opportunity is explicitly clairvoyant and must never be imported by the online estimator/policy/environment. counterfactual runs full physical alternatives and supplies only separately labelled offline evaluation evidence.

## Minimal compatible adaptation

Add optional read-only current-environment and completed-interval notifications to LocalExecutor. They expose scalar current CI/weather and a copied completed interval record respectively, never the frozen future calendar or mutable pool. Legacy policies lack these methods and retain their numerical path. The completed hook runs only after real measure/integration; at decision time the estimator can only hold earlier completed power. No production scheduling or physical equation changes.

New modules separate causal CI forecast, Train-only power surrogate, incremental full-occupancy carbon gain, unchanged RiskOnly-style deadline guard, EDF gain policy, independent Gym environment, and experiment/replay driver. CI forecast returns inclusive interval starts now + k*STEP; at most 96 samples, all historical sources <= now. Known-running release profiles are the shared background; other pending starts and unknown arrivals are omitted in both branches. A target occupies ceil(duration/15) slots. Both branches use the same horizon and current-weather persistence; unknown horizon or unavailable estimate executes immediately.

The upstream HVAC code computes current nonlinear cooling power; the persistent object alone is not evidence of thermal inertia. The V1 HVAC surrogate therefore uses a bounded static response (dynamic coefficient a=0, fixed tail=0), fitted exclusively to Train records. Its predictive and incremental errors are reported. It never calls real measure for hypothetical candidates. M1 and M2 share CI, IT, background, horizon, deadline protection and thresholds; only HVAC inclusion differs.

## Required verification

Tests precede each new module. Assert forecast provenance/cold boundaries; incremental units/full duration/common horizon; synthetic IT/HVAC sign reversal; precise deadline guard; capacity-first EDF/backfill; same-step resource updates; future-trajectory fork invariance; legacy reward/action/checkpoint interfaces; deterministic fixed and random rollouts; minimal SB3 updates. Re-run old E1 tests and byte-exact scientific baseline comparisons. Freeze configuration, surrogate parameters and Train-derived moderate threshold before any development Validation. Test remains sealed. Preserve failed runs and negative scientific results; original archives are never rewritten.
