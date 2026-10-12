# E1-5A: RL environment contract

This stage validates a mode-selection environment, without training a policy or claiming a learned task-level safe-delay budget. `experiments/e1/rl_env.py` implements Gymnasium `CarbonSchedulingEnv`; `engine.py` contains its sole execution kernel. The upstream physical formulas and existing policy classes are unchanged. Schema: `e1-5a-mode-env-v1`.

## Interaction and execution

Construct with `(workload, environment, capacity, power_factory, parameters, *, reward_config=None, normalization=None, max_episode_steps=None, stop_at_drain=False)`. `power_factory()` must create a fresh model for each reset. The same model, pending/running queues, Task instances and resource pool persist across every step. `reset(seed=..., options=...)` returns `(observation, info)`; supported options are `max_episode_steps` and `stop_at_drain`. `step(action)` returns `(observation, float_reward, terminated, truncated, info)`; stepping a finished or closed environment requires another reset. Repeated `close()` is safe.

At a ready boundary the simulator has registered arrivals, released completed tasks and enqueued arrivals, but has not sorted, made policy decisions, started tasks or called physics. One action then runs exactly one 15-minute interval: order all pending tasks, notify observed state, capacity-first check, decide only fitting candidates, allocate immediately, backfill, assert conservation, measure persistent IT/HVAC at the resulting resource fractions and real corresponding environmental sample, integrate energy/carbon, and log. The next boundary releases and observes newly arrived information before returning the next observation. At an exclusive final boundary only completions are flushed: no arrivals, starts or extra power interval.

The shared iterator is the explicit change from upstream E1 batch behavior: the original loop can now suspend before each policy decision. `run()` drives that same iterator. A running task is never preempted by a mode change. Both Carbon/EDF and RiskOnly histories continuously observe arrivals across mode switches; original selected-policy hooks deduplicate these notifications. Zero-interval `result()` is explicitly invalid. A partial cutoff before arrival-end cannot mark the finite cohort drained or permit idle extension.

## Actions

`Discrete(3)` selects one mode for every candidate of the current step, using the existing implementations with the same EDF order, deadline, capacity and backfilling semantics:

| Action | Existing policy | Meaning |
|---|---|---|
| 0 | `edf_backfill` | EDF immediate, with backfilling |
| 1 | `edf_carbon_threshold` | EDF plus fixed Carbon-Threshold eligibility |
| 2 | `cars_risk_only` | EDF plus original 120-minute arrival-pressure risk buffer |

The interval's first allocation immediately reduces availability for later candidates. Thresholds are supplied before an episode; `future_ci=True` and a history window other than 120 minutes are rejected. No Shadow Planner or queue-cap fallback is added. Legacy equivalence uses the sealed float `137.64999999999998`; the proposed new Train/Validation/Test split uses earlier calibration `161.075`. These roles are deliberately distinct.

## Observation

`Box(-1, 1, shape=(40,), dtype=float32)` has the ordered fields below. Let `B(x;s)=x/(s+abs(x))`, `C_r` be resource capacity, `d_occ=ceil(duration_minutes/15)*15`. Default engineering scales are CI 1000 gCO2e/kWh, task count 1000, slack/work 120 min, elapsed 24 h. They are not fitted to held-out data. Numeric configuration is validated finite and canonicalized to float.

| Index | Fields | Raw unit and normalization | Empty/cold handling |
|---|---|---|---|
| 0 | current CI | gCO2e/kWh; `B(CI;1000)` | current observed CI required |
| 1–4 | CI at previous 1–4 observed steps | gCO2e/kWh; same transform | pad unavailable lags with first observed CI |
| 5 | valid lag fraction | available previous lags / 4 | 0 at reset |
| 6 | current CI minus lag1 | gCO2e/kWh; `B(delta;1000)` | 0 until a previous CI exists |
| 7–9 | CPU/GPU/MEM utilization | occupied / capacity | 0 when idle |
| 10–11 | running/pending counts | tasks; `B(count;1000)` | 0 |
| 12–14 | observed arrival CPU/GPU/MEM work | core-, GPU-, memory-unit-min; `B(work/(C_r*120);1)` | no arrivals: 0 |
| 15–17 | arrival-pressure rho per resource | work / (capacity * effective history min); `B(rho;1)` | unknown cold rho encoded 0; index19 distinguishes cold |
| 18 | effective history fraction | elapsed observed history capped at120 /120 | 0 at reset |
| 19 | cold-history flag | 1 iff rho unknown | 1 at reset |
| 20 | RiskOnly buffer | buffer steps /2; possible0,0.5,1 | original cold buffer2 |
| 21–24 | pending solo latest-start slack min/mean/p10/max | `latest_start-now` minutes; `B(slack;120)` | all0 for empty queue; negative overdue slack retained |
| 25 | urgent pending count | tasks with `now + 15*(1+buffer) > latest_start`; `B(count;1000)` | 0 |
| 26 | deadline-due pending count | tasks with `deadline<=now`; `B(count;1000)` | 0 |
| 27 | solo latest-start-passed count | tasks with `latest_start<now`; `B(count;1000)` | 0 |
| 28–30 | sum pending CPU/GPU/MEM demands | demand /capacity; `B(ratio;1)` | 0 |
| 31–33 | pending CPU/GPU/MEM work | sum demand*d_occ; `B(work/(C_r*120);1)` | 0 |
| 34–35 | UTC daily sin/cos | known clock minute/1440 *2pi | [-1,1] |
| 36–37 | UTC weekday sin/cos | known weekday/7 *2pi | [-1,1] |
| 38 | elapsed observation time | hours; `B(hours;24)` | 0 at reset |
| 39 | configured arrival window open | `now<arrival_end` | 0 in Drain |

Nonnegative bounded fields lie in [0,1); signed slack/delta in (-1,1); flags, utilization and history fractions in [0,1]. Floating-point rounding may reach the Box endpoints. No full-cohort counts, future arrivals, future CI, workload/hash contents or other policies' outcomes appear in the vector or agent-facing info. `observation_raw` exposes only the same observed inputs for diagnosis. The simulator necessarily holds frozen future data privately; policy hooks and agent info do not receive it. Declared duration and demands are assumed known at arrival, as in the original experiment.

CI is current/past as-of observed timestamps. Weather interpolation remains an offline physical-accounting input and is absent from observations and decisions. Terminal observations retain the last allowed CI rather than reading the next environmental sample. A fork test with identical observed history and different future workload/CI/weather pins equal current observations, action decisions and interval integrals; the next observation can differ when the fork actually arrives.

## Reward

For interval `[t,t+15min)` with post-decision pending count `q_t`, and boundary `t+15min` after releases:

`r = -(alpha * carbon_kg/1000 + beta * (15*q_t)/15000 + gamma * new_miss_count/1000 + terminal_weight * liability)`

Defaults are alpha=1, beta=1, gamma=10, terminal_weight=10. All divisors and weights are configurable; terminal weight and divisors must be strictly positive. These constants were not tuned on outcomes. Actual energy is `IT_kW*0.25`, `HVAC_kW*0.25`, total their sum in kWh; actual carbon is total_kWh * CI_g_per_kWh /1000 in kgCO2e. Transmission metrics are unused.

One task receives its deadline component only once per episode: either it completes late, or it remains unfinished at a boundary with `deadline<=boundary`. A task that finishes exactly at its deadline is met even if its resources are released at the next boundary. Later completion cannot repeat an earlier miss penalty. No future completion outcomes are consulted for current sensors.

At truncation only, `liability = arrived_unfinished_count/1000 + max_r(remaining_resource_work_r/(C_r*120min))`. Pending tasks owe full occupied duration; running tasks owe occupied remaining declared duration after the final release. This includes right-censored tasks, so their unfinished work has nonzero terminal cost. Deadline and backlog components intentionally measure different liabilities. Future tasks not yet arrived are neither counted nor penalized, and the policy cannot choose the configured cutoff. Record remaining resource work in core-/GPU-/memory-unit-min, not as energy.

Each `info['reward']` records raw carbon, task-minutes, newly missed IDs/count, normalized components, remaining work, unfinished count, weighted costs and terminal liability. Validation trajectories save those terms. Nonzero backlog cost closes the free-truncation accounting loophole; it does not prove every configurable weighting prevents reward exploitation. Formal comparisons must still audit completed work, missed/censored tasks and full common-window energy/carbon; return alone is insufficient.

## End conditions and determinism

With `stop_at_drain=False`, even a completed cohort continues idle physics through the specified common endpoint. With `stop_at_drain=True`, a complete finite cohort stops at its first drain boundary. `terminated=True` only when execution has stopped, arrival-end has passed and both queues are empty. A training limit or exhausted coverage with incomplete cohort produces `truncated=True`, arrived-only final ledger, missed/censored classification and positive liability when arrived unfinished work remains. Before arrival-end an already-arrived prefix may be complete while the cohort remains truncated; its backlog liability is zero and future tasks remain uncounted. `info['final_summary']` retains raw scientific metrics and `termination_reason` states completion versus episode/coverage limit.

Reset recreates policies, history, reward counted IDs and one fresh physical object. Seed controls Gymnasium action sampling and local RNG; frozen inputs have no stochastic regeneration. The physical reset seed remains the original fixed seed, which is not an independent experimental replication. The seeded random rollout is an API/conservation smoke test, with its own drain endpoint, and is not compared as a carbon-performance benchmark.

## Reproduction

`python -m pytest -q tests/e1 -p no:cacheprovider`

`python -m experiments.e1.rl_validation --e1-3 research_notes/E1/E1-3/results-001 --e1-4a research_notes/E1/E1-4A/results-001 --output research_notes/E1/E1-5A/results-001 --seed 20261012`

Use a new output directory for a repeat. Real-data historical validation requires the private complete sealed archives and their original non-kernel source bytes/dependencies; they are intentionally absent from GitHub. Public source and synthetic behavior tests can run after installing the documented Python environment. Old strict replay still requires its old source snapshot: the new independent driver verifies archive integrity and allows only the declared kernel change/new modules before requiring exact historical numerical output. It does not rewrite old provenance or relax old replay tools.
