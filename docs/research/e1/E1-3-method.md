# E1-3 CARS-v1 method contract

This is an online heuristic for a single aggregate, nonpreemptive CPU/GPU/MEM pool. It neither proves that a feasible schedule exists nor guarantees SLA. Task duration and resource demands are immutable. The common executor alone admits jobs, releases capacity, and integrates upstream IT/HVAC power.

All four new strategies use stable EDF `(deadline, arrival, task_id)` and one-pass backfilling. `edf_carbon_threshold` isolates ordering; `cars_known_only` adds the shadow comparison; `cars_risk_only` adds the history buffer; `cars_full` combines them. The three historical policies and their registry remain unchanged.

## Predeclared rules

CI threshold is inherited from the disjoint E1-2A calibration: 137.65 gCO2e/kWh (archive float representation 137.64999999999998). Decision step is 15 min. High CI means strictly greater than the threshold. Exact theoretical latest start is `deadline-duration`; grid latest start is its floor on the global UTC 15 min grid. No future CI or arrivals are supplied to a policy.

History window is `(now-120min,now]`. Each observed arrival contributes fixed demand times `ceil(duration/15min)*15min` resource-minutes, once. The denominator uses `min(120min, now-experiment_start)`. At the initial boundary it is zero: rho is undefined, serialized as null, and the declared cold-start buffer is two steps. There is no invented prehistory. Subsequently rho is the maximum of three resource work/capacity/window ratios; values below .5 give zero buffer, [.5,1) one step, and >=1 two steps. These are design assumptions, not calibrated probabilities. Drain slides the same history window. Delay needs `now+STEP+buffer*STEP <= latest_start`.

The known-task planner checks all current pending tasks up to a declared cap of 256, plus running reservations. It uses eight decision boundaries from now through now+105min and a terminal boundary now+120min. Scenario A forces the candidate to start now. Scenario B bars it until the next step. Both use the same EDF/backfill, fixed running reservations and three-resource limits. Earlier deferrals committed during this actual decision step remain barred until the next step in both scenarios; this does not reconsider an action already issued. Started tasks use precise predicted finish for deadlines and its ceiling grid for resource release, including reservations beyond the planner horizon.

For the specified forward plans, compare missed-ID sets rather than counts. Any newly missed ID in `M_defer-M_now` blocks deferral. Shared missed IDs are recorded as known pressure under these plans, not proof of global infeasibility. Already-started jobs can have known predicted outcomes after the horizon. Unscheduled jobs whose latest start is before the terminal boundary are missed under that plan; other unscheduled jobs are unknown. Any unresolved outcome, or a queue over the cap, yields uncertain and conservative execution. No wall-clock timeout changes actions.

Cached immutable job conversion and shared step state reduce repeated work; real starts and earlier deferrals invalidate state-dependent planning. Planner timing is kept separately from deterministic scientific outputs. Planner calculations cannot call the power model or mutate real tasks/resources.

## Source ideas and differences

[Chronus](https://yezhisheng.com.cn/publication/chronus/chronus_preprint.pdf) motivates considering deadlines jointly with resources; its MILP admission and preemptible leases are not implemented. [AFS](https://www.usenix.org/system/files/nsdi21-hwang.pdf) motivates conservative treatment of future resource shares; the historical work ratio here is our separate heuristic, not its elastic GPU sharing algorithm. [GREEN](https://www.usenix.org/system/files/nsdi25-xu-kaiqiang.pdf) motivates carbon/JCT tradeoffs under finite capacity; it assumes preemptible jobs and explicitly does not ensure individual SLOs. These connections do not establish novelty.

## Accounting and limits

All seven policies share each exact E1-2B frozen cohort and real corresponding CI/weather through drain. Common end is the maximum observed drain boundary, or the fixed safety cap if incomplete. Earlier drains integrate the same idle physical state to that end. Carbon is `(IT_kWh+HVAC_kWh)*CI_g_per_kWh/1000`, in kgCO2e. Fixed, own-drain and common windows are reported separately. Historical three-policy numerical regression uses their original common endpoint before seven-policy extension.

Limits include synthetic deadlines, aggregate/fractional resources, fixed durations, discretized release, one DC, noncontemporaneous trace mapping, and simultaneous synthetic clones. This finite-cohort experiment does not model a stationary continuously arriving deployment. No significance or general superiority is inferred from one window and one stress seed.


## Formal notation

For fixed task demand q_i=(CPU_i,GPU_i,MEM_i), arrival a_i, duration r_i and synthetic deadline d_i, actual finish is f_i=s_i+r_i and miss_i=1[f_i>d_i]. The public pool enforces Σ_running q_i,r≤C_r in each resource dimension. Fifteen-minute left-boundary holding keeps resources until ceil_grid(f_i); latest feasible global start is floor_grid(d_i−r_i).

Separate objectives are Carbon=Σ_t[(IT_kWh_t+HVAC_kWh_t)×CI_g_per_kWh_t/1000] and DMR=Σmiss_i/N_arrived, retaining censoring if incomplete. This version does not combine them into an arbitrary reward.

For r∈{CPU,GPU,MEM}, H_r(t)=Σ_{observed a_j∈(t−W,t]}q_j,r×ceil(r_j/STEP)×STEP_minutes; rho_r=H_r/(C_r×W_eff), rho=max_r rho_r. W=120min and W_eff=min(W,t−experiment_start). At W_eff=0, rho is undefined and the declared two-step cold-start buffer applies. B(t)=STEP×{0 if rho<0.5;1 if 0.5≤rho<1;2 otherwise}. Risk permits consideration of defer only when t+STEP+B(t)≤d_i−r_i; Known additionally requires no unknown and M_defer\M_now empty under the two finite plans. Neither permits future-information or physical reservation of that buffer.
