# CARS-v1 pseudocode

```
on_step_start(observed immutable pending/running/arrivals/capacity snapshot):
    record unseen arrivals; expire arrival_time <= now - 120 min
    rho = max(resource_work / (capacity * W_eff))
    buffer = 2 when W_eff == 0 else bins(rho: .5, 1 -> 0, 1, 2)
    retain current EDF order, real running finishes, empty same-step blocked set

for every pending candidate in stable EDF order, including old backlog:
    if public executor cannot fit CPU/GPU/MEM: capacity_wait (no policy decision)
    else:
        refresh current available resources from public executor
        if current CI <= fixed threshold: execute [low_ci_execute]
        elif now + STEP > deadline - duration: execute [no_grid_slack]
        elif risk_enabled and now + STEP + buffer * STEP > latest_start:
            execute [risk_buffer_guard]
        elif known_enabled:
            compare A: candidate forced now, B: barred until next STEP
            retain earlier actual deferrals in both plans
            simulate only known nonpreemptive jobs using EDF + backfilling
            preserve CPU/GPU/MEM until ceil(exact_finish, global 15 min)
            if unresolved/cap exceeded: execute [known_feasibility_uncertain]
            elif M_defer - M_now nonempty: execute [known_deadline_risk]
            else: defer one step [high_ci_defer]
        else: defer one step [high_ci_defer]
    public executor starts/defer/waits, owns resource and task state
    notify policy of true starts or issued deferrals before next candidate
integrate unchanged IT/HVAC model for resulting interval load
```

All decisions re-evaluate at the next boundary. The history buffer neither reserves nor subtracts capacity. The shadow planner neither executes tasks nor reads future trace/environment samples.
