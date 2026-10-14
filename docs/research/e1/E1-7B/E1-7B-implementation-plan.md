# CGD-RL V1 implementation plan

**Goal:** Implement an online full-interval IT+HVAC gain estimator, common deadline guard, fixed policies and a trainable independent Gym interface; validate without formal PPO training.

**Spec:** E1-7B user attachment, SHA recorded in E1-7B-code-adaptation.md. The user explicitly requests continuous implementation after audit, so no repeated design approval is needed.

**Architecture:** New experimental modules reuse the sole LocalExecutor and unchanged physical model. Separate Train-fitted prediction from actual scientific accounting and separately labelled offline paired diagnostics.

**Constraints:** capacity 50000/1000/80000; 15min; demand multiplier5; SLA1.5; CI threshold161.075; nonpreemption; RiskOnly120min/cold2/0,1,2; no Test; no formal PPO; no old archives/checkpoints/main changes.

- [ ] Capture old files/source/Git protection metadata before product edits; preserve original dirty state.
- [ ] Forecast/guard: failing tests, causal rolling CI inclusive horizons and legal prehistory, exact old pressure mapping, tests pass.
- [ ] Surrogate/gain: failing tests, bounded Train-only models and complete occupancy two-branch incremental IT/HVAC, causal past power/weather; tests pass.
- [ ] Policy/kernel: failing tests, two optional read-only hooks, common EDF guard, M0/M1/M2 and candidate diagnostics; legacy numerical path tests.
- [ ] RL: failing tests, independent schema/base40+small gain block, three gain-threshold actions, reuse reward, fixed/random/SB3 smoke verification.
- [ ] Experiment driver: freeze registered protocol and input identities, fit Train model and thresholds, then full Train 1/1.5/2 and development Validation 1/1.5/2; complete Drain/common idle, exact historical controls and frozen replay.
- [ ] Analyse CI/model prediction errors separately from actual scheduling; reuse all 12 preselected E1-7A pairs without selecting successes; six figures with small underlying CSV.
- [ ] Run old/new regressions, independent review, verify original protection, curate release into clean publication branch, test applicable publication suite, make two commits and normal push, verify SHA/main/clean status.

## Review focus

No future samples cached online; current interval power is unavailable before measurement; background is identical across branches and updated after each real start; >horizon tasks execute with explicit unknown status; moderate threshold comes only from Train and M1/M2 share it; forecast/power errors and varying state trajectories prevent per-event causal claims from whole-policy results.
