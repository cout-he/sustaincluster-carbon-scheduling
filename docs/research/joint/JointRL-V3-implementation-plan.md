# RAC-MAPPO V3 implementation and execution ledger

**Goal:** Implement and audit the user-specified causal risk/carbon scheduler, train under a frozen finite budget, and report actual evidence including failures.

**Spec:** The two user-supplied JointRL-V3/RAC-MAPPO task documents dated 2026-10-17 are the binding design. The user explicitly requested continuous implementation without intermediate design or module approval. This supersedes optional skill approval gates.

**Architecture:** Extend V2Env over the unchanged LocalExecutor and joint_full_hvac_v1 model. Use event/grid EDF reservation trajectories for both risk and carbon counterfactuals; fit and calibrate a small CI predictor on Train only. Separate Task40/Cooling17 actors use masked joint-slot PPO and centralized queue-pooled carbon/deadline critics at their respective decision boundaries. Retain waiting, thermal and terminal liability as an explicit auxiliary cost channel.

**Global constraints:** No future tasks/real future CI online; no September Test payload access; no changes to V1/V2/E1 or physical parameters. Every trajectory settles the same known jobs or invalidates positive-delay guidance. Six positive-floor group multipliers. Three seeds 20261017/18/19, 65536 physical steps and at most 21600 seconds per seed, complete episode commits. Large artifacts local only. No adaptive experiment expansion or hyperparameter search.

**Review focus:** Future-arrival rejection; nonpreemptive rounded releases versus actual finish/deadline; counterfactual work beyond H; empty/forced-action joint likelihood; resume identity/RNG/transaction boundaries; role guard before payload reads.

## Tasks and acceptance

- [x] Audit clean research checkout and remote V2 HEAD; create research/rac-mappo-v3.
- [x] A: Add reservation.py, risk_estimator.py, action_shield.py with tests for queue competition, future rejection, safe budget and legal fallback.
- [x] B: Add ci_forecaster.py, uncertainty.py, carbon_counterfactual.py with Train-only temporal calibration, correlated scenarios, full-work/tail tests and unchanged-model checks.
- [x] C: Add env.py, teacher.py with Task40 compatibility, per-slot causal histories, post-task global observation and estimator diagnostics; real plant integration tests.
- [x] D: Add critics.py, mappo.py, reward.py; grouped constraints, dual GAE and single combined scaling; joint log probability and forced-gradient tests.
- [x] E: Add config/pipeline/train/evaluate/monitor; frozen identity, atomic episode checkpoints, exact resume, model selection and protected Test entry; end-to-end smoke.
- [x] F: Run regression and independent review, freeze source/config/evaluation before formal execution.
- [ ] G: Run finite formal seeds and separately trained ablations, Development selection and B0–B5; report true progress if execution cannot finish.
- [ ] H: Generate method/training/results documents and real-data figures; commit and push source plus lightweight evidence.

## Decisions and evidence

- 2026-10-17: Actual checkout is publication/sustaincluster-carbon-scheduling-001. upstream is a dirty E1 checkout and is not modified.
- Remote ls-remote confirms V2 bf8a35ef647f19b5ef10ce1b99f723370b7a0732; no V3 remote branch existed at audit.
- Work in the clean explicitly requested research branch in place; a second checkout is unnecessary and would duplicate large local scenario dependencies.
- Baseline APIs: V2 preserves Task35, Cooling17, complete accounting and deterministic idle18 settling. V3 inherits these semantics and changes no historical source.
- Engineering acceptance: two complete Train episodes, 241 physical slots, 7892 tasks complete, no Deadline or temperature misses. Never count these as formal budget or scientific comparison.
- Full repository discovery also crawls historical experiment snapshots and fails duplicate-module collection; use explicit research test directories. The first research run had missing legacy OR-Tools and a live-edit module reload mismatch. Retesting with existing solver-runtime passed25 offline tests; final core regression passed158 tests, and the final pipeline/report change passed9 targeted tests. No failure is hidden as a full-repository green result.
- Independent review covered causal snapshots, capacity/rounded finish clocks, dual-cost PPO, forced gradients, grouped multipliers and transaction rollback. Ex-post mechanism labels use physically observed CI/weather after done, and never enter actor decisions or gradient labels.
- B0/B1 retain their stated fixed22/fixed18 control for the entire fixed window. B2 has no V2 Task reservation guard, ensuring actual EDF immediate execution. Historical V2 weight bytes are never modified.
- Frozen formal-001 has18 jobs (full+five separately trained ablations, three fixed seeds each), max3 concurrent CPU processes. Every job uses the same65536-slot target/21600-second maximum. No adaptive budget expansion or Test dispatch exists.
