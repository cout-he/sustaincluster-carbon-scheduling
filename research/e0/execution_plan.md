# E0 execution plan

Goal: establish a trustworthy original SustainCluster simulation baseline.

- [x] Create independent checkout and research/rl-scheduling-v1 from exact main snapshot 749dc6c1868065c12a71bac71b3ae08a25c45f56.
- [x] Verify original source manifest against upstream 3f6ea95cb835b89ba50b0ef76d66d14b8037643e.
- [x] Restore and hash original runtime inputs from that upstream commit; keep all raw data ignored.
- [x] Prepare isolated local environment and record inherited dependency versions.
- [x] Execute original unit tests verbatim and retain failures; no training.
- [x] Execute reset/step, short simulation, rule controls through one observer harness; audit input/resource/energy/carbon consistency.
- [x] Write environment.md, code_audit.md, baseline_results.csv and E0-report.md from actual evidence.
- [x] Independently review evidence, check secrets/size and prepare publication only on research/rl-scheduling-v1. Commit/push outcome is verified against the remote SHA and reported in the final response.

Constraints: do not merge main or import research code/models/tasks; no ACP-PPO or new RL training. Core simulator remains unchanged. Original data acquisition example contains a pre-existing embedded credential; replace it with an external environment-variable requirement without changing simulation behavior.

Decisions: user provided an executable five-task scope, so proceed without an additional design approval gate. Use a fresh clone rather than modifying existing research checkouts. Original failing tests remain failures; no fabricated statistics or silent compatibility patches.
