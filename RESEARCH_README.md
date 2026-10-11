# SustainCluster Carbon Scheduling — E1 research

This research branch builds on [HewlettPackard/sustain-cluster](https://github.com/HewlettPackard/sustain-cluster), baseline `3f6ea95cb835b89ba50b0ef76d66d14b8037643e`. The original [MIT license](LICENSE), copyright and [upstream README](UPSTREAM_README.md) are retained. Research code is in [experiments/e1](experiments/e1/README.md); production scheduling and physical equations are unchanged.

Completed stages: [E1-0 audit](docs/research/e1/E1-0.md), [E1-1A execution foundation](docs/research/e1/E1-1A.md), [E1-1B baselines](docs/research/e1/E1-1B.md), and [E1-2A temporal shifting validation](docs/research/e1/E1-2A.md). No new RL method, training or large parameter scan is included.

The three policies share a single executor, task specifications, resources, backfilling and IT/HVAC model. In the E1-2A finite cohort all policies completed 9943 tasks with zero synthetic-deadline violations. Over a common 56.75h accounting interval including idle tails, Carbon-Threshold emitted 97.013 kgCO2e (0.742%) less than FIFO/EDF, with 21.799min more mean waiting and a 3h longer drain. This one-window model result does not establish general superiority or steady-state benefits.

See [reproduction and data boundaries](docs/research/e1/REPRODUCIBILITY.md), [tested environment](docs/research/e1/environment-tested.json) and [version workflow](docs/research/e1/VERSION_CONTROL.md). Full local experiment archives, task rows, raw datasets and weights are excluded from new research commits. Summary CSVs and hashes are published separately from local archival evidence. This publication has a new curated-source history and excludes the original data/weight objects. The original upstream SHA is attribution, not this repository main SHA. See README.md for external data setup.

From the repository root, in the compatible Python environment:

```bash
python -m pytest tests/e1 -q
python -m experiments.e1.run --config "$E1_LOCAL_CONFIG" --output /outside-git/new-fifo-001
python -m experiments.e1.temporal --config "$E1_LOCAL_CONFIG" --output /outside-git/new-temporal-001
python -m experiments.e1.temporal --replay /outside-git/new-temporal-001 --output /outside-git/new-temporal-replay-001
python -m experiments.e1.plot_temporal --experiment /outside-git/new-temporal-001 --output /outside-git/new-temporal-figures-001
```

Real-data runs require the trusted workload pickle, CI and weather specified by the configuration. Unit tests generate synthetic inputs; the optional local historical-artifact regression skips when that archive is absent. Never overwrite existing experiment directories. Old archived runs require their recorded source bytes and dependencies; publishing Git commits does not retroactively change their provenance.

Before real-data commands use an outside-Git YAML with external absolute input paths as described in [publication README](README.md). Historical108testpasses refer to the private original checkout, not a new clean-install claim.

## E1-4A: CARS feasibility and cap fallback diagnosis

The [E1-4A report](docs/research/e1/E1-4A/E1-4A-report.md) isolates `risk_cap_only` and `risk_shadow_only` without changing the original executor, CARS policies or power equations. Both 1.5× and 2.0× frozen E1-3 cohorts completed under five policies with a common drain/idle accounting endpoint. Shadow-only changed deadline misses by +7/−10 relative to RiskOnly; cap-only by 0/−9. Full remained worse than cap-only by 35/25 misses. These already-seen-window diagnostics do not establish generalization or that Shadow is always ineffective.

See [common metrics](docs/research/e1/E1-4A/common_metrics.csv), [guard definitions and counts](docs/research/e1/E1-4A/guard_counts.csv), [reproduction commands and archive boundaries](docs/research/e1/E1-4A/E1-4A-reproducibility.md), and [validation](docs/research/e1/E1-4A/validation.md). Original research tests: 225 passed; publication tests: 223 passed, 2 private-archive-dependent tests skipped. Strict replay of 231 scientific outputs and byte regression of 72 old-policy outputs were verified in the original research environment. Historical input archives and original source bytes remain external requirements; this curated clone is not a self-contained historical replay bundle.
